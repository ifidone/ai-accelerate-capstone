"""LabBot MCP server.

Expose LabBot's deterministic equipment-management capabilities as MCP tools.

Run manually with:

    python -m app.mcp_server

This is an stdio MCP server. It is designed to be launched by an MCP-aware
host or by scripts/mcp_client_demo.py.

Important demo-mode authorization note:
- The current LabBot app uses a persona selector and X-User-Id header.
- MCP does not share that browser session, so tools explicitly accept
  `actor_user_id`.
- Every manager action verifies the actor's role in deterministic code.
- When Google login is added, replace actor_user_id input with the
  authenticated identity supplied by the MCP host/session.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from mcp.server.fastmcp import FastMCP

from . import calendar_client, gmail_client, store

mcp = FastMCP(
    "LabBot Equipment Checkout",
    instructions=(
        "LabBot manages shared lab hardware. Use availability tools before "
        "requesting equipment. Student actions are scoped to the supplied "
        "actor_user_id. Manager-only tools require an actor whose role is "
        "lab_manager. A checkout request becomes pending until a manager "
        "approves it."
    ),
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _require_user(actor_user_id: str) -> dict:
    """Load an existing LabBot user or raise a tool-visible error."""
    user = store.get_user(actor_user_id)

    if not user:
        raise ValueError(
            f"Unknown LabBot user '{actor_user_id}'. "
            "Use a valid user ID from the configured LabBot users."
        )

    return user


def _require_manager(actor_user_id: str) -> dict:
    """Load and authorize a Lab Manager in deterministic Python code."""
    user = _require_user(actor_user_id)

    if user.get("role") != "lab_manager":
        raise PermissionError(
            "This action requires a Lab Manager role."
        )

    return user


def _safe_calendar_create(
    item_name: str,
    due_date: str,
    checkout_id: str,
) -> dict:
    """Calendar failures must never roll back an approved checkout."""
    return calendar_client.create_due_date_event(
        item_name=item_name,
        due_date=due_date,
        checkout_id=checkout_id,
    )


def _safe_calendar_delete(event_id: str | None) -> dict:
    """Calendar failures must never undo a completed return."""
    return calendar_client.delete_event(event_id)


# ---------------------------------------------------------------------------
# Student-accessible tools
# ---------------------------------------------------------------------------
@mcp.tool(
    name="check_equipment_availability",
    description=(
        "Check whether lab equipment matching an item name, inventory ID, or "
        "category is available. Use this before requesting a checkout. "
        "Examples: 'oscilloscope', 'ESP32', 'sensor kit', or 'scope-01'."
    ),
)
def check_equipment_availability(query: str) -> dict[str, Any]:
    """Return matching available and unavailable inventory units."""
    term = query.strip()

    if not term:
        raise ValueError("Provide an item name, inventory ID, or category.")

    matches = store.find_items_by_category(term)

    # Exact inventory ID may not match a category/name substring.
    exact = store.find_item(term)
    if exact and all(item["item_id"] != exact["item_id"] for item in matches):
        matches.append(exact)

    return {
        "ok": True,
        "query": term,
        "available": [
            {
                "item_id": item["item_id"],
                "name": item["name"],
                "category": item.get("category", ""),
                "checkout_limit_days": item.get("checkout_limit_days"),
            }
            for item in matches
            if item.get("status") == "available"
        ],
        "unavailable": [
            {
                "item_id": item["item_id"],
                "name": item["name"],
                "category": item.get("category", ""),
                "status": item.get("status"),
            }
            for item in matches
            if item.get("status") != "available"
        ],
    }


@mcp.tool(
    name="request_equipment_checkout",
    description=(
        "Create a checkout request for the current student. The request is "
        "pending until a Lab Manager approves it. The item can be an inventory "
        "ID such as 'scope-01' or a natural item/category query such as "
        "'oscilloscope' or 'ESP32'."
    ),
)
def request_equipment_checkout(
    actor_user_id: str,
    item_query: str,
    requested_days: int | None = None,
) -> dict[str, Any]:
    """Create a pending equipment request for the named student."""
    _require_user(actor_user_id)

    if requested_days is not None and requested_days <= 0:
        raise ValueError("requested_days must be a positive whole number.")

    record = store.find_available_item(item_query.strip())

    if not record:
        return {
            "ok": False,
            "reason": f"Could not identify equipment matching '{item_query}'.",
        }

    result = store.request_checkout(
        user_id=actor_user_id,
        item_id=record["item_id"],
        days=requested_days or record.get("checkout_limit_days", 3),
    )

    return result


@mcp.tool(
    name="get_my_checkout_status",
    description=(
        "Return the supplied student's own pending, active, overdue, or denied "
        "checkout requests and whether they are on hold. Never use this tool "
        "to inspect another student's records."
    ),
)
def get_my_checkout_status(actor_user_id: str) -> dict[str, Any]:
    """Show the current student's own request and checkout status."""
    _require_user(actor_user_id)
    return store.my_status(actor_user_id)


@mcp.tool(
    name="get_checkout_due_status",
    description=(
        "Check the due date and time remaining for one of the supplied "
        "student's active checkouts. If the student has exactly one active "
        "item, item_query may be omitted."
    ),
)
def get_checkout_due_status(
    actor_user_id: str,
    item_query: str | None = None,
) -> dict[str, Any]:
    """Return due-date information for one active checkout."""
    _require_user(actor_user_id)

    item_id = None

    if item_query:
        record = store.find_user_checkout_item(
            actor_user_id,
            item_query.strip(),
        )

        if not record:
            return {
                "ok": False,
                "reason": (
                    f"Could not find '{item_query}' among this student's "
                    "active checkouts."
                ),
            }

        item_id = record["item_id"]

    return store.time_remaining(actor_user_id, item_id)


@mcp.tool(
    name="report_equipment_return",
    description=(
        "Record that one item currently checked out by the supplied student "
        "has been returned. This changes the checkout to returned, makes the "
        "inventory available again, and attempts to remove the associated "
        "calendar reminder."
    ),
)
def report_equipment_return(
    actor_user_id: str,
    item_query: str,
) -> dict[str, Any]:
    """Return one item belonging to the supplied student."""
    user = _require_user(actor_user_id)

    record = store.find_user_checkout_item(
        actor_user_id,
        item_query.strip(),
    )

    if not record:
        return {
            "ok": False,
            "reason": (
                f"Could not find '{item_query}' among "
                f"{user.get('name', actor_user_id)}'s active checkouts."
            ),
        }

    result = store.report_return(
        user_id=actor_user_id,
        item_id=record["item_id"],
    )

    if result.get("ok"):
        result["calendar"] = _safe_calendar_delete(
            result.get("calendar_event_id")
        )
        result["email"] = gmail_client.send_return_confirmation(
            user=user,
            item_name=record["name"],
            return_date=date.today().isoformat(),
        )

    return result


@mcp.tool(
    name="report_equipment_damage",
    description=(
        "Create a damage report for equipment currently checked out by the "
        "supplied student. Students can report an issue but cannot mark an "
        "item repaired, retired, or under repair; those are manager actions."
    ),
)
def report_equipment_damage(
    actor_user_id: str,
    item_query: str,
    description: str,
) -> dict[str, Any]:
    """Record a student damage report against their active checkout."""
    _require_user(actor_user_id)

    if not description.strip():
        raise ValueError("Provide a description of the damage or problem.")

    record = store.find_user_checkout_item(
        actor_user_id,
        item_query.strip(),
    )

    if not record:
        return {
            "ok": False,
            "reason": (
                "Damage can be reported only for equipment currently checked "
                "out by the supplied student."
            ),
        }

    return store.report_damage(
        user_id=actor_user_id,
        item_id=record["item_id"],
        description=description.strip(),
    )


# ---------------------------------------------------------------------------
# Manager-only tools
# ---------------------------------------------------------------------------
@mcp.tool(
    name="list_pending_checkout_requests",
    description=(
        "List all equipment checkout requests awaiting Lab Manager approval. "
        "Each request includes a checkout ID, item name, requesting student, "
        "and requested duration. Requires a Lab Manager actor."
    ),
)
def list_pending_checkout_requests(
    actor_user_id: str,
) -> dict[str, Any]:
    """List all pending checkout requests for a manager."""
    _require_manager(actor_user_id)

    return {
        "ok": True,
        "requests": store.pending_checkouts(),
    }


@mcp.tool(
    name="approve_or_deny_checkout_request",
    description=(
        "Approve or deny a pending equipment checkout request. Approval "
        "activates the checkout, reserves the inventory unit, creates a "
        "calendar reminder, and sends a confirmation email. Denial leaves "
        "the item available. Requires a Lab Manager actor."
    ),
)
def approve_or_deny_checkout_request(
    actor_user_id: str,
    checkout_id: str,
    decision: str,
    manager_note: str = "",
) -> dict[str, Any]:
    """Approve or deny a pending request and run approval side effects."""
    manager = _require_manager(actor_user_id)

    normalized_decision = decision.lower().strip()

    if normalized_decision not in {"approve", "deny"}:
        raise ValueError("decision must be exactly 'approve' or 'deny'.")

    # Preserve the original student ID for email delivery. Display summaries
    # intentionally hide internal IDs, so use the raw record internally.
    original_checkout = store.find_checkout(checkout_id)

    result = store.approve_or_deny(
        checkout_id=checkout_id,
        decision=normalized_decision,
        manager_id=manager["id"],
        manager_note=manager_note.strip(),
    )

    if not result.get("ok") or normalized_decision != "approve":
        return result

    approved_checkout = store.find_checkout(checkout_id)

    if not approved_checkout:
        result["calendar"] = {
            "ok": False,
            "reason": (
                "The request was approved, but LabBot could not reload the "
                "checkout for calendar and email follow-up."
            ),
        }
        return result

    item = store.find_item(approved_checkout["item_id"])
    student = store.get_user(approved_checkout["student_id"])

    item_name = (
        item["name"]
        if item
        else approved_checkout["item_id"]
    )

    calendar_result = _safe_calendar_create(
        item_name=item_name,
        due_date=approved_checkout["due_date"],
        checkout_id=approved_checkout["checkout_id"],
    )

    result["calendar"] = calendar_result

    if calendar_result.get("ok"):
        try:
            store.set_calendar_event_id(
                approved_checkout["checkout_id"],
                calendar_result.get("event_id"),
            )
        except Exception as exc:
            result["calendar"] = {
                "ok": False,
                "reason": (
                    "The calendar event may have been created, but LabBot "
                    f"could not save its event ID: {exc}"
                ),
            }

    result["email"] = gmail_client.send_checkout_confirmation(
        user=student,
        item_name=item_name,
        due_date=approved_checkout["due_date"],
        checkout_id=approved_checkout["checkout_id"],
    )

    return result


@mcp.tool(
    name="list_outstanding_lab_equipment",
    description=(
        "List all pending, active, and overdue equipment checkouts across the "
        "lab. Requires a Lab Manager actor."
    ),
)
def list_outstanding_lab_equipment(
    actor_user_id: str,
) -> dict[str, Any]:
    """List all outstanding lab equipment records."""
    _require_manager(actor_user_id)

    return {
        "ok": True,
        "checkouts": store.outstanding_checkouts(),
    }


@mcp.tool(
    name="list_overdue_lab_equipment",
    description=(
        "List all overdue equipment checkouts. This also performs the overdue "
        "status sweep that changes past-due active records to overdue. "
        "Requires a Lab Manager actor."
    ),
)
def list_overdue_lab_equipment(
    actor_user_id: str,
) -> dict[str, Any]:
    """List overdue equipment across the lab."""
    _require_manager(actor_user_id)

    return {
        "ok": True,
        "overdue": store.overdue_items(),
    }


@mcp.tool(
    name="send_overdue_checkout_nudge",
    description=(
        "Send an immediate email reminder to the student responsible for one "
        "overdue checkout. The checkout must already be marked overdue. "
        "Requires a Lab Manager actor."
    ),
)
def send_overdue_checkout_nudge(
    actor_user_id: str,
    checkout_id: str,
) -> dict[str, Any]:
    """Send a manager-triggered overdue email reminder."""
    _require_manager(actor_user_id)

    checkout = store.find_checkout(checkout_id)

    if not checkout:
        return {
            "ok": False,
            "reason": f"No checkout found with ID '{checkout_id}'.",
        }

    if checkout.get("status") != "overdue":
        return {
            "ok": False,
            "reason": (
                f"Checkout {checkout_id} is not overdue "
                f"(status: {checkout.get('status')})."
            ),
        }

    student = store.get_user(checkout["student_id"])
    item = store.find_item(checkout["item_id"])

    email = gmail_client.send_overdue_nudge(
        user=student,
        item_name=item["name"] if item else checkout["item_id"],
        due_date=checkout["due_date"],
        checkout_id=checkout_id,
    )

    return {
        "ok": email.get("ok", False),
        "checkout_id": checkout_id,
        "email": email,
    }


@mcp.tool(
    name="list_equipment_damage_reports",
    description=(
        "List all student-submitted equipment damage reports, including "
        "reporter, equipment, description, and review status. Requires a "
        "Lab Manager actor."
    ),
)
def list_equipment_damage_reports(
    actor_user_id: str,
) -> dict[str, Any]:
    """List all open and historical damage reports."""
    _require_manager(actor_user_id)

    return {
        "ok": True,
        "reports": store.get_damage_reports(),
    }


@mcp.tool(
    name="update_equipment_condition",
    description=(
        "Change an inventory item's lifecycle status. Valid statuses are "
        "'under_repair', 'damaged', 'available', and 'retired'. Use "
        "'available' after a repair is complete. An actively checked-out "
        "item cannot be moved to a non-checkout status until returned. "
        "Requires a Lab Manager actor."
    ),
)
def update_equipment_condition(
    actor_user_id: str,
    item_query: str,
    new_status: str,
    manager_note: str = "",
) -> dict[str, Any]:
    """Update a manager-controlled inventory condition."""
    manager = _require_manager(actor_user_id)

    allowed_statuses = {
        "under_repair",
        "damaged",
        "available",
        "retired",
    }

    normalized_status = new_status.lower().strip()

    if normalized_status not in allowed_statuses:
        raise ValueError(
            "new_status must be one of: under_repair, damaged, available, retired."
        )

    item = store.find_item(item_query.strip())

    if not item:
        return {
            "ok": False,
            "reason": f"Could not identify inventory item '{item_query}'.",
        }

    return store.set_inventory_status(
        item_id=item["item_id"],
        new_status=normalized_status,
        manager_id=manager["id"],
        note=manager_note.strip(),
    )


if __name__ == "__main__":
    # Do not print to stdout here: stdio is reserved for MCP protocol messages.
    mcp.run(transport="stdio")