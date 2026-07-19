"""LabBot data access and deterministic business rules."""

from __future__ import annotations

import json
import re
import uuid
from datetime import date, datetime, timedelta

from . import config

STUDENT_ITEM_CAP = 2

ACTIVE_CHECKOUT_STATUSES = ("pending", "approved", "active", "overdue")
RETURNABLE_CHECKOUT_STATUSES = ("active", "overdue")
OUTSTANDING_CHECKOUT_STATUSES = ("pending", "approved", "active", "overdue")

INVENTORY_STATUSES = (
    "available",
    "checked_out",
    "damaged",
    "under_repair",
    "retired",
)


# ---------------------------------------------------------------------------
# Raw JSON file access
# ---------------------------------------------------------------------------
def _read(path) -> list[dict]:
    return json.loads(path.read_text()) if path.exists() else []


def _write(path, data) -> None:
    path.write_text(json.dumps(data, indent=2))


def load_users() -> dict[str, dict]:
    return {user["id"]: user for user in _read(config.DATA_DIR / "users.json")}


def get_user(user_id: str) -> dict | None:
    return load_users().get(user_id)

def user_display_name(user_id: str | None) -> str | None:
    """Resolve an internal user ID to a human-readable name for display."""
    if not user_id:
        return None

    user = get_user(user_id)

    if not user:
        return "Unknown user"

    return user.get("full_name") or user.get("name") or "Unknown user"


def load_records() -> list[dict]:
    return _read(config.DATA_DIR / "records.json")


def save_records(records: list[dict]) -> None:
    _write(config.DATA_DIR / "records.json", records)


def load_checkouts() -> list[dict]:
    return _read(config.DATA_DIR / "checkouts.json")


def save_checkouts(checkouts: list[dict]) -> None:
    _write(config.DATA_DIR / "checkouts.json", checkouts)


# ---------------------------------------------------------------------------
# Matching and lookup helpers
# ---------------------------------------------------------------------------
def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _match_score(query_tokens: set[str], record: dict) -> int:
    record_tokens = (
        _tokens(record["item_id"])
        | _tokens(record["name"])
        | _tokens(record.get("category", ""))
    )
    return len(query_tokens & record_tokens)


def find_item(query: str) -> dict | None:
    query_l = query.lower().strip()
    records = load_records()

    for record in records:
        if record["item_id"].lower() == query_l:
            return record

    query_tokens = _tokens(query)
    best, best_score = None, 0

    for record in records:
        score = _match_score(query_tokens, record)
        if score > best_score:
            best, best_score = record, score

    return best


def find_available_item(query: str) -> dict | None:
    query_l = query.lower().strip()
    records = load_records()

    for record in records:
        if record["item_id"].lower() == query_l:
            return record

    query_tokens = _tokens(query)

    candidates = [
        (record, _match_score(query_tokens, record))
        for record in records
    ]
    candidates = [
        (record, score)
        for record, score in candidates
        if score > 0
    ]

    if not candidates:
        return None

    top_score = max(score for _, score in candidates)
    top_matches = [
        record
        for record, score in candidates
        if score == top_score
    ]

    available = [
        record
        for record in top_matches
        if record["status"] == "available"
    ]

    return available[0] if available else top_matches[0]


def find_items_by_category(query: str) -> list[dict]:
    query_l = query.lower().strip()

    return [
        record
        for record in load_records()
        if query_l in record["name"].lower()
        or query_l in record.get("category", "").lower()
        or query_l in record["item_id"].lower()
    ]


def active_checkouts_for_user(user_id: str) -> list[dict]:
    return [
        checkout
        for checkout in load_checkouts()
        if checkout["student_id"] == user_id
        and checkout["status"] in ACTIVE_CHECKOUT_STATUSES
    ]


def find_active_checkout(user_id: str, item_id: str) -> dict | None:
    for checkout in active_checkouts_for_user(user_id):
        if checkout["item_id"] == item_id:
            return checkout

    return None


def find_checkout(checkout_id: str) -> dict | None:
    return next(
        (
            checkout
            for checkout in load_checkouts()
            if checkout["checkout_id"] == checkout_id
        ),
        None,
    )


def find_user_checkout_item(user_id: str, query: str) -> dict | None:
    query_l = query.lower().strip()
    records = {record["item_id"]: record for record in load_records()}
    checkouts = active_checkouts_for_user(user_id)

    for checkout in checkouts:
        if checkout["item_id"].lower() == query_l:
            return records.get(checkout["item_id"])

    query_tokens = _tokens(query)
    best, best_score = None, 0

    for checkout in checkouts:
        record = records.get(checkout["item_id"])

        if not record:
            continue

        score = _match_score(query_tokens, record)

        if score > best_score:
            best, best_score = record, score

    return best


def is_on_hold(user_id: str) -> bool:
    """A student is on hold only if they have an overdue active checkout.

    Pending requests have no due date until a manager approves them, so they
    must not be treated as overdue or parsed as dated checkouts.
    """
    today = date.today()

    for checkout in active_checkouts_for_user(user_id):
        if checkout["status"] not in ("active", "overdue"):
            continue

        due_date = checkout.get("due_date")

        if not due_date:
            continue

        due = datetime.strptime(due_date, "%Y-%m-%d").date()

        if checkout["status"] == "overdue" or due < today:
            return True

    return False

def _display_damage_reports(damage_reports: list[dict]) -> list[dict]:
    return [
        {
            "report_id": report.get("report_id"),
            "reported_on": report.get("reported_on"),
            "reported_by_name": user_display_name(report.get("reported_by")),
            "description": report.get("description", ""),
            "status": report.get("status", "open"),
            "manager_note": report.get("manager_note", ""),
        }
        for report in damage_reports
    ]

def _checkout_summary(checkout: dict) -> dict:
    records = {record["item_id"]: record for record in load_records()}
    record = records.get(checkout["item_id"], {})

    return {
        "checkout_id": checkout["checkout_id"],
        "item_id": checkout["item_id"],
        "item_name": record.get("name", checkout["item_id"]),

        # Display names only for user-facing summaries.
        "student_name": user_display_name(checkout.get("student_id")),
        "approved_by_name": user_display_name(checkout.get("approved_by")),

        "checkout_date": checkout.get("checkout_date"),
        "due_date": checkout.get("due_date"),
        "return_date": checkout.get("return_date"),
        "requested_days": checkout.get("requested_days"),
        "status": checkout.get("status"),
        "notes": checkout.get("notes", ""),
        "damage_reports": _display_damage_reports(checkout.get("damage_reports", [])),
    }


# ---------------------------------------------------------------------------
# Student actions
# ---------------------------------------------------------------------------
def request_checkout(user_id: str, item_id: str, days: int) -> dict:
    """Create a PENDING request.

    Inventory remains available until a lab manager approves the request.
    The item cap includes pending requests so a student cannot flood the
    queue with requests.
    """
    records = load_records()
    record = next(
        (
            item
            for item in records
            if item["item_id"] == item_id
        ),
        None,
    )

    if record is None:
        return {
            "ok": False,
            "reason": f"No item with id '{item_id}' exists.",
        }

    if record["status"] != "available":
        return {
            "ok": False,
            "reason": (
                f"{record['name']} ({item_id}) is not currently available "
                f"(status: {record['status']})."
            ),
        }

    if is_on_hold(user_id):
        return {
            "ok": False,
            "reason": (
                "You have an overdue item, so you are on hold until it is returned."
            ),
        }

    if len(active_checkouts_for_user(user_id)) >= STUDENT_ITEM_CAP:
        return {
            "ok": False,
            "reason": (
                f"You already have {STUDENT_ITEM_CAP} pending or active "
                "checkout requests, which is the limit."
            ),
        }

    limit = record.get("checkout_limit_days", days)
    requested_days = min(days, limit) if days else limit

    today = date.today()

    checkout = {
        "checkout_id": f"c-{uuid.uuid4().hex[:8]}",
        "item_id": item_id,
        "student_id": user_id,
        "checkout_date": today.isoformat(),
        "due_date": None,
        "return_date": None,
        "requested_days": requested_days,
        "status": "pending",
        "approved_by": None,
        "notes": "",
        "damage_reports": [],
        "calendar_event_id": None,
    }

    checkouts = load_checkouts()
    checkouts.append(checkout)
    save_checkouts(checkouts)

    return {
        "ok": True,
        "checkout": checkout,
        "item_name": record["name"],
        "days": requested_days,
    }


def report_return(user_id: str, item_id: str) -> dict:
    checkout = find_active_checkout(user_id, item_id)

    if checkout is None:
        return {
            "ok": False,
            "reason": (
                "I cannot find an active checkout for that item under your account."
            ),
        }

    if checkout["status"] not in RETURNABLE_CHECKOUT_STATUSES:
        return {
            "ok": False,
            "reason": (
                f"That checkout cannot be returned while its status is "
                f"'{checkout['status']}'."
            ),
        }

    calendar_event_id = checkout.get("calendar_event_id")

    checkouts = load_checkouts()

    for saved_checkout in checkouts:
        if saved_checkout["checkout_id"] == checkout["checkout_id"]:
            saved_checkout["status"] = "returned"
            saved_checkout["return_date"] = date.today().isoformat()

    save_checkouts(checkouts)

    records = load_records()

    for record in records:
        if record["item_id"] == item_id:
            record["status"] = "available"
            record["checked_out_by"] = None

    save_records(records)

    return {
        "ok": True,
        "item_id": item_id,
        "calendar_event_id": calendar_event_id,
    }


def report_all_returns(user_id: str) -> dict:
    active = [
        checkout
        for checkout in load_checkouts()
        if checkout["student_id"] == user_id
        and checkout["status"] in RETURNABLE_CHECKOUT_STATUSES
    ]

    if not active:
        return {
            "ok": False,
            "reason": "You do not have any active checked-out items to return.",
        }

    return {
        "ok": True,
        "returned": [
            report_return(user_id, checkout["item_id"])
            for checkout in active
        ],
    }


def report_damage(user_id: str, item_id: str, description: str) -> dict:
    """Students may report damage only for equipment assigned to them."""
    checkout = find_active_checkout(user_id, item_id)

    if checkout is None:
        return {
            "ok": False,
            "reason": (
                "You can report damage only for equipment currently checked out "
                "under your account."
            ),
        }

    report = {
        "report_id": f"d-{uuid.uuid4().hex[:8]}",
        "reported_on": date.today().isoformat(),
        "reported_by": user_id,
        "description": description.strip(),
        "status": "open",
        "manager_note": "",
    }

    checkouts = load_checkouts()

    for saved_checkout in checkouts:
        if saved_checkout["checkout_id"] == checkout["checkout_id"]:
            saved_checkout.setdefault("damage_reports", []).append(report)

            note = (
                f"[Damage report {report['report_id']} on {report['reported_on']}] "
                f"{report['description']}"
            )
            saved_checkout["notes"] = (
                f"{saved_checkout.get('notes', '').strip()}\n{note}"
            ).strip()

    save_checkouts(checkouts)

    return {
        "ok": True,
        "item_id": item_id,
        "checkout_id": checkout["checkout_id"],
        "report": report,
    }


def my_status(user_id: str) -> dict:
    checkouts = [
        _checkout_summary(checkout)
        for checkout in load_checkouts()
        if checkout["student_id"] == user_id
        and checkout["status"] in (
            "pending",
            "active",
            "overdue",
            "denied",
        )
    ]

    return {
        "ok": True,
        "on_hold": is_on_hold(user_id),
        "checkouts": checkouts,
    }


def time_remaining(user_id: str, item_id: str | None) -> dict:
    active = [
        checkout
        for checkout in active_checkouts_for_user(user_id)
        if checkout["status"] in RETURNABLE_CHECKOUT_STATUSES
    ]

    if not active:
        return {
            "ok": False,
            "reason": "You do not have anything actively checked out right now.",
        }

    if item_id:
        checkout = next(
            (
                candidate
                for candidate in active
                if candidate["item_id"] == item_id
            ),
            None,
        )

        if checkout is None:
            return {
                "ok": False,
                "reason": f"You do not have '{item_id}' actively checked out.",
            }

    elif len(active) == 1:
        checkout = active[0]

    else:
        items = ", ".join(checkout["item_id"] for checkout in active)

        return {
            "ok": False,
            "reason": (
                f"You have more than one item out ({items}) — which one do you mean?"
            ),
        }

    due = datetime.strptime(checkout["due_date"], "%Y-%m-%d").date()
    delta = (due - date.today()).days

    record = next(
        (
            item
            for item in load_records()
            if item["item_id"] == checkout["item_id"]
        ),
        None,
    )

    return {
        "ok": True,
        "item_id": checkout["item_id"],
        "item_name": record["name"] if record else checkout["item_id"],
        "due_date": checkout["due_date"],
        "days_remaining": delta,
        "overdue": delta < 0,
    }


# ---------------------------------------------------------------------------
# Manager actions
# ---------------------------------------------------------------------------
def approve_or_deny(
    checkout_id: str,
    decision: str,
    manager_id: str,
    manager_note: str = "",
) -> dict:
    """Approve or deny a pending request.

    Approval checks inventory again because another manager may have approved
    a competing request after the student submitted this request.
    """
    if decision not in ("approve", "deny"):
        return {
            "ok": False,
            "reason": "Decision must be either 'approve' or 'deny'.",
        }

    checkouts = load_checkouts()
    checkout = next(
        (
            candidate
            for candidate in checkouts
            if candidate["checkout_id"] == checkout_id
        ),
        None,
    )

    if checkout is None:
        return {
            "ok": False,
            "reason": f"No checkout found with id {checkout_id}.",
        }

    if checkout["status"] != "pending":
        return {
            "ok": False,
            "reason": (
                f"Checkout {checkout_id} is not pending "
                f"(status: {checkout['status']})."
            ),
        }

    if decision == "deny":
        checkout["status"] = "denied"
        checkout["approved_by"] = manager_id

        if manager_note.strip():
            checkout["notes"] = (
                f"{checkout.get('notes', '').strip()}\n"
                f"[Manager denial note] {manager_note.strip()}"
            ).strip()

        save_checkouts(checkouts)

        return {
            "ok": True,
            "decision": "deny",
            "checkout": _checkout_summary(checkout),
        }

    records = load_records()
    record = next(
        (
            item
            for item in records
            if item["item_id"] == checkout["item_id"]
        ),
        None,
    )

    if record is None:
        return {
            "ok": False,
            "reason": (
                f"The inventory record for {checkout['item_id']} no longer exists."
            ),
        }

    if record["status"] != "available":
        return {
            "ok": False,
            "reason": (
                f"{record['name']} is no longer available "
                f"(status: {record['status']}). "
                "This pending request remains pending."
            ),
        }

    today = date.today()
    requested_days = checkout.get(
        "requested_days",
        record.get("checkout_limit_days", 3),
    )

    checkout["status"] = "active"
    checkout["approved_by"] = manager_id
    checkout["checkout_date"] = today.isoformat()
    checkout["due_date"] = (today + timedelta(days=requested_days)).isoformat()

    if manager_note.strip():
        checkout["notes"] = (
            f"{checkout.get('notes', '').strip()}\n"
            f"[Manager approval note] {manager_note.strip()}"
        ).strip()

    record["status"] = "checked_out"
    record["checked_out_by"] = checkout["student_id"]

    save_checkouts(checkouts)
    save_records(records)

    return {
        "ok": True,
        "decision": "approve",
        "checkout": _checkout_summary(checkout),
        "item_name": record["name"],
        "student": load_users().get(checkout["student_id"]),
    }


def outstanding_checkouts() -> list[dict]:
    return [
        _checkout_summary(checkout)
        for checkout in load_checkouts()
        if checkout["status"] in OUTSTANDING_CHECKOUT_STATUSES
    ]


def pending_checkouts() -> list[dict]:
    return [
        _checkout_summary(checkout)
        for checkout in load_checkouts()
        if checkout["status"] == "pending"
    ]


def overdue_items() -> list[dict]:
    today = date.today()
    changed = False
    checkouts = load_checkouts()

    for checkout in checkouts:
        if checkout["status"] != "active" or not checkout.get("due_date"):
            continue

        due = datetime.strptime(checkout["due_date"], "%Y-%m-%d").date()

        if due < today:
            checkout["status"] = "overdue"
            changed = True

    if changed:
        save_checkouts(checkouts)

    return [
        _checkout_summary(checkout)
        for checkout in checkouts
        if checkout["status"] == "overdue"
    ]


def get_damage_reports() -> list[dict]:
    records = {record["item_id"]: record for record in load_records()}
    reports = []

    for checkout in load_checkouts():
        record = records.get(checkout["item_id"], {})

        for report in checkout.get("damage_reports", []):
            reports.append(
                {
                    "report_id": report["report_id"],
                    "reported_on": report["reported_on"],
                    "reported_by_name": user_display_name(
                        report.get("reported_by")
                    ),
                    "student_name": user_display_name(
                        checkout.get("student_id")
                    ),
                    "item_name": record.get("name", checkout["item_id"]),
                    "description": report["description"],
                    "status": report.get("status", "open"),
                    "manager_note": report.get("manager_note", ""),
                    "checkout_id": checkout["checkout_id"],
                    "item_id": checkout["item_id"],
                }
            )

    return reports


def set_inventory_status(
    item_id: str,
    new_status: str,
    manager_id: str,
    note: str = "",
) -> dict:
    """Manager-only status update. Caller must enforce role authorization."""
    if new_status not in INVENTORY_STATUSES:
        return {
            "ok": False,
            "reason": f"Unsupported inventory status '{new_status}'.",
        }

    records = load_records()
    record = next(
        (
            item
            for item in records
            if item["item_id"] == item_id
        ),
        None,
    )

    if record is None:
        return {
            "ok": False,
            "reason": f"No inventory item found with id '{item_id}'.",
        }

    if record["status"] == "checked_out" and new_status != "checked_out":
        active = [
            checkout
            for checkout in load_checkouts()
            if checkout["item_id"] == item_id
            and checkout["status"] in RETURNABLE_CHECKOUT_STATUSES
        ]

        if active:
            return {
                "ok": False,
                "reason": (
                    f"{record['name']} is actively checked out and cannot be "
                    f"marked '{new_status}' until it is returned."
                ),
            }

    record["status"] = new_status

    if new_status != "checked_out":
        record["checked_out_by"] = None

    record["manager_note"] = (
        f"[{date.today().isoformat()} by {manager_id}] {note.strip()}"
        if note.strip()
        else f"[{date.today().isoformat()} by {manager_id}] Status set to {new_status}."
    )

    save_records(records)

    return {
        "ok": True,
        "item": record,
        "new_status": new_status,
    }

def set_calendar_event_id(checkout_id: str, event_id: str | None) -> None:
    """Save the Calendar event created after a manager approves a request."""

    checkouts = load_checkouts()

    for checkout in checkouts:
        if checkout["checkout_id"] == checkout_id:
            checkout["calendar_event_id"] = event_id
            break

    save_checkouts(checkouts)