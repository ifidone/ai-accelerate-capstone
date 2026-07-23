"""Shared checkout approval actions.

Both the LangGraph chat workflow and direct manager-dashboard buttons use
these functions. This keeps checkout approval, Calendar synchronization, and
email behavior consistent across interfaces.
"""

from __future__ import annotations

from . import calendar_client, gmail_client, store


def approve_checkout_request(
    checkout_id: str,
    manager_id: str,
    manager_note: str = "",
) -> dict:
    """Approve a pending request and run non-blocking integrations."""
    result = store.approve_or_deny(
        checkout_id=checkout_id,
        decision="approve",
        manager_id=manager_id,
        manager_note=manager_note,
    )

    result["action"] = "approve"

    if not result.get("ok"):
        return result

    checkout_summary = result["checkout"]
    raw_checkout = store.find_checkout(checkout_summary["checkout_id"])

    if not raw_checkout:
        result["calendar"] = {
            "ok": False,
            "reason": (
                "The request was approved, but LabBot could not reload it "
                "for Calendar synchronization."
            ),
        }
        return result

    calendar = calendar_client.create_checkout_events(
        student_user_id=raw_checkout["student_id"],
        item_name=result["item_name"],
        due_date=raw_checkout["due_date"],
        checkout_id=raw_checkout["checkout_id"],
    )

    result["calendar"] = calendar

    # Persist whichever event IDs were successfully created. A later return
    # can delete either event independently.
    store.set_calendar_event_ids(
        checkout_id=raw_checkout["checkout_id"],
        event_ids=calendar.get("event_ids", {}),
    )

    result["email"] = gmail_client.send_checkout_confirmation(
        user=result.get("student"),
        item_name=result["item_name"],
        due_date=raw_checkout["due_date"],
        checkout_id=raw_checkout["checkout_id"],
    )

    return result


def deny_checkout_request(
    checkout_id: str,
    manager_id: str,
    manager_note: str = "",
) -> dict:
    """Deny a pending request without changing inventory availability."""
    result = store.approve_or_deny(
        checkout_id=checkout_id,
        decision="deny",
        manager_id=manager_id,
        manager_note=manager_note,
    )

    result["action"] = "deny"
    return result