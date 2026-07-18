"""LabBot data access.

Everything that touches data/records.json, data/checkouts.json, or
data/users.json goes through here — action nodes in app/graph.py should
never open these files directly. This is also where the *deterministic*
business rules live (item cap, hold status, overdue calc): rules that must
be enforced in code, not left to the LLM to remember or apply consistently.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import date, datetime, timedelta

from . import config

STUDENT_ITEM_CAP = 2


# ---------------------------------------------------------------------------
# Raw file IO
# ---------------------------------------------------------------------------
def _read(path):
    return json.loads(path.read_text()) if path.exists() else []


def _write(path, data):
    path.write_text(json.dumps(data, indent=2))


def load_users() -> dict[str, dict]:
    return {u["id"]: u for u in _read(config.DATA_DIR / "users.json")}


def get_user(user_id: str) -> dict | None:
    return load_users().get(user_id)


def load_records() -> list[dict]:
    return _read(config.DATA_DIR / "records.json")


def save_records(records: list[dict]) -> None:
    _write(config.DATA_DIR / "records.json", records)


def load_checkouts() -> list[dict]:
    return _read(config.DATA_DIR / "checkouts.json")


def save_checkouts(checkouts: list[dict]) -> None:
    _write(config.DATA_DIR / "checkouts.json", checkouts)


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------
def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _match_score(query_tokens: set[str], record: dict) -> int:
    """How many query tokens overlap with this record's id/name/category
    tokens. Token overlap instead of substring containment, so 'ESP32 kit'
    still matches 'ESP32 Dev Kit' (substring matching missed this — 'esp32
    kit' is not a literal substring of 'esp32 dev kit')."""
    record_tokens = (
        _tokens(record["item_id"]) | _tokens(record["name"]) | _tokens(record.get("category", ""))
    )
    return len(query_tokens & record_tokens)


def find_item(query: str) -> dict | None:
    """Best single match for a free-text item query against the WHOLE
    inventory, by token overlap. Use this for requests that are genuinely
    about the catalog (e.g. checking out something new). For resolving
    which item a specific user means among their OWN checkouts, use
    find_user_checkout_item instead — matching against the whole catalog
    is wrong there if there are multiple similar items (e.g. two
    oscilloscopes) and the user only holds one of them.
    """
    query_l = query.lower().strip()
    records = load_records()
    for r in records:
        if r["item_id"].lower() == query_l:
            return r

    query_tokens = _tokens(query)
    best, best_score = None, 0
    for r in records:
        score = _match_score(query_tokens, r)
        if score > best_score:
            best, best_score = r, score
    return best


def find_available_item(query: str) -> dict | None:
    """Like find_item, but among tied/matching candidates prefers one that
    is actually available — so 'check out an ESP32 kit' doesn't get stuck
    on a specific unavailable unit when a sibling unit is free. Falls back
    to the best match regardless of availability if none are available, so
    the caller can still report a sensible 'not available' reason."""
    query_l = query.lower().strip()
    records = load_records()
    for r in records:
        if r["item_id"].lower() == query_l:
            return r

    query_tokens = _tokens(query)
    candidates = [(r, _match_score(query_tokens, r)) for r in records]
    candidates = [(r, s) for r, s in candidates if s > 0]
    if not candidates:
        return None
    max_score = max(s for _, s in candidates)
    top = [r for r, s in candidates if s == max_score]
    available = [r for r in top if r["status"] == "available"]
    return available[0] if available else top[0]


def find_user_checkout_item(user_id: str, query: str) -> dict | None:
    """Match a free-text item query against ONLY this user's active
    checkouts (by token overlap against each checked-out item's id/name/
    category), not the whole inventory. This is what report_return and
    check_time_remaining should use — if the user has scope-02 checked out
    and asks about 'the oscilloscope', this returns scope-02 even though
    scope-01 also matches 'oscilloscope' and happens to sort first in the
    inventory file.
    """
    query_l = query.lower().strip()
    records = {r["item_id"]: r for r in load_records()}
    checkouts = active_checkouts_for_user(user_id)

    for c in checkouts:
        if c["item_id"].lower() == query_l:
            return records.get(c["item_id"])

    query_tokens = _tokens(query)
    best, best_score = None, 0
    for c in checkouts:
        r = records.get(c["item_id"])
        if not r:
            continue
        score = _match_score(query_tokens, r)
        if score > best_score:
            best, best_score = r, score
    return best


def find_items_by_category(query: str) -> list[dict]:
    query_l = query.lower().strip()
    return [
        r for r in load_records()
        if query_l in r["name"].lower() or query_l in r.get("category", "").lower()
    ]


def active_checkouts_for_user(user_id: str) -> list[dict]:
    return [
        c for c in load_checkouts()
        if c["student_id"] == user_id and c["status"] in ("pending", "approved", "active", "overdue")
    ]


def find_active_checkout(user_id: str, item_id: str) -> dict | None:
    for c in active_checkouts_for_user(user_id):
        if c["item_id"] == item_id:
            return c
    return None


def is_on_hold(user_id: str) -> bool:
    """A student with any overdue item is on hold (Checkout Policy, Late
    Returns & Holds)."""
    today = date.today()
    for c in active_checkouts_for_user(user_id):
        due = datetime.strptime(c["due_date"], "%Y-%m-%d").date()
        if c["status"] != "returned" and due < today:
            return True
    return False


# ---------------------------------------------------------------------------
# Write actions
# ---------------------------------------------------------------------------
def request_checkout(user_id: str, item_id: str, days: int) -> dict:
    """Attempt to check out an item for `user_id`. Enforces the 2-item cap
    and hold status in code — not left to the LLM to decide. Returns a
    structured result dict; never raises for "expected" failures like the
    item being unavailable, since the caller needs to relay that honestly.
    """
    records = load_records()
    record = next((r for r in records if r["item_id"] == item_id), None)
    if record is None:
        return {"ok": False, "reason": f"No item with id '{item_id}' exists."}

    if record["status"] != "available":
        return {"ok": False, "reason": f"{record['name']} ({item_id}) is not currently available."}

    if is_on_hold(user_id):
        return {"ok": False, "reason": "You have an overdue item, so you're on hold until it's returned."}

    if len(active_checkouts_for_user(user_id)) >= STUDENT_ITEM_CAP:
        return {"ok": False, "reason": f"You already have {STUDENT_ITEM_CAP} items checked out, which is the cap."}

    limit = record.get("checkout_limit_days", days)
    requested_days = min(days, limit) if days else limit

    today = date.today()
    checkout = {
        "checkout_id": f"c-{uuid.uuid4().hex[:8]}",
        "item_id": item_id,
        "student_id": user_id,
        "checkout_date": today.isoformat(),
        "due_date": (today + timedelta(days=requested_days)).isoformat(),
        "return_date": None,
        "status": "active",
        "approved_by": None,
        "notes": "",
        "calendar_event_id": None,  # set later by calendar_client.create_due_date_event, if it succeeds
    }

    checkouts = load_checkouts()
    checkouts.append(checkout)
    save_checkouts(checkouts)

    record["status"] = "checked_out"
    record["checked_out_by"] = user_id
    save_records(records)

    return {"ok": True, "checkout": checkout, "item_name": record["name"], "days": requested_days}


def set_calendar_event_id(checkout_id: str, event_id: str | None) -> None:
    """Persist the Google Calendar event id created for a checkout, so a
    later return can delete the right event. Called by app/graph.py after
    calendar_client.create_due_date_event succeeds — never by store.py
    itself, which knows nothing about Google Calendar."""
    checkouts = load_checkouts()
    for c in checkouts:
        if c["checkout_id"] == checkout_id:
            c["calendar_event_id"] = event_id
    save_checkouts(checkouts)


def report_return(user_id: str, item_id: str) -> dict:
    checkout = find_active_checkout(user_id, item_id)
    if checkout is None:
        return {"ok": False, "reason": "I can't find an active checkout for that item under your account."}

    calendar_event_id = checkout.get("calendar_event_id")

    checkouts = load_checkouts()
    for c in checkouts:
        if c["checkout_id"] == checkout["checkout_id"]:
            c["status"] = "returned"
            c["return_date"] = date.today().isoformat()
    save_checkouts(checkouts)

    records = load_records()
    for r in records:
        if r["item_id"] == item_id:
            r["status"] = "available"
            r["checked_out_by"] = None
    save_records(records)

    return {"ok": True, "item_id": item_id, "calendar_event_id": calendar_event_id}


def time_remaining(user_id: str, item_id: str | None) -> dict:
    """Days (or overdue-by-days) left on a specific checkout. If item_id is
    None but the user has exactly one active checkout, use that one."""
    active = active_checkouts_for_user(user_id)
    if not active:
        return {"ok": False, "reason": "You don't have anything checked out right now."}

    if item_id:
        checkout = find_active_checkout(user_id, item_id)
        if checkout is None:
            return {"ok": False, "reason": f"You don't have '{item_id}' checked out."}
    elif len(active) == 1:
        checkout = active[0]
    else:
        items = ", ".join(c["item_id"] for c in active)
        return {"ok": False, "reason": f"You have more than one item out ({items}) — which one do you mean?"}

    due = datetime.strptime(checkout["due_date"], "%Y-%m-%d").date()
    today = date.today()
    delta = (due - today).days

    record = next((r for r in load_records() if r["item_id"] == checkout["item_id"]), None)
    item_name = record["name"] if record else checkout["item_id"]

    return {
        "ok": True,
        "item_id": checkout["item_id"],
        "item_name": item_name,
        "due_date": checkout["due_date"],
        "days_remaining": delta,
        "overdue": delta < 0,
    }


def overdue_items() -> list[dict]:
    today = date.today()
    out = []
    for c in load_checkouts():
        if c["status"] in ("returned",):
            continue
        due = datetime.strptime(c["due_date"], "%Y-%m-%d").date()
        if due < today:
            out.append(c)
    return out


def approve_or_deny(checkout_id: str, decision: str, manager_id: str) -> dict:
    checkouts = load_checkouts()
    for c in checkouts:
        if c["checkout_id"] == checkout_id:
            if c["status"] != "pending":
                return {"ok": False, "reason": f"Checkout {checkout_id} is not pending (status: {c['status']})."}
            c["status"] = "active" if decision == "approve" else "denied"
            c["approved_by"] = manager_id
            save_checkouts(checkouts)
            return {"ok": True, "checkout_id": checkout_id, "decision": decision}
    return {"ok": False, "reason": f"No checkout found with id {checkout_id}."}

def report_all_returns(user_id: str) -> dict:
    returnable_statuses = ("active", "overdue")

    active = [
        c for c in load_checkouts()
        if c["student_id"] == user_id and c["status"] in returnable_statuses
    ]

    if not active:
        return {
            "ok": False,
            "reason": "You don't have any active checked-out items to return.",
        }

    item_ids = {c["item_id"] for c in active}
    returned_on = date.today().isoformat()

    checkouts = load_checkouts()
    for checkout in checkouts:
        if checkout["checkout_id"] in {c["checkout_id"] for c in active}:
            checkout["status"] = "returned"
            checkout["return_date"] = returned_on
    save_checkouts(checkouts)

    records = load_records()
    for record in records:
        if record["item_id"] in item_ids:
            record["status"] = "available"
            record["checked_out_by"] = None
    save_records(records)

    return {
        "ok": True,
        "returned": [
            {
                "checkout_id": c["checkout_id"],
                "item_id": c["item_id"],
                "calendar_event_id": c.get("calendar_event_id"),
            }
            for c in active
        ],
    }