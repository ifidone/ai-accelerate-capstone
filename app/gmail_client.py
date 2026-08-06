"""Gmail integration — transactional emails for checkout events.

Sends three types of email:
  1. Checkout confirmation — immediately when a checkout succeeds
  2. Return confirmation   — immediately when a return succeeds
  3. Due-date reminder     — the day before an item is due, sent by the
                             background scheduler in app/scheduler.py

Authentication: reuses the same OAuth credentials as the Calendar
integration (config.GOOGLE_TOKEN_PATH / config.GOOGLE_CREDENTIALS_PATH).
Scope required: https://www.googleapis.com/auth/gmail.send
Add this to config.GOOGLE_SCOPES alongside the calendar scope.

All send functions return {"ok": bool, ...} and never raise — a failed
email must never roll back a checkout or return that already succeeded
in store.py. Same partial-failure pattern as calendar_client.py.

The emails go to the student's address (looked up from users.json by
their user_id). The debug copy always goes to config.GMAIL_DEBUG_ADDRESS
(your email) so you see every outbound message during development.
"""

from __future__ import annotations

import base64
import json
from email.mime.text import MIMEText

from . import config


# ---------------------------------------------------------------------------
# Gmail service
# ---------------------------------------------------------------------------
def _service():
    from google.auth.exceptions import RefreshError
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    if not config.GOOGLE_TOKEN_PATH.exists():
        raise RuntimeError(
            "No service token found. Run the OAuth consent flow once "
            "(visit /auth/login in the browser) to generate "
            f"{config.GOOGLE_TOKEN_PATH}."
        )
    creds = Credentials.from_authorized_user_file(
        str(config.GOOGLE_TOKEN_PATH), config.GOOGLE_SCOPES
    )
    if not creds.valid:
        if creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                config.GOOGLE_TOKEN_PATH.write_text(creds.to_json())
            except RefreshError as e:
                raise RuntimeError(f"Token refresh failed: {e}") from e
        else:
            raise RuntimeError("Token invalid and cannot be refreshed — re-run the OAuth flow.")
    return build("gmail", "v1", credentials=creds)


def _make_message(to: str, subject: str, body: str) -> dict:
    msg = MIMEText(body, "plain")
    msg["to"] = to
    msg["subject"] = subject
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    return {"raw": raw}


def _send(to: str, subject: str, body: str) -> dict:
    """Send to `to` and BCC a copy to the debug address. Returns {"ok": bool, ...}."""
    try:
        svc = _service()

        recipients = list({to, config.GMAIL_DEBUG_ADDRESS} - {""})
        results = []
        for addr in recipients:
            msg = _make_message(addr, subject, body)
            svc.users().messages().send(userId="me", body=msg).execute()
            results.append(addr)

        return {"ok": True, "sent_to": results}
    except RuntimeError as e:
        return {"ok": False, "reason": str(e)}
    except Exception as e:
        from googleapiclient.errors import HttpError
        if isinstance(e, HttpError):
            status = getattr(getattr(e, "resp", None), "status", None)
            if status == 401:
                return {"ok": False, "reason": "Gmail auth expired — re-run OAuth flow."}
            if status == 403:
                error_text = str(e)

                if (
                    "insufficientPermissions" in error_text
                    or "insufficient authentication scopes" in error_text.lower()
                ):
                    return {
                        "ok": False,
                        "reason": (
                            "Gmail permission is missing. Reauthorize the bot account "
                            "with the gmail.send scope."
                        ),
                    }

                if "Gmail API has not been used" in error_text:
                    return {
                        "ok": False,
                        "reason": (
                            "The Gmail API is not enabled for this Google Cloud project."
                        ),
                    }

                return {
                    "ok": False,
                    "reason": f"Gmail rejected the send request (403): {error_text}",
                }
        return {"ok": False, "reason": f"Unexpected Gmail error: {e}"}


def _student_email(user: dict | None) -> str:
    """Return the student's email address. Falls back to debug address so
    we always get the email even if users.json doesn't have an email field."""
    if user and user.get("email"):
        return user["email"]
    return config.GMAIL_DEBUG_ADDRESS


# ---------------------------------------------------------------------------
# Public send functions
# ---------------------------------------------------------------------------
def send_checkout_confirmation(user: dict | None, item_name: str, due_date: str, checkout_id: str) -> dict:
    """Send immediately after a successful checkout."""
    to = _student_email(user)
    name = user["name"] if user else "Student"
    subject = f"Supply Sage: {item_name} checked out — due {due_date}"
    body = (
        f"Hi {name},\n\n"
        f"Your checkout was confirmed:\n\n"
        f"  Item:        {item_name}\n"
        f"  Checkout ID: {checkout_id}\n"
        f"  Due date:    {due_date}\n\n"
        f"Please return it by end of day on {due_date}. If you need more "
        f"time, request a renewal before the due date (renewals are not "
        f"available once an item is overdue).\n\n"
        f"— Supply Sage"
    )
    return _send(to, subject, body)


def send_return_confirmation(user: dict | None, item_name: str, return_date: str) -> dict:
    """Send immediately after a successful return."""
    to = _student_email(user)
    name = user["name"] if user else "Student"
    subject = f"Supply Sage: {item_name} returned"
    body = (
        f"Hi {name},\n\n"
        f"Your return was recorded:\n\n"
        f"  Item:        {item_name}\n"
        f"  Returned on: {return_date}\n\n"
        f"Thanks for returning it on time. The item is now available "
        f"for other students.\n\n"
        f"— Supply Sage"
    )
    return _send(to, subject, body)


def send_due_date_reminder(user: dict | None, item_name: str, due_date: str, checkout_id: str) -> dict:
    """Send the day before the due date. Called by app/scheduler.py."""
    to = _student_email(user)
    name = user["name"] if user else "Student"
    subject = f"Supply Sage reminder: {item_name} due tomorrow ({due_date})"
    body = (
        f"Hi {name},\n\n"
        f"This is a reminder that the following item is due back tomorrow:\n\n"
        f"  Item:        {item_name}\n"
        f"  Checkout ID: {checkout_id}\n"
        f"  Due date:    {due_date}\n\n"
        f"If you need more time, message Supply Sage to request a renewal "
        f"before it becomes overdue. If the item is already returned, "
        f"you can ignore this message.\n\n"
        f"— Supply Sage"
    )
    return _send(to, subject, body)

def send_overdue_nudge(
    user: dict | None,
    item_name: str,
    due_date: str,
    checkout_id: str,
) -> dict:
    """Send an immediate manager-triggered overdue reminder."""
    to = _student_email(user)
    name = user["name"] if user else "Student"

    subject = f"Supply Sage overdue notice: {item_name} was due {due_date}"

    body = (
        f"Hi {name},\n\n"
        f"This is a Supply Sage reminder that the following equipment is overdue:\n\n"
        f"  Item:        {item_name}\n"
        f"  Checkout ID: {checkout_id}\n"
        f"  Due date:    {due_date}\n\n"
        f"Please return the item as soon as possible, or contact the lab "
        f"manager if there is an issue.\n\n"
        f"— Supply Sage"
    )

    return _send(to, subject, body)