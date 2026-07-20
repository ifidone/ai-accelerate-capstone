"""Google Calendar integration for LabBot.

A successful approved checkout creates two independent due-date events:

1. Student calendar
   Uses the signed-in student's saved Google OAuth token and writes to that
   person's primary Google Calendar.

2. LabBot debug calendar
   Uses the separately authorized bot/debug account token and writes to the
   configured debug calendar.

Calendar failures never roll back a successful equipment checkout or return.
"""

from __future__ import annotations

from datetime import date, timedelta

from . import config, user_google_tokens

BOT_CALENDAR_SCOPES = [
    "https://www.googleapis.com/auth/calendar.events",
]

_bot_service_cache = None


def _event_body(item_name: str, due_date: str, checkout_id: str) -> dict:
    """Create a Google all-day event body.

    Google all-day event end dates are exclusive, so an event due July 21
    starts July 21 and ends July 22.
    """
    due = date.fromisoformat(due_date)
    end_date = (due + timedelta(days=1)).isoformat()

    return {
        "summary": f"Lab equipment due: {item_name}",
        "description": f"LabBot checkout {checkout_id}",
        "start": {"date": due_date},
        "end": {"date": end_date},
        "reminders": {"useDefault": True},
    }


def _describe_http_error(error, operation: str) -> str:
    status = getattr(error.resp, "status", None)

    if status == 401:
        return "Google Calendar authentication expired and must be reauthorized."

    if status == 403:
        return "Google Calendar denied access or its quota was exceeded."

    if status == 404:
        if operation == "create":
            return (
                "The configured calendar could not be found or the authorized "
                "account does not have access to it."
            )

        return "The calendar event was already removed or no longer exists."

    return f"Google Calendar API error during {operation} (status {status})."


def _build_service(credentials):
    from googleapiclient.discovery import build

    return build(
        "calendar",
        "v3",
        credentials=credentials,
        cache_discovery=False,
    )


def _get_bot_credentials():
    from google.auth.exceptions import RefreshError
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    if not config.GOOGLE_TOKEN_PATH.exists():
        raise RuntimeError(
            "The LabBot debug Calendar token does not exist. "
            "Run the bot-calendar authorization script."
        )

    credentials = Credentials.from_authorized_user_file(
        str(config.GOOGLE_TOKEN_PATH),
        BOT_CALENDAR_SCOPES,
    )

    if credentials.valid:
        return credentials

    if not credentials.expired or not credentials.refresh_token:
        raise RuntimeError(
            "The LabBot debug Calendar token is invalid. Reauthorize it."
        )

    try:
        credentials.refresh(Request())
    except RefreshError as exc:
        raise RuntimeError(
            "The LabBot debug Calendar token expired. Reauthorize it."
        ) from exc

    config.GOOGLE_TOKEN_PATH.write_text(credentials.to_json())
    return credentials


def _bot_service():
    global _bot_service_cache

    if _bot_service_cache is None:
        _bot_service_cache = _build_service(_get_bot_credentials())

    return _bot_service_cache


def _student_service(student_user_id: str):
    credentials = user_google_tokens.load(student_user_id)

    if not credentials:
        raise RuntimeError(
            "The student's Google Calendar is not connected. "
            "The student should sign in to LabBot again."
        )

    return _build_service(credentials)


def _create_event(
    service,
    calendar_id: str,
    item_name: str,
    due_date: str,
    checkout_id: str,
) -> dict:
    try:
        from googleapiclient.errors import HttpError

        created = (
            service.events()
            .insert(
                calendarId=calendar_id,
                body=_event_body(item_name, due_date, checkout_id),
            )
            .execute()
        )

        return {
            "ok": True,
            "event_id": created["id"],
        }

    except HttpError as error:
        return {
            "ok": False,
            "reason": _describe_http_error(error, "create"),
        }
    except Exception as error:
        return {
            "ok": False,
            "reason": f"Unexpected Calendar creation error: {error}",
        }


def _delete_event(
    service,
    calendar_id: str,
    event_id: str | None,
) -> dict:
    if not event_id:
        return {
            "ok": True,
            "note": "no event was recorded",
        }

    try:
        from googleapiclient.errors import HttpError

        (
            service.events()
            .delete(
                calendarId=calendar_id,
                eventId=event_id,
            )
            .execute()
        )

        return {"ok": True}

    except HttpError as error:
        status = getattr(error.resp, "status", None)

        if status == 404:
            return {
                "ok": True,
                "note": "event was already removed",
            }

        return {
            "ok": False,
            "reason": _describe_http_error(error, "delete"),
        }
    except Exception as error:
        return {
            "ok": False,
            "reason": f"Unexpected Calendar deletion error: {error}",
        }


def create_checkout_events(
    student_user_id: str,
    item_name: str,
    due_date: str,
    checkout_id: str,
) -> dict:
    """Create the student and debug calendar events independently."""
    targets = []

    try:
        result = _create_event(
            service=_student_service(student_user_id),
            calendar_id="primary",
            item_name=item_name,
            due_date=due_date,
            checkout_id=checkout_id,
        )
        targets.append(
            {
                "target": "student calendar",
                **result,
            }
        )
    except RuntimeError as error:
        targets.append(
            {
                "target": "student calendar",
                "ok": False,
                "reason": str(error),
            }
        )

    try:
        result = _create_event(
            service=_bot_service(),
            calendar_id=config.GOOGLE_DEBUG_CALENDAR_ID,
            item_name=item_name,
            due_date=due_date,
            checkout_id=checkout_id,
        )
        targets.append(
            {
                "target": "LabBot debug calendar",
                **result,
            }
        )
    except RuntimeError as error:
        targets.append(
            {
                "target": "LabBot debug calendar",
                "ok": False,
                "reason": str(error),
            }
        )

    event_ids = {
        "student_event_id": next(
            (
                target["event_id"]
                for target in targets
                if target["target"] == "student calendar"
                and target.get("ok")
            ),
            None,
        ),
        "debug_event_id": next(
            (
                target["event_id"]
                for target in targets
                if target["target"] == "LabBot debug calendar"
                and target.get("ok")
            ),
            None,
        ),
    }

    failures = [
        f"{target['target']}: {target['reason']}"
        for target in targets
        if not target.get("ok")
    ]

    return {
        "ok": not failures,
        "event_ids": event_ids,
        "targets": targets,
        "reason": "; ".join(failures) if failures else None,
    }


def delete_checkout_events(
    student_user_id: str,
    event_ids: dict | None,
) -> dict:
    """Delete student and debug events independently after a return."""
    event_ids = event_ids or {}
    targets = []

    try:
        result = _delete_event(
            service=_student_service(student_user_id),
            calendar_id="primary",
            event_id=event_ids.get("student_event_id"),
        )
        targets.append(
            {
                "target": "student calendar",
                **result,
            }
        )
    except RuntimeError as error:
        targets.append(
            {
                "target": "student calendar",
                "ok": False,
                "reason": str(error),
            }
        )

    try:
        result = _delete_event(
            service=_bot_service(),
            calendar_id=config.GOOGLE_DEBUG_CALENDAR_ID,
            event_id=event_ids.get("debug_event_id"),
        )
        targets.append(
            {
                "target": "LabBot debug calendar",
                **result,
            }
        )
    except RuntimeError as error:
        targets.append(
            {
                "target": "LabBot debug calendar",
                "ok": False,
                "reason": str(error),
            }
        )

    failures = [
        f"{target['target']}: {target['reason']}"
        for target in targets
        if not target.get("ok")
    ]

    return {
        "ok": not failures,
        "targets": targets,
        "reason": "; ".join(failures) if failures else None,
    }


# Compatibility helpers for your existing MCP server.
# They continue to operate only on the LabBot debug calendar.
def create_due_date_event(
    item_name: str,
    due_date: str,
    checkout_id: str,
) -> dict:
    """Create an event only in the LabBot debug calendar."""
    try:
        return _create_event(
            service=_bot_service(),
            calendar_id=config.GOOGLE_DEBUG_CALENDAR_ID,
            item_name=item_name,
            due_date=due_date,
            checkout_id=checkout_id,
        )
    except RuntimeError as error:
        return {
            "ok": False,
            "reason": str(error),
        }


def delete_event(event_id: str | None) -> dict:
    """Delete an event only from the LabBot debug calendar."""
    try:
        return _delete_event(
            service=_bot_service(),
            calendar_id=config.GOOGLE_DEBUG_CALENDAR_ID,
            event_id=event_id,
        )
    except RuntimeError as error:
        return {
            "ok": False,
            "reason": str(error),
        }