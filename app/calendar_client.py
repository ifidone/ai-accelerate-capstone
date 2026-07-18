"""Google Calendar integration — the live external API for Part 3.

Authentication vs authorization, concretely:
  - Authentication: OAuth 2.0 installed-app flow. The FIRST time this runs,
    it opens a browser for you to sign into a Google account and consent.
    After that, a refresh token is cached in config.GOOGLE_TOKEN_PATH and
    used silently — no repeated logins. This proves "LabBot is allowed to
    act on behalf of this one Google account" (the service identity, e.g.
    the lab manager's account, however you set it up).
  - Authorization: SCOPES below is deliberately narrow —
    'calendar.events' only, not full calendar access. LabBot can create and
    delete events it made; it cannot read your other calendars, invite
    guests to unrelated meetings, or see anything beyond events. That's the
    principle of least privilege applied to a real OAuth scope, not just a
    design essay.

Every public function here returns the same {"ok": bool, ...} shape used
throughout app/store.py. That's the boundary the coursework is pointing at:
app/graph.py's action nodes don't need to know or care whether a given
result dict came from a JSON file or a live REST call — same shape, same
handling, same respond_node. That's what makes this swap (relatively)
painless.
"""

from __future__ import annotations

from . import config

SCOPES = ["https://www.googleapis.com/auth/calendar.events"]

_service_cache = None


def _get_credentials():
    from google.auth.exceptions import RefreshError
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    creds = None
    if config.GOOGLE_TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(config.GOOGLE_TOKEN_PATH), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except RefreshError:
                creds = None
        if not creds:
            if not config.GOOGLE_CREDENTIALS_PATH.exists():
                raise RuntimeError(
                    f"Missing {config.GOOGLE_CREDENTIALS_PATH}. In Google Cloud "
                    "Console, create an OAuth client (type: Desktop app), "
                    "download the JSON, and save it at that path."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(config.GOOGLE_CREDENTIALS_PATH), SCOPES)
            creds = flow.run_local_server(port=0)  # opens a browser for consent
        config.GOOGLE_TOKEN_PATH.write_text(creds.to_json())

    return creds


def _service():
    global _service_cache
    if _service_cache is None:
        from googleapiclient.discovery import build
        _service_cache = build("calendar", "v3", credentials=_get_credentials())
    return _service_cache


def _describe_http_error(e) -> str:
    status = getattr(e.resp, "status", None)
    if status == 401:
        return "Calendar authentication expired — LabBot may need to be reauthorized."
    if status == 403:
        return "Calendar API rate limit or quota exceeded — try again shortly."
    if status == 404:
        return "That calendar event no longer exists."
    return f"Calendar API error (status {status})."


def create_due_date_event(item_name: str, due_date: str, checkout_id: str) -> dict:
    """Create an all-day event on the due date. NEVER raises — any failure
    (auth, quota, network, timeout) comes back as {"ok": False, "reason":
    ...} so a calendar outage can never take down a checkout that otherwise
    succeeded in app/store.py. This is the "partial failure" requirement:
    the write to your real system of record already happened; the calendar
    sync is best-effort on top of it.
    """
    try:
        from googleapiclient.errors import HttpError

        event = {
            "summary": f"Lab equipment due: {item_name}",
            "description": f"LabBot checkout {checkout_id}",
            "start": {"date": due_date},
            "end": {"date": due_date},
            "reminders": {"useDefault": True},
        }
        created = _service().events().insert(calendarId=config.GOOGLE_CALENDAR_ID, body=event).execute()
        return {"ok": True, "event_id": created["id"]}
    except HttpError as e:
        return {"ok": False, "reason": _describe_http_error(e)}
    except RuntimeError as e:
        # Raised by _get_credentials when there's no client secrets file yet.
        return {"ok": False, "reason": str(e)}
    except (ConnectionError, TimeoutError) as e:
        return {"ok": False, "reason": f"Could not reach Google Calendar ({e})."}
    except Exception as e:  # last-resort catch: calendar sync must never crash a checkout
        return {"ok": False, "reason": f"Unexpected calendar error: {e}"}


def delete_event(event_id: str | None) -> dict:
    """Delete a previously-created event. A missing event_id or an
    already-gone event (404) both count as success — the end state (no
    event) is what we wanted either way."""
    if not event_id:
        return {"ok": True, "note": "no calendar event was on file for this checkout"}
    try:
        from googleapiclient.errors import HttpError

        _service().events().delete(calendarId=config.GOOGLE_CALENDAR_ID, eventId=event_id).execute()
        return {"ok": True}
    except HttpError as e:
        status = getattr(e.resp, "status", None)
        if status == 404:
            return {"ok": True, "note": "event was already removed"}
        return {"ok": False, "reason": _describe_http_error(e)}
    except RuntimeError as e:
        return {"ok": False, "reason": str(e)}
    except (ConnectionError, TimeoutError) as e:
        return {"ok": False, "reason": f"Could not reach Google Calendar ({e})."}
    except Exception as e:
        return {"ok": False, "reason": f"Unexpected calendar error: {e}"}