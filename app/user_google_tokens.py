"""Server-side storage for signed-in users' Google OAuth credentials.

Each credential file contains a refresh token granted during Google login.
This lets LabBot create and delete due-date events in the student's own
Google Calendar even when a manager later approves the request.

For a production deployment, replace these JSON files with encrypted
database or secrets-manager storage.
"""

from __future__ import annotations

import json
from pathlib import Path

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

from . import config


def _token_path(user_id: str) -> Path:
    """Return a safe server-side path for a known local LabBot user ID."""
    safe_user_id = "".join(
        character
        for character in user_id
        if character.isalnum() or character in {"-", "_"}
    )

    if not safe_user_id:
        raise ValueError("Invalid Supply Sage user ID.")

    return config.USER_GOOGLE_TOKEN_DIR / f"{safe_user_id}.json"


def save(user_id: str, credentials: Credentials) -> None:
    """Persist a signed-in user's OAuth credentials server-side."""
    path = _token_path(user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(credentials.to_json())


def load(user_id: str) -> Credentials | None:
    """Load and refresh a user's Calendar credential when possible."""
    path = _token_path(user_id)

    if not path.exists():
        return None

    try:
        credentials = Credentials.from_authorized_user_info(
            json.loads(path.read_text())
        )
    except (OSError, json.JSONDecodeError, ValueError):
        return None

    if credentials.valid:
        return credentials

    if not credentials.expired or not credentials.refresh_token:
        return None

    try:
        credentials.refresh(Request())
    except RefreshError:
        return None

    path.write_text(credentials.to_json())
    return credentials


def is_connected(user_id: str) -> bool:
    """Return whether the user has a usable saved Google Calendar token."""
    return load(user_id) is not None