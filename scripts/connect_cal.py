"""Authorize the LabBot debug account for Google Calendar and Gmail.

Run once whenever you add/change OAuth scopes or need to refresh the bot token:

    rm -f data/.bot_google_token.json
    python -m scripts.connect_bot_calendar

When Google opens the browser, sign in as the LabBot debug account:
    irene.fidoen10@gmail.com

The resulting refresh token is saved to config.GOOGLE_TOKEN_PATH and is used by:
- app/calendar_client.py to create/delete debug-calendar events
- app/gmail_client.py to send checkout, return, and reminder emails
"""

from __future__ import annotations

import json
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow

from app import config


# These are the permissions held by the bot/debug Google identity.
# Keep them separate from user-login scopes such as openid and profile.
BOT_SCOPES = [
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/gmail.send",
]


def _credential_type(path: Path) -> str:
    """Return the top-level OAuth credential type without printing secrets."""
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Could not read OAuth credentials at {path}: {exc}"
        ) from exc

    if "installed" in payload:
        return "installed"

    if "web" in payload:
        return "web"

    return "unknown"


def main() -> None:
    credentials_path = config.GOOGLE_CREDENTIALS_PATH
    token_path = config.GOOGLE_TOKEN_PATH

    if not credentials_path.exists():
        raise RuntimeError(
            f"Missing OAuth credentials file: {credentials_path}\n\n"
            "In Google Cloud Console, create an OAuth client with application "
            "type 'Desktop app', download its JSON credentials, and save it "
            "at that path."
        )

    credential_type = _credential_type(credentials_path)

    if credential_type != "installed":
        raise RuntimeError(
            f"{credentials_path.name} is a '{credential_type}' OAuth client, "
            "but this script requires a Desktop app credential file with a "
            "top-level 'installed' key.\n\n"
            "Create a separate OAuth client in Google Cloud Console:\n"
            "APIs & Services → Credentials → Create Credentials → OAuth "
            "client ID → Desktop app."
        )

    token_path.parent.mkdir(parents=True, exist_ok=True)

    print("Opening Google authorization in your browser.")
    print("Sign in as: irene.fidoen10@gmail.com")
    print("Requested permissions:")
    for scope in BOT_SCOPES:
        print(f"  - {scope}")

    flow = InstalledAppFlow.from_client_secrets_file(
        str(credentials_path),
        BOT_SCOPES,
    )

    credentials = flow.run_local_server(
        host="localhost",
        port=0,
        open_browser=True,
    )

    token_path.write_text(credentials.to_json())

    print("\nAuthorization complete.")
    print(f"Bot token saved to: {token_path}")
    print(
        f"Calendar events will be written to: "
        f"{config.GOOGLE_DEBUG_CALENDAR_ID}"
    )
    print(
        f"Debug copies of transactional emails will go to: "
        f"{config.GMAIL_DEBUG_ADDRESS}"
    )


if __name__ == "__main__":
    main()