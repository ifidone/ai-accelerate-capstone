"""Google-authenticated LabBot user resolution.

Google proves the visitor's identity. LabBot maps the authenticated email
to a local user record, which supplies the app-specific Student or
Lab Manager role.
"""

from __future__ import annotations

from fastapi import HTTPException, Request

from . import store


def current_user(request: Request) -> dict:
    """Return the authenticated LabBot user from the signed server session."""
    user = request.session.get("user")

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Sign in with Google before using LabBot.",
        )

    return user


def resolve_google_user(profile: dict) -> dict:
    """Map an authenticated Google profile to a configured LabBot persona.

    The role is deliberately sourced from users.json, not from Google and
    not from browser input.
    """
    email = str(profile.get("email", "")).strip().lower()

    if not email:
        raise HTTPException(
            status_code=403,
            detail="Google did not return an email address for this account.",
        )

    for user in store.load_users().values():
        configured_email = str(user.get("email", "")).strip().lower()

        if configured_email == email:
            return {
                "id": user["id"],
                "name": user.get("name") or user.get("full_name") or email,
                "full_name": user.get("full_name") or user.get("name") or email,
                "email": email,
                "role": user["role"],
                "location_code": user.get("location_code", ""),
                "google_sub": str(profile.get("id", "")),
            }

    raise HTTPException(
        status_code=403,
        detail=(
            f"The Google account {email} is not configured as a LabBot user. "
            "Ask a lab manager to add it to data/users.json."
        ),
    )