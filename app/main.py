"""LabBot — authenticated Lab Equipment Checkout Agent."""

from __future__ import annotations

import os
import secrets
from pathlib import Path
import logging

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

from . import auth, config, conversation_store, graph, rag, scheduler, user_google_tokens

log = logging.getLogger(__name__)

if not config.SESSION_SECRET:
    raise RuntimeError(
        "SESSION_SECRET is missing. Add it to your .env before starting LabBot."
    )

# OAuthlib blocks HTTP by default. This is safe only for localhost development.
if config.OAUTHLIB_INSECURE_TRANSPORT:
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

if config.OAUTHLIB_RELAX_TOKEN_SCOPE:
    os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"

app = FastAPI(title="LabBot")

app.add_middleware(
    SessionMiddleware,
    secret_key=config.SESSION_SECRET,
    session_cookie="labbot_session",
    same_site="lax",
    https_only=config.COOKIE_HTTPS_ONLY,
    max_age=60 * 60 * 24 * 14,
)

rag.get_collection()
scheduler.start()


@app.on_event("shutdown")
def shutdown() -> None:
    scheduler.stop()


class ChatRequest(BaseModel):
    message: str


def google_flow(
    state: str | None = None,
    code_verifier: str | None = None,
) -> Flow:
    """Build a Web OAuth flow for browser login."""
    if not config.GOOGLE_WEB_CREDENTIALS_PATH.exists():
        raise RuntimeError(
            "Missing Google Web OAuth credential file at "
            f"{config.GOOGLE_WEB_CREDENTIALS_PATH}. "
            "Create a Google OAuth client of type Web application."
        )

    kwargs = {
        "client_secrets_file": str(config.GOOGLE_WEB_CREDENTIALS_PATH),
        "scopes": config.GOOGLE_LOGIN_SCOPES,
        "redirect_uri": config.GOOGLE_OAUTH_REDIRECT_URI,
    }

    if state:
        kwargs["state"] = state

    if code_verifier:
        kwargs["code_verifier"] = code_verifier

    return Flow.from_client_secrets_file(**kwargs)


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/api/auth/login")
def google_login(request: Request):
    code_verifier = secrets.token_urlsafe(64)
    flow = google_flow(code_verifier=code_verifier)

    authorization_url, state = flow.authorization_url(
        access_type="offline",
        prompt="consent",
    )

    request.session["oauth_state"] = state
    request.session["oauth_code_verifier"] = code_verifier

    log.warning(
        "OAuth login started. Session keys saved: %s",
        list(request.session.keys()),
    )

    return RedirectResponse(authorization_url, status_code=303)


@app.get("/api/auth/google/callback")
def google_callback(request: Request):
    log.warning(
        "OAuth callback received. Session keys available: %s",
        list(request.session.keys()),
    )
    expected_state = request.session.get("oauth_state")
    code_verifier = request.session.get("oauth_code_verifier")
    received_state = request.query_params.get("state")

    if not expected_state:
        raise HTTPException(
            status_code=400,
            detail=(
                "OAuth session state is missing. The browser did not send "
                "the LabBot session cookie back to the callback."
            ),
        )

    if received_state != expected_state:
        raise HTTPException(
            status_code=400,
            detail=(
                "OAuth state mismatch. Clear localhost cookies and begin "
                "a fresh sign-in attempt."
            ),
        )

    if not code_verifier:
        raise HTTPException(
            status_code=400,
            detail=(
                "PKCE verifier is missing from the session. Clear cookies "
                "and start the sign-in flow again."
            ),
        )

    flow = google_flow(
        state=expected_state,
        code_verifier=code_verifier,
    )

    flow.fetch_token(authorization_response=str(request.url))

    profile = (
        build(
            "oauth2",
            "v2",
            credentials=flow.credentials,
            cache_discovery=False,
        )
        .userinfo()
        .get()
        .execute()
    )

    user = auth.resolve_google_user(profile)

    # This token belongs to the signed-in LabBot user and enables events in
    # their own primary Google Calendar.
    user_google_tokens.save(user["id"], flow.credentials)

    request.session["user"] = user
    request.session["conversation_id"] = secrets.token_urlsafe(20)

    request.session.pop("oauth_state", None)
    request.session.pop("oauth_code_verifier", None)

    return RedirectResponse("/", status_code=303)


@app.post("/api/auth/logout")
def logout(request: Request):
    request.session.clear()
    return {"ok": True}


@app.get("/api/me")
def me(request: Request):
    return auth.current_user(request)


@app.post("/api/chat")
def chat(req: ChatRequest, request: Request):
    """Run a chat request as the Google-authenticated LabBot user."""
    user = auth.current_user(request)

    conversation_id = request.session.get("conversation_id")

    if not conversation_id:
        conversation_id = secrets.token_urlsafe(20)
        request.session["conversation_id"] = conversation_id

    history = conversation_store.get_history(
        user_id=user["id"],
        conversation_id=conversation_id,
    )

    result = graph.run(
        req.message,
        user,
        history=history,
    )

    conversation_store.append_turn(
        user_id=user["id"],
        conversation_id=conversation_id,
        user_message=req.message,
        assistant_reply=result["reply"],
    )

    return result