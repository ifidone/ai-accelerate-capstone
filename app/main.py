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

from . import auth, config, chat_history, checkout_actions
from . import graph, rag, scheduler, user_google_tokens, store

log = logging.getLogger(__name__)

if not config.SESSION_SECRET:
    raise RuntimeError(
        "SESSION_SECRET is missing. Add it to your .env before starting LabBot."
    )

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
chat_history.initialize()


@app.on_event("shutdown")
def shutdown() -> None:
    scheduler.stop()


class ChatRequest(BaseModel):
    message: str

class ManagerDecisionRequest(BaseModel):
    note: str = ""

class DamageReviewRequest(BaseModel):
    status: str
    note: str = ""


class InventoryConditionRequest(BaseModel):
    status: str
    note: str = ""

def require_manager(request: Request) -> dict:
    user = auth.current_user(request)

    if user.get("role") != "lab_manager":
        raise HTTPException(
            status_code=403,
            detail="Only a lab manager can perform that action.",
        )

    return user

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

@app.get("/inventory")
def inventory_page():
    """Authenticated inventory browser page."""
    return FileResponse(Path(__file__).parent / "inventory.html")

@app.get("/my-checkouts")
def my_checkouts_page(request: Request):
    user = auth.current_user(request)

    if user.get("role") == "lab_manager":
        return RedirectResponse("/manager/requests", status_code=303)

    return FileResponse(Path(__file__).parent / "my_checkouts.html")

@app.get("/checkout-history")
def checkout_history_page(request: Request):
    user = auth.current_user(request)

    if user.get("role") == "lab_manager":
        return RedirectResponse("/manager/requests", status_code=303)

    return FileResponse(Path(__file__).parent / "checkout_history.html")

@app.get("/manager")
def manager_dashboard_page(request: Request):
    require_manager(request)
    return FileResponse(Path(__file__).parent / "manager_dashboard.html")

@app.get("/manager/damage-reports")
def manager_damage_reports_page(request: Request):
    require_manager(request)
    return FileResponse(Path(__file__).parent / "manager_damage_reports.html")

@app.delete("/api/my-checkouts/{checkout_id}")
def cancel_my_checkout_request(
    checkout_id: str,
    request: Request,
):
    user = auth.current_user(request)

    if user.get("role") == "lab_manager":
        raise HTTPException(
            status_code=403,
            detail="Lab managers use the request queue instead.",
        )

    return store.cancel_pending_request(
        user_id=user["id"],
        checkout_id=checkout_id,
    )

@app.get("/api/inventory")
def inventory(request: Request):
    """Return inventory records appropriate for the signed-in user."""
    user = auth.current_user(request)

    return {
        "items": store.inventory_catalog(
            include_manager_details=user.get("role") == "lab_manager",
        )
    }

@app.get("/api/my-checkouts")
def my_checkouts(request: Request):
    user = auth.current_user(request)

    if user.get("role") == "lab_manager":
        raise HTTPException(
            status_code=403,
            detail="Lab managers use the request queue instead.",
        )

    items = store.my_current_checkouts(user["id"])

    return {
        "items": items,
        "summary": {
            "pending": sum(item["status"] == "pending" for item in items),
            "active": sum(item["status"] == "active" for item in items),
            "overdue": sum(item["status"] == "overdue" for item in items),
            "denied": sum(item["status"] == "denied" for item in items),
        },
    }

@app.get("/api/checkout-history")
def checkout_history(request: Request):
    user = auth.current_user(request)

    if user.get("role") == "lab_manager":
        raise HTTPException(
            status_code=403,
            detail="Lab managers use the request queue instead.",
        )

    items = store.my_checkout_history(user["id"])

    return {
        "items": items,
        "summary": {
            "returned": sum(item["status"] == "returned" for item in items),
            "denied": sum(item["status"] == "denied" for item in items),
            "damage_reports": sum(
                item.get("damage_report_count", 0)
                for item in items
            ),
        },
    }

@app.get("/api/manager/summary")
def manager_summary(request: Request):
    require_manager(request)

    return store.manager_operations_summary()

@app.get("/api/manager/damage-reports")
def manager_damage_reports(request: Request):
    require_manager(request)

    reports = store.get_damage_reports()

    return {
        "reports": reports,
        "summary": {
            "total": len(reports),
            "open": sum(
                report.get("status", "open") == "open"
                for report in reports
            ),
            "reviewed": sum(
                report.get("status") == "reviewed"
                for report in reports
            ),
            "resolved": sum(
                report.get("status") == "resolved"
                for report in reports
            ),
        },
    }

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

    user_google_tokens.save(user["id"], flow.credentials)

    request.session["user"] = user

    request.session.pop("oauth_state", None)
    request.session.pop("oauth_code_verifier", None)

    return RedirectResponse("/", status_code=303)


@app.post("/api/auth/logout")
def logout(request: Request):
    request.session.clear()
    return {"ok": True}

@app.post("/api/manager/damage-reports/{report_id}/review")
def review_damage_report(
    report_id: str,
    body: DamageReviewRequest,
    request: Request,
):
    manager = require_manager(request)

    return store.update_damage_report(
        report_id=report_id,
        manager_id=manager["id"],
        new_status=body.status,
        manager_note=body.note,
    )

@app.post("/api/manager/inventory/{item_id}/condition")
def update_inventory_condition(
    item_id: str,
    body: InventoryConditionRequest,
    request: Request,
):
    manager = require_manager(request)

    return store.set_inventory_status(
        item_id=item_id,
        new_status=body.status,
        manager_id=manager["id"],
        note=body.note,
    )


@app.get("/api/me")
def me(request: Request):
    return auth.current_user(request)

@app.get("/api/history")
def get_history(request: Request):
    """Return saved history for the authenticated Google user."""
    user = auth.current_user(request)

    return chat_history.get_display_messages(
        user_id=user["id"],
        limit=100,
    )


@app.delete("/api/history")
def clear_history(request: Request):
    """Delete only the current user's saved chat history."""
    user = auth.current_user(request)

    chat_history.clear_messages(user["id"])

    return {"ok": True}

@app.get("/manager/requests")
def manager_requests_page(request: Request):
    require_manager(request)
    return FileResponse(Path(__file__).parent / "manager_requests.html")

@app.get("/api/manager/requests")
def manager_requests(request: Request):
    require_manager(request)

    return {
        "requests": store.pending_checkouts(),
        "summary": {
            "pending": len(store.pending_checkouts()),
        },
    }

@app.post("/api/manager/requests/{checkout_id}/approve")
def approve_manager_request(
    checkout_id: str,
    body: ManagerDecisionRequest,
    request: Request,
):
    manager = require_manager(request)

    return checkout_actions.approve_checkout_request(
        checkout_id=checkout_id,
        manager_id=manager["id"],
        manager_note=body.note.strip(),
    )

@app.post("/api/manager/requests/{checkout_id}/deny")
def deny_manager_request(
    checkout_id: str,
    body: ManagerDecisionRequest,
    request: Request,
):
    manager = require_manager(request)

    return checkout_actions.deny_checkout_request(
        checkout_id=checkout_id,
        manager_id=manager["id"],
        manager_note=body.note.strip(),
    )

@app.post("/api/chat")
def chat(req: ChatRequest, request: Request):
    """Run a chat request as the Google-authenticated LabBot user."""
    user = auth.current_user(request)

    history = chat_history.get_recent_messages(
        user_id=user["id"],
        limit=20,
    )

    result = graph.run(
        req.message,
        user,
        history=history,
    )

    chat_history.append_message(
        user_id=user["id"],
        role="user",
        content=req.message,
    )

    chat_history.append_message(
        user_id=user["id"],
        role="assistant",
        content=result["reply"],
    )

    return result