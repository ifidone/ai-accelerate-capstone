"""LabBot configuration — paths, OAuth clients, integrations, and secrets."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs" / "policies"
CHROMA_DIR = ROOT / ".chroma"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Azure OpenAI
# ---------------------------------------------------------------------------
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.getenv(
    "AZURE_OPENAI_API_VERSION",
    "2024-08-01-preview",
)
AZURE_EMBED_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_EMBED_DEPLOYMENT",
    "text-embedding-3-small",
)

# ---------------------------------------------------------------------------
# Anthropic (Claude) — chat completions
# ---------------------------------------------------------------------------
# All chat completions (classification, extraction, response generation) go
# through Claude. Embeddings stay on Azure OpenAI above, since Anthropic has
# no embeddings endpoint.
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL")
ANTHROPIC_HAIKU_MODEL = os.getenv("ANTHROPIC_HAIKU_MODEL", "claude-haiku-4-5")
ANTHROPIC_SONNET_MODEL = os.getenv("ANTHROPIC_SONNET_MODEL", "claude-sonnet-4-6")

# ---------------------------------------------------------------------------
# Google OAuth — browser login / LabBot personas
# ---------------------------------------------------------------------------
# This must be a Google OAuth client of type "Web application".
#
# It is separate from google_credentials.json, which is the Desktop OAuth
# client used by the LabBot Calendar/Gmail automation account.
GOOGLE_WEB_CREDENTIALS_PATH = ROOT / os.getenv(
    "GOOGLE_WEB_CREDENTIALS",
    "google_web_credentials.json",
)

# Must exactly match the Authorized redirect URI configured for the Web
# OAuth client in Google Cloud Console.
GOOGLE_OAUTH_REDIRECT_URI = os.getenv(
    "GOOGLE_OAUTH_REDIRECT_URI",
    "http://localhost:8000/api/auth/google/callback",
)

# These scopes identify the person signing into LabBot. Calendar and Gmail
# permissions belong to the separate bot/debug OAuth identity below.
GOOGLE_LOGIN_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/calendar.events",
]

# User OAuth refresh tokens are stored server-side by local LabBot user ID.
# For production, use encrypted database storage or a secrets manager.
USER_GOOGLE_TOKEN_DIR = DATA_DIR / "user_google_tokens"
USER_GOOGLE_TOKEN_DIR.mkdir(parents=True, exist_ok=True)

# Calendar scope used when loading a signed-in user's saved credentials.
USER_GOOGLE_CALENDAR_SCOPES = [
    "https://www.googleapis.com/auth/calendar.events",
]

# ---------------------------------------------------------------------------
# Google OAuth — LabBot debug Calendar and Gmail automation account
# ---------------------------------------------------------------------------
# This must be a Google OAuth client of type "Desktop app".
#
# It is used by:
#   python -m scripts.connect_bot_calendar
#
# The script authorizes the account that creates Calendar events and sends
# LabBot Gmail messages.
GOOGLE_CREDENTIALS_PATH = ROOT / os.getenv(
    "GOOGLE_BOT_CREDENTIALS",
    "google_credentials.json",
)

# Keep this aligned with scripts/connect_bot_calendar.py,
# app/calendar_client.py, and app/gmail_client.py.
GOOGLE_TOKEN_PATH = ROOT / os.getenv(
    "GOOGLE_BOT_TOKEN_PATH",
    ".google_token.json",
)

# These scopes are for the LabBot automation account only.
GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/gmail.send",
]

# ---------------------------------------------------------------------------
# Google Calendar and Gmail
# ---------------------------------------------------------------------------
# "primary" writes events to the primary calendar of whichever account was
# authorized by scripts/connect_bot_calendar.py.
GOOGLE_DEBUG_CALENDAR_ID = os.getenv(
    "GOOGLE_DEBUG_CALENDAR_ID",
    "primary",
)

# Compatibility name expected by the current calendar_client.py.
GOOGLE_CALENDAR_ID = GOOGLE_DEBUG_CALENDAR_ID

# Receives a debug copy of all outgoing confirmation/reminder emails.
GMAIL_DEBUG_ADDRESS = os.getenv(
    "GMAIL_DEBUG_ADDRESS",
    "irene.fidone10@gmail.com",
)

# ---------------------------------------------------------------------------
# Browser sessions and local OAuth development
# ---------------------------------------------------------------------------
SESSION_SECRET = os.getenv("SESSION_SECRET", "")

COOKIE_HTTPS_ONLY = (
    os.getenv("COOKIE_HTTPS_ONLY", "false").strip().lower()
    in {"1", "true", "yes"}
)

# OAuthlib normally requires HTTPS. Enable only for localhost development.
# Do not set this to true in a deployed environment.
OAUTHLIB_INSECURE_TRANSPORT = (
    os.getenv("OAUTHLIB_INSECURE_TRANSPORT", "false").strip().lower()
    in {"1", "true", "yes"}
)

OAUTHLIB_RELAX_TOKEN_SCOPE = (
    os.getenv("OAUTHLIB_RELAX_TOKEN_SCOPE", "false").strip().lower()
    in {"1", "true", "yes"}
)

# ---------------------------------------------------------------------------
# Optional email-to-role mapping
# ---------------------------------------------------------------------------
# Your current Google-login implementation should map authenticated emails
# against data/users.json. This allowlist is available if you later choose
# to derive manager role directly from environment configuration.
LAB_MANAGER_EMAILS = {
    email.strip().lower()
    for email in os.getenv(
        "LAB_MANAGER_EMAILS",
        "ifidone@andrew.cmu.edu",
    ).split(",")
    if email.strip()
}

# ---------------------------------------------------------------------------
# RAG corpus trust boundary
# ---------------------------------------------------------------------------
TRUSTED_POLICY_SOURCES = (
    {path.name for path in DOCS_DIR.glob("*.md")}
    if DOCS_DIR.exists()
    else set()
)