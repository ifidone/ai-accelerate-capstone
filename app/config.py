"""Configuration — environment variables and paths."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs" / "policies"
CHROMA_DIR = ROOT / ".chroma"

# ---------------------------------------------------------------------------
# Azure OpenAI
# ---------------------------------------------------------------------------
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini")
AZURE_EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-small")

# ---------------------------------------------------------------------------
# Google OAuth — shared across login + calendar
# ---------------------------------------------------------------------------
# Download an OAuth client (Desktop app) from Google Cloud Console.
# The same client_secrets file covers both user login and calendar access
# because we request all scopes in one consent flow.
GOOGLE_CREDENTIALS_PATH = ROOT / "google_credentials.json"
GOOGLE_TOKEN_PATH = ROOT / ".google_token.json"     # service-level / debug token

GOOGLE_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/gmail.send",
]

# The redirect URI registered in Google Cloud Console → OAuth client → Authorized redirect URIs.
# For local dev: http://localhost:8000/auth/callback
# Update this when you deploy.
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/callback")

# Extracted from google_credentials.json at import time (needed to verify id_tokens).
_creds_path = GOOGLE_CREDENTIALS_PATH
GOOGLE_CLIENT_ID: str = ""
if _creds_path.exists():
    import json as _json
    _raw = _json.loads(_creds_path.read_text())
    _web_or_installed = _raw.get("web") or _raw.get("installed") or {}
    GOOGLE_CLIENT_ID = _web_or_installed.get("client_id", "")

# ---------------------------------------------------------------------------
# Google Calendar
# ---------------------------------------------------------------------------
# The debug/admin calendar that always receives a copy of every event.
# Set this to your own Google Calendar ID (usually your Gmail address for
# the primary calendar, or a specific calendar's ID from Calendar settings).
GOOGLE_DEBUG_CALENDAR_ID = os.getenv("GOOGLE_DEBUG_CALENDAR_ID", "primary")
GOOGLE_CALENDAR_ID = GOOGLE_DEBUG_CALENDAR_ID

# ---------------------------------------------------------------------------
# Gmail
# ---------------------------------------------------------------------------
# BCC address that receives a copy of every outbound email — lets you see
# all student emails during development without logging into student accounts.
GMAIL_DEBUG_ADDRESS = os.getenv("GMAIL_DEBUG_ADDRESS", "irene.fidone10@gmail.com")

# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------
# Generate a strong random secret: python -c "import secrets; print(secrets.token_hex(32))"
# Add it to your .env as SESSION_SECRET=...
SESSION_SECRET = os.getenv("SESSION_SECRET", "change-me-in-production")

# ---------------------------------------------------------------------------
# Role mapping — emails that get lab_manager role on login
# ---------------------------------------------------------------------------
LAB_MANAGER_EMAILS: set[str] = set(
    e.strip() for e in os.getenv("LAB_MANAGER_EMAILS", "irene.fidone10@gmail.com").split(",") if e.strip()
)

# ---------------------------------------------------------------------------
# RAG corpus trust boundary
# ---------------------------------------------------------------------------
# Auto-snapshot at import time — files added after server start won't be
# indexed until restart. See app/rag.py::_load_docs().
TRUSTED_POLICY_SOURCES = {p.name for p in DOCS_DIR.glob("*.md")} if DOCS_DIR.exists() else set()