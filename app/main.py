"""LabBot — Lab Equipment Checkout Agent (Path B shell).

This is your starting point. The chat works: a message you type goes to an
LLM and comes back as plain conversation, and you can switch which user
you're acting as. That's all it does right now.

It cannot yet check availability, log a checkout, log a return, answer
policy questions, or approve requests. Building those capabilities is the
lab — see coursework.md.
"""

import json
from pathlib import Path

from fastapi import FastAPI, Header
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import config

app = FastAPI(title="LabBot")

# ---------------------------------------------------------------------------
# Load mock data
# ---------------------------------------------------------------------------
_USERS = {
    u["id"]: u for u in json.loads((config.DATA_DIR / "users.json").read_text())
}

_RECORDS = json.loads((config.DATA_DIR / "records.json").read_text())


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------
def _client():
    """Return an LLM client. Swap this out if you use a different provider."""
    from openai import AzureOpenAI

    if not (config.AZURE_ENDPOINT and config.AZURE_API_KEY):
        raise RuntimeError(
            "Azure OpenAI credentials are missing. Copy .env.example to .env "
            "and fill in your values."
        )
    return AzureOpenAI(
        azure_endpoint=config.AZURE_ENDPOINT,
        api_key=config.AZURE_API_KEY,
        api_version=config.AZURE_API_VERSION,
    )


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/users")
def users():
    """The users / personas available in the UI switcher."""
    return list(_USERS.values())


@app.post("/api/chat")
def chat(req: ChatRequest, x_user_id: str = Header(default="", alias="X-User-Id")):
    """
    Plain conversation with the LLM — no domain logic yet.

    TODO (later parts): This is the function you'll extend most. Eventually
    it will:
      1. Classify what the user wants (intent detection: check availability,
         request checkout, report return, ask policy, manager approve/flag).
      2. Route to the right handler (a LangGraph node or plain function),
         gated by role (student vs lab_manager).
      3. Execute the action against data/records.json and data/checkouts.json,
         or run RAG over the policy docs.
      4. Return a grounded, natural-language reply.

    For now it just talks.
    """
    user = _USERS.get(x_user_id)

    who = (
        f"You are talking to {user['name']}, a {user['role'].replace('_', ' ')}."
        if user
        else "You are talking to a user."
    )

    system = (
        "You are LabBot, an assistant that helps students and lab managers "
        "check out and return shared lab hardware like dev boards, "
        "oscilloscopes, and sensor kits. "
        + who
        + " Right now you can only make small talk. You cannot yet check "
        "equipment availability, log a checkout or return, answer policy "
        "questions, or approve requests. "
        "If the user asks for any of those, say plainly that you can't do "
        "that yet."
    )

    resp = _client().chat.completions.create(
        model=config.AZURE_CHAT_DEPLOYMENT,
        temperature=0.5,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": req.message},
        ],
    )
    return {"reply": resp.choices[0].message.content}


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "index.html")