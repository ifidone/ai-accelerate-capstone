"""LabBot — Lab Equipment Checkout Agent.

Part 0 shipped plain chat. Part 1 adds RAG: policy questions are answered
from docs/policies/*.md via retrieval, not from the model's memory.

Flow for every chat message:
  1. classify_intent() — cheap LLM call, "policy" or "chat".
  2. "policy"  -> rag.query() for the top chunks, answer grounded in them.
     "chat"    -> same small-talk path as Part 0.

Still no write actions (checkout/return/approve) — that's a later part.
"""

import json
from pathlib import Path

from fastapi import FastAPI, Header
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import config, rag

app = FastAPI(title="LabBot")

# ---------------------------------------------------------------------------
# Load mock data
# ---------------------------------------------------------------------------
_USERS = {
    u["id"]: u for u in json.loads((config.DATA_DIR / "users.json").read_text())
}

_RECORDS = json.loads((config.DATA_DIR / "records.json").read_text())

# Build the RAG index once at startup (structure-aware — see app/rag.py).
# Re-run scripts/build_index.py whenever docs/policies/*.md changes instead
# of relying on server restarts, once you're iterating on doc content.
rag.get_collection()


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
# Intent classification
# ---------------------------------------------------------------------------
def classify_intent(message: str) -> str:
    """Returns "policy" if the message is asking about a rule/policy that
    should be looked up in the docs, otherwise "chat".

    This is intentionally a single cheap classification call, not a full
    LangGraph router yet — that structure comes in a later part once there
    are more than two branches (availability, checkout, return, approve...).
    """
    system = (
        "Classify the user's message into exactly one label: "
        "'policy' if they're asking about a rule, duration, fee, deadline, "
        "or procedure for checking out, returning, damaging, or reserving "
        "lab equipment; 'chat' for anything else (greetings, small talk, "
        "questions unrelated to lab equipment policy). "
        "Reply with only the single word label, nothing else."
    )
    resp = _client().chat.completions.create(
        model=config.AZURE_CHAT_DEPLOYMENT,
        temperature=0,
        max_tokens=5,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": message},
        ],
    )
    label = (resp.choices[0].message.content or "").strip().lower()
    return "policy" if "policy" in label else "chat"


# ---------------------------------------------------------------------------
# Policy (RAG) answering
# ---------------------------------------------------------------------------
def answer_policy_question(message: str, user: dict | None) -> str:
    chunks = rag.query(message, k=4)

    if not chunks:
        return "I don't have any policy documents indexed yet, so I can't answer that."

    context = "\n\n---\n\n".join(
        f"[{c['source']} — {c['heading']}]\n{c['text']}" for c in chunks
    )

    role = user["role"] if user else "unknown"
    who = f"The user is {user['name']}, role: {role}." if user else "The user's role is unknown."

    system = (
        "You are LabBot, answering a policy question about lab equipment "
        "checkout using ONLY the context passages below. "
        + who
        + " If the passages mention role-specific rules (e.g. something only "
        "a lab manager can do), make that distinction clear and relevant to "
        "the current user's role. "
        "If the passages do not contain enough information to answer, say "
        "plainly that you don't know / it isn't covered in the policy docs — "
        "do not guess or fill in gaps from general knowledge. "
        "Keep the answer concise and cite which policy section it's drawn "
        "from (e.g. 'per the Checkout Policy's Renewals section').\n\n"
        f"CONTEXT:\n{context}"
    )

    resp = _client().chat.completions.create(
        model=config.AZURE_CHAT_DEPLOYMENT,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": message},
        ],
    )
    return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# Plain chat (Part 0 behavior)
# ---------------------------------------------------------------------------
def answer_chat(message: str, user: dict | None) -> str:
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
        + " You can answer policy questions grounded in the lab's policy "
        "documents. You cannot yet check equipment availability, log a "
        "checkout or return, or approve requests — if asked, say plainly "
        "that you can't do that yet."
    )
    resp = _client().chat.completions.create(
        model=config.AZURE_CHAT_DEPLOYMENT,
        temperature=0.5,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": message},
        ],
    )
    return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/users")
def users():
    """The users / personas available in the UI switcher."""
    return list(_USERS.values())


@app.post("/api/chat")
def chat(req: ChatRequest, x_user_id: str = Header(default="", alias="X-User-Id")):
    user = _USERS.get(x_user_id)

    intent = classify_intent(req.message)
    if intent == "policy":
        reply = answer_policy_question(req.message, user)
    else:
        reply = answer_chat(req.message, user)

    return {"reply": reply, "intent": intent}


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "index.html")