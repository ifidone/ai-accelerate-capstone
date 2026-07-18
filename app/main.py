"""LabBot — Lab Equipment Checkout Agent.

Part 0: plain chat.
Part 1: RAG-grounded policy answers.
Part 2: full intent graph (app/graph.py) — availability, checkout, return,
time remaining, policy, and manager actions, each a real action against
data/records.json and data/checkouts.json via app/store.py.

main.py itself is now thin: load data for the /api/users endpoint, build
the RAG index at startup, and hand every chat message to the graph.
"""

from pathlib import Path

from fastapi import FastAPI, Header
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import graph, rag, store
from . import conversation_store, graph, rag, store

app = FastAPI(title="LabBot")

# Build the RAG index once at startup (structure-aware — see app/rag.py).
# Re-run scripts/build_index.py whenever docs/policies/*.md changes.
rag.get_collection()


class ChatRequest(BaseModel):
    message: str


@app.get("/api/users")
def users():
    """The users / personas available in the UI switcher."""
    return list(store.load_users().values())

@app.post("/api/chat")
def chat(
    req: ChatRequest,
    x_user_id: str = Header(default="", alias="X-User-Id"),
    x_conversation_id: str = Header(default="", alias="X-Conversation-Id"),
):
    user = store.get_user(x_user_id)

    # Keep the identifier bounded so arbitrary headers cannot create
    # unbounded keys in the in-memory conversation store.
    conversation_id = x_conversation_id[:100] or "default"

    history = conversation_store.get_history(x_user_id, conversation_id)
    result = graph.run(req.message, user, history)

    conversation_store.append_turn(
        user_id=x_user_id,
        conversation_id=conversation_id,
        user_message=req.message,
        assistant_reply=result["reply"],
    )

    return result


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "index.html")