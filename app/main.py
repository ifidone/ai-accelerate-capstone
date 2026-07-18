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
def chat(req: ChatRequest, x_user_id: str = Header(default="", alias="X-User-Id")):
    user = store.get_user(x_user_id)
    result = graph.run(req.message, user)
    # `intent` and `result` are returned alongside `reply` so you can see them
    # in the UI's "Details" disclosure (index.html already renders `extra`
    # if you pass it — wire it up if you want visibility while debugging).
    return result


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "index.html")