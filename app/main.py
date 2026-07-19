"""LabBot — Lab Equipment Checkout Agent."""

from pathlib import Path

from fastapi import FastAPI, Header
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import conversation_store, graph, rag, scheduler, store

app = FastAPI(title="LabBot")

rag.get_collection()
scheduler.start()   # daily 8 AM reminder emails


@app.on_event("shutdown")
def _shutdown():
    scheduler.stop()


class ChatRequest(BaseModel):
    message: str


@app.get("/api/users")
def users():
    return list(store.load_users().values())


@app.post("/api/chat")
def chat(
    req: ChatRequest,
    x_user_id: str = Header(default="", alias="X-User-Id"),
    x_conversation_id: str = Header(default="", alias="X-Conversation-Id"),
):
    user = store.get_user(x_user_id)
    conversation_id = x_conversation_id[:100] or "default"
    history = conversation_store.get_history(x_user_id, conversation_id)
    result = graph.run(req.message, user, history=history)
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