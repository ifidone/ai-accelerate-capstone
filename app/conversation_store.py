from __future__ import annotations

from collections import OrderedDict, deque
from threading import Lock

MAX_CONVERSATIONS = 200
MAX_TURNS = 12

_lock = Lock()
_conversations: OrderedDict[tuple[str, str], deque[dict]] = OrderedDict()


def get_history(user_id: str, conversation_id: str) -> list[dict]:
    key = (user_id, conversation_id)

    with _lock:
        history = _conversations.get(key)
        if history is None:
            history = deque(maxlen=MAX_TURNS)
            _conversations[key] = history

        _conversations.move_to_end(key)
        return list(history)


def append_turn(
    user_id: str,
    conversation_id: str,
    user_message: str,
    assistant_reply: str,
) -> None:
    key = (user_id, conversation_id)

    with _lock:
        history = _conversations.setdefault(key, deque(maxlen=MAX_TURNS))
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": assistant_reply})
        _conversations.move_to_end(key)

        while len(_conversations) > MAX_CONVERSATIONS:
            _conversations.popitem(last=False)