# AgentBot — Path B Shell

This is your **starting point** for Path B: Build Your Own Agent.

The shell gives you a running chat UI and a FastAPI backend that talks to an
LLM. It does *not* answer domain-specific questions, look up records, or take
actions yet — **you design and build all of that.**

**👉 The lab instructions live in [`coursework.md`](coursework.md). Start there.**

---

## What's in the box

```
app/
  config.py       env vars and paths — add your own here
  main.py         FastAPI: /api/users, /api/chat (plain chat today)
  index.html      single-file chat UI with a user switcher
data/
  users.json      placeholder users — replace with your own personas
  records.json    placeholder records — replace with your domain's data
docs/
  policies/
    TEMPLATE-policy.md   a template for your knowledge documents
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env    # then fill in your Azure OpenAI credentials

uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000>.

## What works on day one

- The UI loads with the user switcher populated from `data/users.json`.
- You can chat with the bot — it talks, but it knows nothing about your domain.
- Ask it anything domain-specific and it will tell you it can't help yet.

Everything past "it can chat" is yours to build. Head to [`coursework.md`](coursework.md).

> Add libraries to `requirements.txt` as you go. At minimum you'll likely
> want `langgraph` for orchestration and `chromadb` for your vector store.
