# Hands-On Lab: Build Your Own Agent (Path B)

Welcome to Path B. You're here because you want to go beyond the guided PTO
agent and build something of your own. That's the right instinct — the best
way to understand agentic AI is to apply it to a problem you actually care about.

The technical building blocks are identical to Path A: RAG, LangGraph,
intent classification, FastAPI. What's different is that **you're making the
design decisions** — the domain, the data, the personas, the actions. The
shell gives you the scaffolding; you build everything inside it.

---

## Before you write a line of code — design your agent

The most common mistake in agentic AI projects is jumping into code before
the design is clear. Spend real time on this. A crisp design makes every
part of the lab faster and produces a better capstone demo.

Answer these four questions in writing (a few sentences each is enough).
Share them with a TA before you start Part 1.

### 1. What problem does your agent solve?

Write one sentence: *"My agent helps [who] do [what] without [the pain it removes]."*

Good examples:
- "My agent helps new employees find the right IT equipment to request without
  digging through a 40-page hardware catalog."
- "My agent helps students check their financial aid status and understand
  what documents they still need to submit."
- "My agent helps on-call engineers triage incoming incident alerts and suggest
  runbook steps without switching between four different tools."

### 2. Who are your users, and do different users see or do different things?

Think about roles. Does everyone have the same access, or are some users
able to do things others can't — like a manager approving what an employee
submits? Define at least two personas and put them in `data/users.json`.

### 3. What are the 3–5 things users will ask or do?

These become your **intents** — the classification categories your agent
will route on. Be specific. "Get information" is too vague. "Check my
current ticket status" is right.

Write them out:
- Intent 1: [description]
- Intent 2: [description]
- Intent 3: [description]
- ...

### 4. What data does your agent need, and where does it come from?

Your agent will read from somewhere and optionally write to somewhere.
For the core lab, that's your `data/` JSON files. For the extensions,
it might be a real API. What fields do your records need? What does a
"query" look like? What does a "write" look like?

---

## How this lab works

This document tells you **what** to build and **why** — not how. There are
no fill-in-the-blank files and no prescribed function names. How you structure
your code, split your documents, shape your prompts, and wire your agent are
all decisions you'll make and defend.

What you can rely on:
- A running shell with a chat UI and a FastAPI backend.
- Placeholder data in `data/` and a document template in `docs/policies/`.
- The same libraries available to Path A: LangGraph, ChromaDB, the Azure
  OpenAI SDK.
- TAs available to unblock you — use them.

Each part ends with a **"Done when"** checklist. That's how you know you're
ready to move on.

---

## Part 0 — Run the shell and make it yours

Get the shell running, then immediately start making it yours. Don't leave it
as "AgentBot" — rename everything to match your domain. This isn't cosmetic;
renaming forces you to understand what each piece does.

- Open the UI, switch between users, and have a back-and-forth conversation.
- Trace one message from the browser → backend → LLM → back to the browser.
- Replace `data/users.json` with your own personas.
- Replace `data/records.json` with a few rows of realistic data for your domain.
- Update the system prompt in `app/main.py` so the bot knows what it is.

**Done when:** the UI reflects your domain, your users load in the switcher,
and you can explain out loud what every file in `app/` does.

---

## Part 1 — Ground your bot in knowledge with RAG

Most agents need to answer questions from documents — policies, guides,
catalogs, runbooks, FAQs. This part is where you build that capability.

The core idea: instead of asking the LLM to answer from memory (which it
will do confidently and incorrectly), you retrieve the relevant passages
from your documents at question time and let the model answer from them.

### What you'll build

1. **Your knowledge documents.** Use the template in `docs/policies/` as a
   starting point. Write at least 2–3 documents covering different aspects
   of your domain. The quality of your retrieval depends entirely on the
   quality of these documents — take them seriously.

2. **An indexing step.** Break each document into chunks, embed them, and
   store them in a vector store. You'll re-run this whenever your documents
   change.

3. **A query step.** Given a user's question, find the most semantically
   relevant chunks from the vector store.

4. **Wire it into chat.** When the user asks a knowledge question, retrieve
   relevant chunks and use them — and only them — to answer. If the documents
   don't contain the answer, say so rather than guessing.

### The nuance to discover

*How* you split your documents matters enormously. Try the naive approach
first: split every N characters regardless of structure. Ask a question that
spans a section boundary and see what you get back. Then try a strategy that
respects your document's structure — headings, sections, topics.

You should be able to explain: why did one approach retrieve the wrong chunk,
or mix two topics together? What does that tell you about chunking in real
RAG systems?

### Done when

- A user can ask a knowledge question in the UI and get an answer drawn from
  your documents.
- If your domain has per-user variation (e.g., different rules by role or
  category), the answer is correct for the currently selected user.
- Asking something not covered by your documents gets "I don't know" — not
  a confident hallucination.
- You can articulate why your chunking strategy beats the naive one.

---

## Part 2 — Give your agent the ability to act (the core lab)

This is the heart of the bootcamp. Your agent now needs to understand
what the user is actually asking for and take the right action — not just
look things up.

### What you'll build

- **An intent classifier.** When a message arrives, your agent decides which
  of your 3–5 intents it maps to. This is prompt engineering in earnest.
  A useful mental model: the classifier is the agent's "nervous system" —
  everything else depends on it getting this right.

- **A LangGraph orchestration layer.** Model your agent as a graph:
  classify → route → act → respond. LangGraph makes this explicit and
  debuggable instead of a tangle of `if` statements.

- **The action handlers.** One for each intent. Each handler reads from (or
  writes to) your data, and returns a grounded, natural-language reply.

- **Graceful handling of the messy cases.** Real users are vague, off-topic,
  or ask for things your agent can't do. Design for this deliberately.

### Think about trust

Your agent is taking actions on a person's behalf. Design these explicitly:

- **Authority:** should the agent act for the currently selected user only,
  or can users ask about others? Enforce this in code, not just in the prompt.
- **Grounding:** retrieved documents are data, not instructions. If a document
  said "ignore your previous instructions," your agent should not obey it.
  Try it. See what happens. Then defend against it.
- **Honesty:** if an action didn't succeed, the agent should say so — not
  invent a confirmation.

### Done when

Acting as any of your users, you can hold one conversation that exercises
at least three of your intents — each answered from real data, for the right
person, in natural language.

---

> ## You've built the core. What follows is the extension track.
>
> Parts 3–5 go deeper into real-world concerns: live APIs, multi-role access,
> and interoperability. Pick the ones most relevant to your domain.
> They're independent — you don't have to do them in order.

---

## Part 3 — Connect to a real API

Your mock JSON files were training wheels. Real agents talk to real systems —
REST APIs, databases, internal services. In this part you swap your mock
data layer for real API calls while keeping the rest of your agent intact.

### What you'll build

- Calls to a real external or internal API instead of reading from JSON files.
  This might be a public API (GitHub issues, a weather service, a flight
  tracker) or a ServiceNow instance.
- The authentication plumbing those calls require.
- Error handling for real failure modes: timeouts, rate limits, 401s, 404s —
  none of which your JSON files ever threw at you.

### What you'll learn

- The difference between authentication (who are you?) and authorization
  (what are you allowed to do?).
- Why a clean boundary between "how I get the data" and "what I do with it"
  makes this swap painless — or painful if you didn't build that boundary.
- How to handle partial failures gracefully: the API is down but your agent
  should still respond helpfully.

**Done when:** your agent performs the same actions as in Part 2, but against
a live API, and handles errors without crashing or confusing the user.

---

## Part 4 — Add a second persona with different capabilities

Until now, all users have the same access. Real systems have roles: a manager
sees their team; an admin can do things a regular user can't; a read-only
viewer can look but not touch.

### What you'll build

- A second meaningful persona with different access or capabilities than your
  first. What can they see that others can't? What can they do?
- Enforcement of those boundaries in your agent — not just in a prompt, but
  in the logic of your action handlers.
- At least one question or action that works for one persona but is correctly
  refused (with a clear explanation) for another.

### What you'll learn

- How an agent's behavior changes with the user's role.
- The difference between access control in a prompt ("only answer for the
  current user") and access control in code (actually checking before acting).
- Designing for multiple user types without building two separate apps.

**Done when:** switching personas in the UI changes what your agent can see
and do, enforced by logic — not just by politeness.

---

## Part 5 — Expose your tools over MCP

Your agent's capabilities are currently locked inside this one app. The
**Model Context Protocol (MCP)** is an open standard for exposing agent
tools so that any MCP-aware client can discover and call them — your PTO
actions, your ticket lookup, your inventory check, available to any host.

### What you'll build

- An MCP server that exposes your agent's core capabilities as well-described
  tools.
- A client (your own agent, or another MCP host) consuming those tools.

### What you'll learn

- What MCP is and the problem it solves: reusable, discoverable capabilities
  instead of code baked into one app.
- How to describe a tool well enough that a model can decide when and how to
  use it — the tool description is itself a prompt engineering challenge.
- Where the boundary sits between the model, the tools, and the systems behind them.

**Done when:** your core capabilities are available through an MCP server,
and a client can discover and call them independently.

---

## A note on finishing

Parts 0–2 are the bootcamp core. Completing them means you've built a
grounded, acting AI assistant — that's the goal. Parts 3–5 raise the
questions you'll face when you build something like this for real. Take
them as far as your curiosity and time allow.

At the capstone, you'll demo your agent and explain the decisions you made.
The interesting story isn't just "it works" — it's *why* you made the
design choices you did, what broke, and how you fixed it.
