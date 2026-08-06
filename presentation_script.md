# LabBot Presentation Script

Two versions of the same talk: a full **prose script** you can read almost
verbatim, and a condensed **bullet-point script** for glancing at while you
present. Both follow the deck slide by slide. Slides 3–11 are one continuous
demo walkthrough, so they're scripted as a single flowing section rather than
nine separate ones — slide 15 is the full recap timeline, used to close the
demo, not narrated step by step again.

---

## Part 1 — Prose Script

### Slide 1 — Title

Hi, I'm Irene, and this is LabBot — a shared lab equipment checkout system I
built for this capstone. The problem it solves is a familiar one: students
need to find equipment, understand how long they can keep it, request it,
track due dates, and report returns. Lab managers need to approve those
requests, watch for overdue equipment, and manage inventory and repair
status. Before LabBot, that was a spreadsheet. LabBot replaces spreadsheet
tracking with a Google-authenticated, role-aware checkout agent — you talk to
it in plain language, and it handles the rest.

### Slide 2 — Student and Lab Manager Workflows

There are two personas here, and they get very different capabilities.
Students can browse inventory, ask policy questions — that's backed by
retrieval over the real policy documents — request a checkout, view their
pending and active checkouts, cancel a pending request, report a return,
report damage, and review their history. Lab managers get a different set of
tools entirely: reviewing and approving or rejecting requests, an operations
dashboard, monitoring overdue equipment, sending reminders, reviewing damage
reports, and updating equipment condition.

The part I want to highlight is the security boundary at the bottom. Google
login identifies the user. The backend maps their email to a role — student
or lab manager. And deterministic code — not the language model — authorizes
every single write. That distinction matters a lot for what's coming later in
this talk.

### Slides 3–11 — Live Demo: Request to Approval

Let me walk through one full workflow end to end, across both personas,
because I think it's the clearest way to show how the pieces fit together.

It starts with the student. They ask a policy question in chat — "How long
can I check out an ESP32?" — and the agent answers correctly using the real
policy document, not a guess. Then they check inventory: the ESP32 is
available, up to three days. So they request it directly in the same
conversation — "Can I check out an ESP32 for 3 days?" — and the system
creates the request. If they check My Checkouts right after, they'll see it
sitting there as pending manager approval — not active yet, because a
student's own request is never enough to check something out on its own.

Now we switch to the lab manager. They open the request queue and see the
pending request with the student's info attached. They approve it — optionally
attaching a note — and at that point three things happen together: the
checkout moves from pending to active, the inventory record flips to checked
out, and LabBot creates a calendar due-date event and sends a confirmation
email.

Switching back to the student: they refresh My Checkouts and now it shows
active, with the due date visible, days remaining calculated, and the
approving manager's name recorded. Later, when they're done, they just tell
the chat "I'm returning the ESP32" — no separate return form, no navigating
anywhere — and the system closes the loop, cleans up the calendar event, and
sends a return confirmation.

The reason I'm walking through the whole thing instead of just describing
each screen is that it's the same underlying agent handling every single one
of these steps — the natural language interface doesn't change based on
whether you're a student or a manager, or whether you're asking, requesting,
approving, or returning. What changes is what the backend allows you to do.

### Slide 12 — Architecture and Design Choice

This is the design decision I most want to talk about: where model reasoning
belongs, and where it explicitly does not.

Every request goes through a LangGraph state machine. It's a state object —
the message, the authenticated user, their role, conversation history — that
flows through a graph of nodes. The first node classifies intent: is this an
availability check, a checkout request, a cancellation, a return, a policy
question, a manager action? That's one LLM call, and I deliberately run it on
Haiku, because it's a short, structured, low-creativity task — classifying a
label doesn't need a slow, expensive model.

From there, the graph routes to one of eleven action nodes, one per intent.
Each of those nodes does a little bit of LLM-based extraction — pulling the
item name, the requested number of days, the checkout ID — and then hands off
to a deterministic store layer written in plain Python. That store layer is
where the real rules live: role checks, ownership checks, the two-item cap,
overdue holds, the pending-to-active transition, inventory state changes.
Every one of those is a hard rule enforced in code — none of it is left to
the model's judgment.

Every node, regardless of intent, converges on one final response node before
the graph ends, and that's where the natural-language reply gets generated.
That one runs on Sonnet instead of Haiku, because that prompt is explicitly
instructed never to invent IDs, policies, or outcomes — that's the one place
in the pipeline where a model mistake would actually reach the user, so it
gets the stronger model.

The principle underneath all of this: I never let the model decide whether an
action is allowed. LangGraph routes the request and generates the language
around it, but authorization is deterministic. That makes every action
auditable, and it means no amount of clever prompt wording can bypass a
checkout policy or an authorization check — because the model was never the
one enforcing it in the first place.

### Slide 13 — Evaluation, Safety, and a Failure I Fixed

An agent you can't evaluate is an agent you can't ship, so I built a
regression suite against a golden dataset — 28 real conversational cases
covering every intent the agent supports. Right now it's at 100% intent
accuracy, 100% result correctness against the deterministic store layer, and
a 5 out of 5 on an LLM-judged response faithfulness score — meaning the
natural-language reply never contradicts, invents, or omits what actually
happened.

That faithfulness judge doesn't grade tone or writing style — it only checks
whether the reply is honest about the result. And I didn't just trust that
score at face value. I built a calibration set: fixed, hand-written replies
of known quality — some deliberately unfaithful, some genuinely faithful —
and ran them through the same judge to prove it actually discriminates
between good and bad rather than just scoring everything the same. That's 16
out of 16 calibration cases passing, split across both directions: catching
real failures, and not punishing correct behavior.

I also want to be honest about a real failure this process caught. During
testing, the phrase "What requests are waiting for my approval?" was being
routed as a personal status check instead of the manager request queue — a
classification miss. Because I had a regression suite in place, I caught it
immediately, added deterministic routing for that specific phrasing pattern,
and the full suite went back to 100%. That's the actual value of building
evaluation in from the start — it's not a one-time check, it's the thing that
tells you when something breaks.

On the safety side: every identity is Google-authenticated, every role and
ownership check happens in the backend, the RAG corpus is limited to an
allowlisted set of real policy documents so the model can't be fed arbitrary
content, there's a provider-level content filter with graceful handling
rather than a crash, and an output filter that strips internal IDs, emails,
and raw error text before anything reaches the user. And all of that is
covered by an adversarial test suite specifically probing for prompt
injection and policy bypass attempts — not just happy-path testing.

### Slide 14 — Impact, Learning, and Next Steps

For students, this means less uncertainty about equipment access, a clear
view of pending requests and due dates, and an easier path to reporting
returns and damage. For lab managers, it's a faster approval workflow, one
central request queue instead of scattered emails, and real visibility into
overdue and damaged equipment.

What I learned building this: separate LLM reasoning from deterministic
action logic early, because retrofitting that boundary is much harder than
designing for it from the start. Auth and role gates belong in backend code,
never in a prompt. Evaluation is regression infrastructure, not a phase you
do once before shipping. Design integrations — calendar, email — for partial
failure from day one, because they will fail sometimes and the core action
still has to succeed. And test the safety layer adversarially, not just on
the happy path, because that's where the real risks actually show up.

Looking ahead: I'd move off flat JSON files onto a real transactional
database, encrypt the persisted OAuth tokens rather than storing them in
plaintext, add a retry queue for calendar and email failures instead of just
surfacing them, build out manager-facing analytics on utilization and damage
patterns, and bind the MCP tool identity to the same production auth context
the web app uses today.

At its core, LabBot helps students and lab managers manage shared hardware
safely and transparently — combining role-aware workflows, deterministic
business rules, and an agent whose behavior is actually regression-tested,
not just demoed once and hoped for.

### Slide 15 — Recap

*(This slide shows the full twelve-step timeline from the live demo in one
view — use it to close out the demo section visually. No new narration
needed; a short line like "so that's the complete loop, start to finish" is
enough before moving to Q&A.)*

---

## Part 2 — Bullet-Point Script

### Slide 1 — Title
- LabBot: shared lab equipment, checked out and tracked
- Problem: students need to find/understand/request/track/return equipment; managers need to approve, monitor overdue, manage inventory
- Solution: replaces spreadsheet tracking with a Google-authenticated, role-aware checkout agent

### Slide 2 — Workflows & Security
- Two personas, two different capability sets — same chat interface
- Student: browse, ask policy (RAG), request, view status, cancel, return, report damage, history
- Manager: approve/reject, dashboard, overdue monitoring, reminders, damage review, inventory condition updates
- Security boundary: Google login → backend maps email to role → **deterministic code** authorizes every write

### Slides 3–11 — Live Demo Walkthrough
- Student asks a policy question in chat → correct, RAG-grounded answer
- Student checks availability → ESP32 available, up to 3 days
- Student requests it in the same conversation → request created
- My Checkouts shows **pending manager approval**, not active
- Manager opens request queue → sees pending request + student info
- Manager approves (optional note) → checkout goes active, inventory flips to checked out, calendar event + confirmation email created together
- Student refreshes My Checkouts → active, due date, days remaining, approver name all visible
- Student returns via plain chat message — no separate form
- **Key point:** same agent, same interface, for every step — backend permissions change, not the UI

### Slide 12 — Architecture and Design Choice
- LangGraph = state machine, not the decision-maker
- `AgentState` carries message, authenticated user/role, history, intent, result
- Node 1: intent classification — one LLM call, runs on **Haiku** (short, structured, low-stakes)
- Conditional routing → one of 11 action nodes per intent
- Action nodes: light LLM extraction (item, days, ID) → hands off to deterministic `store.py`
- Store layer owns: role checks, ownership checks, item cap, overdue holds, pending→active transition, inventory state
- All nodes converge on one response node → runs on **Sonnet** (explicitly forbidden from inventing IDs/policies/outcomes)
- Core principle: **the model never decides if an action is allowed** — auditable, prompt-injection-resistant by construction

### Slide 13 — Evaluation, Safety, and a Failure I Fixed
- 28 golden test cases, 100% intent accuracy, 100% result correctness
- 5/5 LLM-judged response faithfulness (checks honesty vs. result, not tone)
- Judge is **calibrated**, not just trusted: 16/16 on fixed known-good/known-bad replies, both directions tested
- Real failure caught: "what requests are waiting for my approval" misrouted as personal status → fixed with deterministic routing → suite back to 100%
- Safety controls: Google auth identity, backend role/ownership checks, allowlisted RAG corpus, provider content filter (graceful, not a crash), output filter (strips IDs/emails/raw errors), adversarial guardrail suite
- What's tested: role gates, approval lifecycle, item cap/overdue, cross-user access attempts, damage workflows, calendar/email partial failure, multi-turn context, prompt injection

### Slide 14 — Impact, Learning, Next Steps
- Students: less uncertainty, clear pending/due-date visibility, easier returns/damage reporting
- Managers: faster approvals, central queue, overdue/damage visibility
- Learned: separate reasoning from action logic early; auth belongs in code not prompts; evaluation is infrastructure, not a phase; design for partial integration failure; test safety adversarially
- Next: real transactional DB (off JSON), encrypt OAuth tokens, retry queue for Calendar/Gmail, manager analytics, bind MCP identity to production auth
- Closing line: role-aware workflows + deterministic business rules + regression-tested agent behavior

### Slide 15 — Recap
- Full 12-step timeline in one view — visual close to the demo section
- One line into Q&A: "that's the complete loop, start to finish"
