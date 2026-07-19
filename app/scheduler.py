"""Background scheduler — sends due-date reminder emails.

Runs a daily check (default: 8 AM) and sends a reminder email for every
active checkout whose due_date is tomorrow. Uses APScheduler's
BackgroundScheduler so it runs in-process alongside uvicorn without needing
a separate worker process or cron job.

Started by app/main.py at server startup via start().

Design note: the scheduler only *sends emails* — it never modifies
checkouts.json or records.json. State changes (marking items overdue,
etc.) remain in store.py and are triggered by user actions, not by
background jobs. This keeps the scheduler's failure domain narrow: if the
Gmail send fails, the only consequence is a missed reminder, not a corrupt
data file.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

from apscheduler.schedulers.background import BackgroundScheduler

from . import gmail_client, store

log = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None


def _send_reminders(target_due_date: str | None = None) -> None:
    """Called once per day. Finds every active checkout due tomorrow and
    sends a reminder email to the student."""
    # Updates active records that have passed their due date to overdue.
    store.overdue_items()

    tomorrow = target_due_date or (
        date.today() + timedelta(days=1)
    ).isoformat()
    checkouts = store.load_checkouts()
    users = store.load_users()
    records = {r["item_id"]: r for r in store.load_records()}

    sent, failed = 0, 0
    for c in checkouts:
        if c.get("status") not in ("active", "approved", "pending"):
            continue
        if c.get("due_date") != tomorrow:
            continue

        user = users.get(c["student_id"])
        record = records.get(c["item_id"])
        item_name = record["name"] if record else c["item_id"]

        result = gmail_client.send_due_date_reminder(
            user=user,
            item_name=item_name,
            due_date=c["due_date"],
            checkout_id=c["checkout_id"],
        )
        if result.get("ok"):
            sent += 1
            log.info("Reminder sent for checkout %s (%s)", c["checkout_id"], item_name)
        else:
            failed += 1
            log.warning(
                "Reminder failed for checkout %s: %s",
                c["checkout_id"],
                result.get("reason"),
            )

    if sent or failed:
        log.info("Daily reminder run: %d sent, %d failed", sent, failed)


def start(hour: int = 8, minute: int = 0) -> None:
    """Start the background scheduler. Safe to call multiple times — only
    starts once. Called from app/main.py on server startup."""
    global _scheduler
    if _scheduler is not None:
        return

    _scheduler = BackgroundScheduler()
    _scheduler.add_job(
        _send_reminders,
        trigger="cron",
        hour=hour,
        minute=minute,
        id="due_date_reminders",
        replace_existing=True,
    )
    _scheduler.start()
    log.info("Reminder scheduler started (daily at %02d:%02d)", hour, minute)


def stop() -> None:
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)
        _scheduler = None


def run_now(target_due_date: str | None = None) -> None:
    """Run reminders immediately.

    With no argument, runs the normal daily behavior and targets equipment
    due tomorrow. Supply a date for testing, for example:

        scheduler.run_now("2026-07-20")
    """
    log.info("Manual reminder run triggered")
    _send_reminders(target_due_date=target_due_date)