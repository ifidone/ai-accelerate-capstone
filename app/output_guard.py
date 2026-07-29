"""Final output filtering for LabBot user-visible replies.

This is a last-line deterministic safety layer. It does not replace:
- authentication
- role authorization
- deterministic store validation
- trusted RAG corpus handling
- Azure provider safety filters
- response faithfulness evaluation

Its job is to prevent accidental rendering of sensitive internal identifiers,
credentials, raw technical errors, and unexpectedly large outputs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


MAX_REPLY_CHARS = 3500


@dataclass
class OutputGuardResult:
    allowed: bool
    reply: str
    reason: str | None = None


# Internal user IDs should never appear in a normal user-facing response.
# Checkout IDs such as c-abc123 remain allowed because managers need them.
INTERNAL_USER_ID_PATTERN = re.compile(
    r"\bu[0-9]+\b",
    re.IGNORECASE,
)

EMAIL_PATTERN = re.compile(
    r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b",
    re.IGNORECASE,
)

# Common credential-like patterns. If any of these appear, block rather than
# merely redact because the reply may contain additional sensitive material.
CREDENTIAL_PATTERNS = [
    re.compile(r"\bAIza[0-9A-Za-z\-_]{20,}\b"),  # Google API-key pattern
    re.compile(r"\bsk-[A-Za-z0-9\-_]{16,}\b"),   # common API-key pattern
    re.compile(r"\bBearer\s+[A-Za-z0-9\-._~+/]+=*\b", re.IGNORECASE),
    re.compile(r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b"),
    re.compile(r"refresh_token", re.IGNORECASE),
    re.compile(r"client_secret", re.IGNORECASE),
    re.compile(r"authorization:\s*bearer", re.IGNORECASE),
]

# Raw framework/provider errors should not be presented to users.
TECHNICAL_ERROR_PATTERNS = [
    re.compile(r"traceback $$$most recent call last$$$", re.IGNORECASE),
    re.compile(r"openai\.[a-z_]*error", re.IGNORECASE),
    re.compile(r"azure openai", re.IGNORECASE),
    re.compile(r"content_filter_result", re.IGNORECASE),
    re.compile(r"responsibleaipolicyviolation", re.IGNORECASE),
]


def _remove_control_characters(text: str) -> str:
    """Remove non-printable characters except normal whitespace."""
    return "".join(
        character
        for character in text
        if character.isprintable() or character in "\n\t"
    )


def filter_output(reply: str) -> OutputGuardResult:
    """Validate and sanitize a user-visible LabBot reply."""

    if not isinstance(reply, str):
        return OutputGuardResult(
            allowed=False,
            reply=(
                "I couldn't safely format that response. Please try asking "
                "your LabBot equipment question again."
            ),
            reason="non_string_reply",
        )

    cleaned = _remove_control_characters(reply).strip()

    if not cleaned:
        return OutputGuardResult(
            allowed=False,
            reply=(
                "I couldn't generate a response for that request. Please try "
                "again or ask about lab equipment availability, checkouts, "
                "returns, or policy."
            ),
            reason="empty_reply",
        )

    if len(cleaned) > MAX_REPLY_CHARS:
        return OutputGuardResult(
            allowed=False,
            reply=(
                "I couldn't safely provide that full response. Please ask a "
                "more specific LabBot equipment question."
            ),
            reason="reply_too_long",
        )

    for pattern in CREDENTIAL_PATTERNS:
        if pattern.search(cleaned):
            return OutputGuardResult(
                allowed=False,
                reply=(
                    "I can't provide sensitive system or credential "
                    "information. I can help with lab equipment availability, "
                    "checkout requests, returns, status, and policy."
                ),
                reason="credential_pattern_detected",
            )

    for pattern in TECHNICAL_ERROR_PATTERNS:
        if pattern.search(cleaned):
            return OutputGuardResult(
                allowed=False,
                reply=(
                    "I couldn't safely complete that response. Please try "
                    "again or ask a more specific LabBot question."
                ),
                reason="technical_error_pattern_detected",
            )

    # Redact rather than block common accidental leakage of internal IDs
    # and email addresses. The core user-facing answer can still be useful.
    sanitized = INTERNAL_USER_ID_PATTERN.sub(
        "[internal account]",
        cleaned,
    )

    sanitized = EMAIL_PATTERN.sub(
        "[email redacted]",
        sanitized,
    )

    if sanitized != cleaned:
        return OutputGuardResult(
            allowed=True,
            reply=sanitized,
            reason="sensitive_data_redacted",
        )

    return OutputGuardResult(
        allowed=True,
        reply=cleaned,
        reason=None,
    )