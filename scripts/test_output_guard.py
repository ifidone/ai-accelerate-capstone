"""Simple deterministic tests for LabBot's final output filter.

Run:
    python -m scripts.test_output_guard
"""

from app.output_guard import filter_output


def assert_equal(actual, expected, label):
    if actual != expected:
        raise AssertionError(
            f"{label}: expected {expected!r}, received {actual!r}"
        )


def main():
    safe = filter_output(
        "Your sensor kit request is pending manager approval."
    )
    assert_equal(safe.allowed, True, "safe reply allowed")
    assert_equal(
        safe.reply,
        "Your sensor kit request is pending manager approval.",
        "safe reply unchanged",
    )

    internal_id = filter_output(
        "The item is currently assigned to u3."
    )
    assert_equal(internal_id.allowed, True, "internal ID reply allowed")
    assert_equal(
        internal_id.reply,
        "The item is currently assigned to [internal account].",
        "internal ID redacted",
    )

    email = filter_output(
        "A reminder was sent to priya.nair@example.edu."
    )
    assert_equal(email.allowed, True, "email reply allowed")
    assert_equal(
        email.reply,
        "A reminder was sent to [email redacted].",
        "email redacted",
    )

    credential = filter_output(
        "The API key is sk-this-should-never-be-displayed."
    )
    assert_equal(credential.allowed, False, "credential reply blocked")
    assert_equal(
        credential.reason,
        "credential_pattern_detected",
        "credential block reason",
    )

    provider_error = filter_output(
        "Azure OpenAI ResponsibleAIPolicyViolation traceback "
        "content_filter_result"
    )
    assert_equal(provider_error.allowed, False, "provider error blocked")
    assert_equal(
        provider_error.reason,
        "technical_error_pattern_detected",
        "provider error block reason",
    )

    oversized = filter_output("x" * 4000)
    assert_equal(oversized.allowed, False, "oversized reply blocked")
    assert_equal(
        oversized.reason,
        "reply_too_long",
        "oversized block reason",
    )

    print("Output guard tests passed.")


if __name__ == "__main__":
    main()