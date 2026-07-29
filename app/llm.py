"""Shared LLM client for LabBot. One place to swap providers."""

from . import config

class ContentFilteredError(Exception):
    """Raised when Azure OpenAI blocks content before model generation."""

def client():
    """Return an LLM client. Swap this out if you use a different provider."""
    from openai import AzureOpenAI

    if not (config.AZURE_ENDPOINT and config.AZURE_API_KEY):
        raise RuntimeError(
            "Azure OpenAI credentials are missing. Copy .env.example to .env "
            "and fill in your values."
        )
    return AzureOpenAI(
        azure_endpoint=config.AZURE_ENDPOINT,
        api_key=config.AZURE_API_KEY,
        api_version=config.AZURE_API_VERSION,
    )


def complete(
    system: str,
    user: str,
    temperature: float = 0.3,
    max_tokens: int | None = None,
) -> str:
    """One-shot chat completion helper.

    Raises ContentFilteredError when Azure blocks a prompt due to content
    safety policy. The graph catches that error and returns a safe user-facing
    response rather than crashing the FastAPI request.
    """
    from openai import BadRequestError

    kwargs = {}

    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    try:
        response = client().chat.completions.create(
            model=config.AZURE_CHAT_DEPLOYMENT,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **kwargs,
        )

        return response.choices[0].message.content or ""

    except BadRequestError as error:
        # Azure returns a BadRequestError with a content_filter code when a
        # jailbreak or other safety policy is triggered.
        body = getattr(error, "body", {}) or {}
        error_data = body.get("error", {}) if isinstance(body, dict) else {}

        if (
            error_data.get("code") == "content_filter"
            or "content management policy" in str(error).lower()
            or "jailbreak" in str(error).lower()
        ):
            raise ContentFilteredError(
                "The request was blocked by the provider safety filter."
            ) from error

        raise