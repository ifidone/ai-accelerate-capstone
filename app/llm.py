"""Shared LLM client for LabBot. One place to swap providers."""

from . import config


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


def complete(system: str, user: str, temperature: float = 0.3, max_tokens: int | None = None) -> str:
    """One-shot chat completion helper: system + single user turn -> text."""
    kwargs = {}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    resp = client().chat.completions.create(
        model=config.AZURE_CHAT_DEPLOYMENT,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        **kwargs,
    )
    return resp.choices[0].message.content or ""