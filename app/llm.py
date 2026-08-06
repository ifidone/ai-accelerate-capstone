"""Shared LLM client for LabBot. One place to swap providers.

Chat completions go through Claude (Anthropic). Embeddings stay on Azure
OpenAI via client() below — see app/rag.py — since Anthropic has no
embeddings endpoint.
"""

from . import config

class ContentFilteredError(Exception):
    """Raised when the provider blocks content before model generation."""


_TIER_MODELS = {
    "haiku": lambda: config.ANTHROPIC_HAIKU_MODEL,
    "sonnet": lambda: config.ANTHROPIC_SONNET_MODEL,
}


def client():
    """Azure OpenAI client. Still used for embeddings (app/rag.py) — chat
    completions go through anthropic_client() / complete() instead."""
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


def anthropic_client():
    """Return the Claude chat client used by complete()."""
    from anthropic import Anthropic

    if not (config.ANTHROPIC_API_KEY and config.ANTHROPIC_BASE_URL):
        raise RuntimeError(
            "Anthropic credentials are missing. Set ANTHROPIC_API_KEY and "
            "ANTHROPIC_BASE_URL in .env."
        )
    return Anthropic(
        api_key=config.ANTHROPIC_API_KEY,
        base_url=config.ANTHROPIC_BASE_URL,
    )


def complete(
    system: str,
    user: str,
    temperature: float = 0.3,
    max_tokens: int | None = None,
    model: str = "sonnet",
) -> str:
    """One-shot chat completion helper.

    `model` selects a tier, not a literal model name:
      - "haiku"  — classification and structured extraction: short, low-
                    creativity, latency/cost-sensitive.
      - "sonnet" — user-facing generation: policy answers and action-result
                    narration, where instruction-following and resistance to
                    hallucinated IDs/policies/outcomes matter more than speed.

    Raises ContentFilteredError when the provider blocks a prompt due to
    content safety policy. The graph catches that error and returns a safe
    user-facing response rather than crashing the FastAPI request.
    """
    from anthropic import BadRequestError

    if model not in _TIER_MODELS:
        raise ValueError(f"Unknown model tier '{model}'. Use 'haiku' or 'sonnet'.")

    model_name = _TIER_MODELS[model]()

    try:
        response = anthropic_client().messages.create(
            model=model_name,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens or 1024,
        )

        if getattr(response, "stop_reason", None) == "refusal":
            raise ContentFilteredError(
                "The request was blocked by the provider safety filter."
            )

        return "".join(
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        )

    except BadRequestError as error:
        error_text = str(error).lower()

        if "content" in error_text and (
            "polic" in error_text or "filter" in error_text
        ) or "jailbreak" in error_text:
            raise ContentFilteredError(
                "The request was blocked by the provider safety filter."
            ) from error

        raise
