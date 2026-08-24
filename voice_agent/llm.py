"""Groq streaming completion -- classifies ticket urgency/category and
generates the spoken response. Streams tokens (not a blocking
completion) so tts.py can start synthesizing before the LLM finishes,
and so an in-flight generation can be cancelled when a barge-in fires
mid-response.

Same provider choice as the llm-cost-router project (Groq, $0 free
tier), same model (`openai/gpt-oss-20b`) for the same reason: verified
live and working there already, no reason to re-gamble on a different
free-tier model here. See DECISIONS.md.
"""

from typing import AsyncIterator

from groq import AsyncGroq

from voice_agent.config import settings

SYSTEM_PROMPT = (
    "You are a support ticket triage agent on a phone call. The caller has "
    "just described an issue by voice (transcribed below, may contain minor "
    "transcription errors). In one or two short spoken sentences: (1) briefly "
    "acknowledge the issue in your own words, (2) state its urgency "
    "(low/medium/high/critical) and category (billing/technical/account/other), "
    "and (3) tell them what happens next. Keep it conversational and brief -- "
    "this is spoken aloud, not read as text. Do not use markdown, bullet "
    "points, or any formatting -- plain spoken sentences only."
)

_client: AsyncGroq | None = None


def _get_client() -> AsyncGroq:
    """Explicitly passes `settings.groq_api_key` rather than letting
    AsyncGroq() fall back to its own bare `os.environ["GROQ_API_KEY"]`
    lookup. That fallback is a real trap: `voice_agent.config.settings`
    (pydantic-settings) parses `.env` into its own `Settings` instance,
    but nothing else ever loads `.env` into the actual process
    environment when running the real server -- only `tests/conftest.py`
    does that, and only for pytest. Found the hard way: a background
    `asyncio.create_task()` (see pipeline.py's CallSession) that hit
    AsyncGroq()'s "api_key must be set" error died silently with zero
    visible symptom (task exception never retrieved, task never awaited
    or garbage-collected during the test) -- looked exactly like a hang
    from the outside. See DECISIONS.md."""
    global _client
    if _client is None:
        _client = AsyncGroq(api_key=settings.groq_api_key or None)
    return _client


async def stream_response(
    transcript: str, model: str = "openai/gpt-oss-20b", client: AsyncGroq | None = None
) -> AsyncIterator[str]:
    """Yield response text tokens as they're generated.

    `client` param exists so tests can inject a fake SDK client instead
    of hitting the network / requiring a real API key -- same pattern as
    llm-cost-router's GroqProvider.
    """
    active_client = client or _get_client()
    stream = await active_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ],
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
