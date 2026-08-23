"""Groq streaming completion -- classifies ticket urgency/category and
generates the spoken response. Must stream tokens (not block on a full
completion) so `tts.py` can start synthesizing before the LLM finishes,
and so an in-flight generation can be cancelled when a barge-in fires
mid-response.

Business logic not yet implemented -- scaffold stage only establishes
the interface.
"""

from typing import AsyncIterator


async def stream_response(transcript: str) -> AsyncIterator[str]:
    """Yield response text tokens as they're generated. Not yet
    implemented (build stage)."""
    raise NotImplementedError("build stage: wire up Groq streaming client")
    yield ""  # pragma: no cover -- makes this an async generator for the type checker
