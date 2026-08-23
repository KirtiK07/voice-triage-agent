"""Streaming speech synthesis. Piper (local, ONNX-backed) -- chosen over
a hosted streaming TTS API specifically so the live demo isn't dependent
on someone else's free-tier rate limits during a live interrupt-heavy
session. See DECISIONS.md.

Must yield audio in chunks (not return one blocking buffer) so playback
can start before the full response is synthesized, and so an in-flight
synthesis can actually be cancelled mid-stream when a barge-in fires --
that cancellation point is what `cutoff_latency` (see eval/) measures.

Business logic not yet implemented -- scaffold stage only establishes
the interface.
"""

from typing import AsyncIterator


async def synthesize_stream(text: str) -> AsyncIterator[bytes]:
    """Yield audio chunks as they're synthesized. Not yet implemented
    (build stage)."""
    raise NotImplementedError("build stage: wire up Piper streaming synthesis")
    yield b""  # pragma: no cover -- makes this an async generator for the type checker
