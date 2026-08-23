"""Streaming speech synthesis. Piper (local, ONNX-backed) -- chosen over
a hosted streaming TTS API specifically so the live demo isn't dependent
on someone else's free-tier rate limits during a live interrupt-heavy
session. See DECISIONS.md.

Yields audio in chunks (not one blocking buffer) so playback can start
before the full response is synthesized, and so an in-flight synthesis
can actually be cancelled mid-stream when a barge-in fires -- that
cancellation point is what `cutoff_latency` (see eval/) measures.

Piper's own `synthesize()` is a synchronous generator (blocking, CPU-bound
ONNX inference per chunk) -- run on a background thread, chunks handed
back across an asyncio.Queue, so a caller `break`-ing out of the async
iteration (e.g. on barge-in) actually stops consuming without blocking
the event loop on the next chunk's inference.
"""

import asyncio
import queue
import threading
from pathlib import Path
from typing import AsyncIterator

from piper import PiperVoice, SynthesisConfig
from piper.download_voices import download_voice

from voice_agent.config import settings

_SENTINEL = object()
_voice: PiperVoice | None = None
_voice_lock = threading.Lock()


def _voice_code_and_dir() -> tuple[str, Path]:
    """`en_US-lessac-medium.onnx` -> ("en_US-lessac-medium", <parent dir>)."""
    model_path = Path(settings.piper_model_path)
    return model_path.stem, model_path.parent


def _load_voice() -> PiperVoice:
    """Lazily download (if needed) and load the configured Piper voice.
    Thread-safe, cached module-wide -- model load is real work (ONNX
    session init), not something to repeat per request."""
    global _voice
    if _voice is not None:
        return _voice
    with _voice_lock:
        if _voice is not None:  # re-check inside the lock
            return _voice
        voice_code, download_dir = _voice_code_and_dir()
        download_dir.mkdir(parents=True, exist_ok=True)
        download_voice(voice_code, download_dir)  # no-op if already present
        model_path = download_dir / f"{voice_code}.onnx"
        _voice = PiperVoice.load(model_path)
        return _voice


def _synthesize_blocking(text: str, chunk_queue: "queue.Queue[bytes | BaseException | object]") -> None:
    """Runs on a background thread: iterates Piper's synchronous
    generator and pushes each chunk's raw PCM bytes onto the queue.

    Exceptions are put on the queue too, not swallowed -- an earlier
    version of this function only had a bare `finally: put(_SENTINEL)`,
    which meant a real failure (e.g. the network error hit downloading
    the voice config during testing) silently produced an empty audio
    stream instead of surfacing to the caller. Caught by a test
    asserting non-empty output, not by inspection.
    """
    try:
        voice = _load_voice()
        for audio_chunk in voice.synthesize(text, SynthesisConfig()):
            chunk_queue.put(audio_chunk.audio_int16_bytes)
    except BaseException as e:  # noqa: BLE001 -- deliberately broad, re-raised on the consumer side
        chunk_queue.put(e)
    finally:
        chunk_queue.put(_SENTINEL)


def output_sample_rate() -> int:
    """Piper's native output sample rate for the configured voice (22050Hz
    for en_US-lessac-medium, verified directly via `voice.config.sample_rate`
    -- not hardcoded, so a future voice swap doesn't silently desync the
    browser client's playback config from the real audio being sent).
    Loads the voice if not already loaded (same cached singleton as
    synthesis itself)."""
    return _load_voice().config.sample_rate


async def synthesize_stream(text: str) -> AsyncIterator[bytes]:
    """Yield 16-bit PCM audio chunks as they're synthesized. Breaking out
    of iteration early (e.g. on barge-in) stops consumption -- the
    background thread finishes its current chunk of ONNX inference (not
    instantly killable) but no further chunks are awaited or played.

    Re-raises any exception the background thread hit, on the consumer's
    side, instead of silently ending the stream early.
    """
    chunk_queue: "queue.Queue[bytes | BaseException | object]" = queue.Queue()
    thread = threading.Thread(target=_synthesize_blocking, args=(text, chunk_queue), daemon=True)
    thread.start()
    loop = asyncio.get_running_loop()
    try:
        while True:
            item = await loop.run_in_executor(None, chunk_queue.get)
            if item is _SENTINEL:
                break
            if isinstance(item, BaseException):
                raise item
            yield item  # type: ignore[misc]
    finally:
        thread.join(timeout=0.1)  # best-effort; daemon thread won't block process exit
