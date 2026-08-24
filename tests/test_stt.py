"""Real round-trip test: Piper synthesizes real speech (22050Hz, its
native rate -- see DECISIONS.md), resampled to faster-whisper's required
16kHz, then actually transcribed. No mocking -- this is the strongest
available check that the STT module works end-to-end, not just that it
doesn't crash on silence.
"""

import pytest

from voice_agent.audio_utils import resample_int16
from voice_agent.stt import transcribe
from voice_agent.tts import synthesize_stream

PIPER_SAMPLE_RATE = 22050
WHISPER_SAMPLE_RATE = 16000


@pytest.mark.asyncio
async def test_transcribe_recovers_real_synthesized_speech():
    text = "The quick brown fox jumps over the lazy dog."
    chunks = [c async for c in synthesize_stream(text)]
    piper_audio = b"".join(chunks)

    whisper_audio = resample_int16(piper_audio, PIPER_SAMPLE_RATE, WHISPER_SAMPLE_RATE)
    result = transcribe(whisper_audio)

    # Not an exact-match assertion -- Whisper's punctuation/casing can
    # differ from the input. Check the actual content words came back.
    result_lower = result.lower()
    for word in ("quick", "brown", "fox", "jumps", "lazy", "dog"):
        assert word in result_lower, f"expected {word!r} in transcription, got: {result!r}"


def test_transcribe_silence_returns_empty_or_near_empty():
    silence = b"\x00\x00" * WHISPER_SAMPLE_RATE  # 1 second of silence
    result = transcribe(silence)
    assert len(result) < 10  # no hallucinated sentence from pure silence
