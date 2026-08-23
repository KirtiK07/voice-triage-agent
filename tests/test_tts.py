"""Real tests against the actual Piper voice -- downloads the real model
(cached after first run, see voice_agent.tts._voice_code_and_dir) and
runs real ONNX inference. No mocking: the whole point of the plan-stage
spike was verifying this stack actually works, so the build-stage tests
hold it to the same standard.
"""

import pytest

import voice_agent.tts as tts_module
from voice_agent.tts import synthesize_stream


@pytest.mark.asyncio
async def test_synthesize_stream_yields_real_audio():
    chunks = []
    async for chunk in synthesize_stream("The quick brown fox jumps over the lazy dog."):
        chunks.append(chunk)

    assert len(chunks) >= 1
    total_bytes = sum(len(c) for c in chunks)
    assert total_bytes > 0
    # 16-bit PCM -- every chunk must be an even number of bytes (whole samples).
    for c in chunks:
        assert len(c) % 2 == 0


@pytest.mark.asyncio
async def test_synthesize_stream_multi_sentence_yields_multiple_chunks():
    """Piper yields one AudioChunk per sentence -- a multi-sentence
    input proves this is genuinely streaming output, not one blocking
    buffer wrapped in an async generator."""
    text = "First sentence here. Second sentence follows. And a third one too."
    chunks = [c async for c in synthesize_stream(text)]
    assert len(chunks) >= 2


@pytest.mark.asyncio
async def test_synthesize_stream_can_be_stopped_early():
    """Simulates a barge-in: the consumer stops iterating before the
    generator is exhausted. Must not raise or hang -- this is the
    behavior cutoff_latency (see voice_agent/pipeline.py) depends on."""
    count = 0
    async for _chunk in synthesize_stream(
        "First sentence here. Second sentence follows. And a third one too."
    ):
        count += 1
        if count == 1:
            break
    assert count == 1


@pytest.mark.asyncio
async def test_synthesize_stream_propagates_real_failures(monkeypatch):
    """Regression test for a real bug caught during build: the
    background-thread helper used to swallow exceptions (a bare
    `finally: put(_SENTINEL)`), so a genuine failure produced a
    silent empty stream instead of an error the caller could see."""

    def _boom():
        raise RuntimeError("simulated voice-load failure")

    monkeypatch.setattr(tts_module, "_load_voice", _boom)

    with pytest.raises(RuntimeError, match="simulated voice-load failure"):
        async for _chunk in synthesize_stream("this should never synthesize"):
            pass
