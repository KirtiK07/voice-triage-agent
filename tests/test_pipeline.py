"""Tests for CallSession's cancel/restart state machine and timing --
not for whether the underlying VAD/LLM/TTS models are individually
correct (that's covered in test_vad.py/test_llm.py/test_tts.py). VAD,
LLM, and TTS are faked here with controllable, deterministic behavior so
these tests exercise the orchestration logic itself: does a barge-in
actually cancel playback, do the three timestamps land in the right
place, does turn_history stay clean.
"""

import asyncio

import pytest

import voice_agent.pipeline as pipeline_module
from voice_agent.pipeline import CallSession, TurnTimings


class _FakeDetector:
    """Fires on demand via `.armed`, not based on real audio content."""

    def __init__(self):
        self.armed = False
        self.reset_calls = 0

    def process_frame(self, _frame: bytes) -> bool:
        return self.armed

    def reset(self) -> None:
        self.reset_calls += 1
        self.armed = False


async def _fake_llm_stream(_transcript, delay=0.0):
    for tok in ("Got", " it", "."):
        if delay:
            await asyncio.sleep(delay)
        yield tok


async def _fake_tts_stream(_text, chunk_delay=0.0):
    for _ in range(3):
        if chunk_delay:
            await asyncio.sleep(chunk_delay)
        yield b"\x00\x00" * 100


@pytest.fixture
def fake_detector(monkeypatch):
    detector = _FakeDetector()
    monkeypatch.setattr(pipeline_module, "get_detector", lambda backend: detector)
    return detector


@pytest.fixture
def sent_audio():
    return []


@pytest.fixture
def session(fake_detector, sent_audio, monkeypatch):
    monkeypatch.setattr(pipeline_module.llm, "stream_response", lambda t: _fake_llm_stream(t))
    monkeypatch.setattr(pipeline_module.tts, "synthesize_stream", lambda t: _fake_tts_stream(t))

    async def send_audio(chunk: bytes) -> None:
        sent_audio.append(chunk)

    return CallSession(send_audio=send_audio, vad_backend="silero")


@pytest.fixture
def slow_session(fake_detector, sent_audio, monkeypatch):
    """Same as `session`, but with a deliberately slow TTS stream so a
    turn stays in flight long enough for a test to interrupt it
    mid-stream -- the plain `session` fixture's instant fakes can race
    to completion within a single `asyncio.sleep(0)`, which produced two
    flaky-by-design failures here before this fixture existed (both
    tests assumed the task was still running purely because they'd
    yielded control once, which isn't a real guarantee)."""
    monkeypatch.setattr(pipeline_module.llm, "stream_response", lambda t: _fake_llm_stream(t))
    monkeypatch.setattr(
        pipeline_module.tts, "synthesize_stream", lambda t: _fake_tts_stream(t, chunk_delay=0.05)
    )

    async def send_audio(chunk: bytes) -> None:
        sent_audio.append(chunk)

    return CallSession(send_audio=send_audio, vad_backend="silero")


@pytest.mark.asyncio
async def test_turn_completes_normally_without_barge_in(session, sent_audio):
    await session.start_turn("hello")
    await session._playback_task
    assert len(sent_audio) > 0
    assert session.is_speaking is False
    # A turn that was never interrupted must not pollute the benchmark history.
    assert session.turn_history == []


@pytest.mark.asyncio
async def test_barge_in_cancels_playback_and_records_timings(slow_session, fake_detector, sent_audio):
    session = slow_session
    await session.start_turn("hello")
    await asyncio.sleep(0.02)  # let the first chunk go out
    assert session.is_speaking is True

    fake_detector.armed = True
    timings = await session.feed_mic_frame(b"\x00\x00" * 512)

    assert timings is not None
    assert timings.t_vad_fire is not None
    assert timings.t_playback_stopped is not None
    assert timings.cutoff_latency_ms is not None
    assert timings.cutoff_latency_ms >= 0
    assert session.is_speaking is False
    assert fake_detector.reset_calls == 1


@pytest.mark.asyncio
async def test_full_barge_in_cycle_produces_one_clean_history_entry(slow_session, fake_detector):
    session = slow_session
    await session.start_turn("first utterance")
    await asyncio.sleep(0.02)  # let it actually start playing before interrupting

    fake_detector.armed = True
    timings = await session.feed_mic_frame(b"\x00\x00" * 512)
    assert timings is not None

    await session.start_turn("the interrupting utterance", continuing=timings)
    await session._playback_task

    assert len(session.turn_history) == 1
    recorded = session.turn_history[0]
    assert recorded is timings
    assert recorded.t_vad_fire is not None
    assert recorded.t_playback_stopped is not None
    assert recorded.t_new_audio_start is not None
    assert recorded.recovery_latency_ms is not None
    assert recorded.recovery_latency_ms >= 0
    # recovery includes the cutoff, so it should never be *less* than cutoff alone.
    assert recorded.recovery_latency_ms >= recorded.cutoff_latency_ms


@pytest.mark.asyncio
async def test_feed_mic_frame_returns_none_when_not_speaking(session, fake_detector):
    fake_detector.armed = True
    result = await session.feed_mic_frame(b"\x00\x00" * 512)
    assert result is None


@pytest.mark.asyncio
async def test_feed_mic_frame_returns_none_when_detector_has_not_fired(slow_session, fake_detector):
    session = slow_session
    await session.start_turn("hello")
    await asyncio.sleep(0.02)  # let it actually start playing
    fake_detector.armed = False
    result = await session.feed_mic_frame(b"\x00\x00" * 512)
    assert result is None
    assert session.is_speaking is True  # not cancelled
    await session._playback_task  # let it finish cleanly


def test_turn_timings_latency_properties_none_when_incomplete():
    t = TurnTimings()
    assert t.cutoff_latency_ms is None
    assert t.recovery_latency_ms is None
