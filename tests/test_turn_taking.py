"""Tests for UtteranceCapture's own buffering/end-detection logic --
whether the underlying VAD backend classifies audio correctly is
test_vad.py's concern, so a fake detector with a directly-settable
per-frame classification is used here, matching the isolation pattern
established in test_pipeline.py.
"""

import pytest

import voice_agent.turn_taking as tt_module
from voice_agent.turn_taking import UtteranceCapture


class _FakeDetector:
    frame_size_samples = 4  # tiny, arbitrary -- only byte-length matters here

    def __init__(self):
        self.speaking = False
        self.reset_calls = 0

    def classify_frame(self, _frame: bytes) -> bool:
        return self.speaking

    def process_frame(self, _frame: bytes) -> bool:  # pragma: no cover -- unused here
        raise NotImplementedError

    def reset(self) -> None:
        self.reset_calls += 1


@pytest.fixture
def fake_detector(monkeypatch):
    detector = _FakeDetector()
    monkeypatch.setattr(tt_module, "get_detector", lambda backend: detector)
    return detector


def _frame(n=4) -> bytes:
    return b"\x00\x00" * n


class TestUtteranceCapture:
    def test_silence_only_never_fires(self, fake_detector):
        # frame_size_samples=4 -> 30ms-equivalent isn't real here; use a
        # small silence_ms_to_end so the test doesn't need hundreds of frames.
        capture = UtteranceCapture(vad_backend="silero", silence_ms_to_end=1)
        fake_detector.speaking = False
        results = [capture.feed(_frame()) for _ in range(20)]
        assert not any(results)

    def test_fires_after_speech_then_sustained_silence(self, fake_detector):
        capture = UtteranceCapture(vad_backend="silero", silence_ms_to_end=100)
        # frame_size_samples=4 @ 16kHz -> 0.25ms/frame -> frames_needed is
        # clamped to at least 1, so compute what this fixture actually needs.
        frames_needed = capture._frames_needed_for_end

        fake_detector.speaking = True
        assert capture.feed(_frame()) is False  # speech frame never ends the utterance

        fake_detector.speaking = False
        results = [capture.feed(_frame()) for _ in range(frames_needed - 1)]
        assert not any(results), "should not fire before the silence threshold is reached"

        assert capture.feed(_frame()) is True  # the Nth silence frame fires

    def test_silence_resets_early_if_speech_resumes(self, fake_detector):
        capture = UtteranceCapture(vad_backend="silero", silence_ms_to_end=100)
        frames_needed = capture._frames_needed_for_end
        if frames_needed < 2:
            pytest.skip("needs at least 2 silence frames to test an interrupted silence run")

        fake_detector.speaking = True
        capture.feed(_frame())
        fake_detector.speaking = False
        capture.feed(_frame())  # 1 silence frame in
        fake_detector.speaking = True
        capture.feed(_frame())  # speech resumes -- silence count should reset
        fake_detector.speaking = False
        results = [capture.feed(_frame()) for _ in range(frames_needed - 1)]
        assert not any(results), "silence count must have reset, not carried over"

    def test_buffers_all_frames_including_trailing_silence(self, fake_detector):
        capture = UtteranceCapture(vad_backend="silero", silence_ms_to_end=1)
        fake_detector.speaking = True
        capture.feed(_frame(4))
        fake_detector.speaking = False
        capture.feed(_frame(4))
        assert capture.audio == _frame(4) * 2

    def test_reset_clears_buffer_and_state_and_resets_detector(self, fake_detector):
        capture = UtteranceCapture(vad_backend="silero", silence_ms_to_end=1)
        fake_detector.speaking = True
        capture.feed(_frame())
        capture.reset()
        assert capture.audio == b""
        assert capture._speech_started is False
        assert fake_detector.reset_calls == 1

    def test_reuses_cached_detector_not_a_new_one(self, monkeypatch):
        """Regression check for the actual design intent: UtteranceCapture
        must call the real get_detector() (which caches by backend), not
        construct its own detector instance -- verified by asserting the
        factory function itself gets called with the right backend name."""
        calls = []

        class _Stub:
            frame_size_samples = 4

            def classify_frame(self, f):
                return False

            def reset(self):
                pass

        def fake_get_detector(backend):
            calls.append(backend)
            return _Stub()

        monkeypatch.setattr(tt_module, "get_detector", fake_get_detector)
        UtteranceCapture(vad_backend="webrtc")
        assert calls == ["webrtc"]
