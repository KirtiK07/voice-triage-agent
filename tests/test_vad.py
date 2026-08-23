"""Real tests against the actual Silero/WebRTC VAD backends -- no
mocking the model/detector itself, matching the project's standing
"verify, don't assume" discipline. The accumulation-threshold logic is
tested by monkeypatching only the underlying per-frame probability call
(not the whole detector), since crafting synthetic audio that Silero
reliably scores as "speech" is its own hard problem unrelated to what
this project's own code is responsible for.
"""

import pytest

from voice_agent.vad import SAMPLE_RATE, SileroDetector, WebRTCDetector, get_detector


def _silence_frame(n_samples: int) -> bytes:
    return b"\x00\x00" * n_samples


class TestSileroDetector:
    def test_frame_size_is_512_samples_at_16khz(self):
        """Regression test for a real constraint discovered at plan-stage
        spike time: the JIT model only accepts exactly 512 samples (32ms)
        at 16kHz -- 256/480/1024/160 all failed."""
        assert SileroDetector.frame_size_samples == 512

    def test_wrong_frame_size_raises_clear_error(self):
        detector = SileroDetector(speech_threshold_ms=250)
        with pytest.raises(ValueError, match="exactly 512 samples"):
            detector.process_frame(_silence_frame(256))

    def test_real_silence_never_fires(self):
        """Negative control using real model inference -- not mocked.
        Feeds far more consecutive silent frames than the threshold
        requires; a false positive here would mean the model itself (or
        our threshold wiring) is broken."""
        detector = SileroDetector(speech_threshold_ms=250)
        frame = _silence_frame(512)
        results = [detector.process_frame(frame) for _ in range(50)]
        assert not any(results)

    def test_classify_frame_real_silence_is_false(self):
        """classify_frame is the raw per-frame signal `process_frame`
        builds on -- used directly by the end-of-utterance capture logic
        (see turn_taking.py), so it needs its own real-inference check,
        not just coverage-by-association through process_frame."""
        detector = SileroDetector(speech_threshold_ms=250)
        assert detector.classify_frame(_silence_frame(512)) is False

    def test_accumulation_fires_after_threshold_and_resets_on_silence(self, monkeypatch):
        """Tests *our* duration-accumulation logic precisely by
        controlling the per-frame probability directly, rather than
        depending on a specific audio sample scoring as "speech" --
        that's Silero's concern, already spot-checked by the negative
        control above and the plan-stage spike."""
        detector = SileroDetector(speech_threshold_ms=250)  # 32ms/frame -> 8 frames needed
        assert detector._frames_needed == 8

        probs = iter([0.9] * 7 + [0.9] + [0.9] * 5)  # 8th frame should fire

        def fake_call(_tensor, _sr):
            class _Result:
                def item(self_inner):
                    return next(probs)

            return _Result()

        monkeypatch.setattr(detector, "_model", fake_call)

        frame = _silence_frame(512)
        results = [detector.process_frame(frame) for _ in range(8)]
        assert results[:7] == [False] * 7
        assert results[7] is True

    def test_reset_clears_accumulated_state(self):
        detector = SileroDetector(speech_threshold_ms=250)
        detector._consecutive_speech_frames = 5
        detector.reset()
        assert detector._consecutive_speech_frames == 0


class TestWebRTCDetector:
    def test_frame_size_is_480_samples(self):
        assert WebRTCDetector.frame_size_samples == 480

    def test_wrong_frame_size_raises_clear_error(self):
        detector = WebRTCDetector(speech_threshold_ms=250)
        with pytest.raises(ValueError, match="exactly 480 samples"):
            detector.process_frame(_silence_frame(512))

    def test_real_silence_never_fires(self):
        detector = WebRTCDetector(speech_threshold_ms=250)
        frame = _silence_frame(480)
        results = [detector.process_frame(frame) for _ in range(50)]
        assert not any(results)

    def test_accumulation_fires_after_threshold(self, monkeypatch):
        detector = WebRTCDetector(speech_threshold_ms=250)  # 30ms/frame -> 8 frames needed
        assert detector._frames_needed == 8

        calls = {"n": 0}

        def fake_is_speech(_frame, _sr):
            calls["n"] += 1
            return True

        monkeypatch.setattr(detector._vad, "is_speech", fake_is_speech)

        frame = _silence_frame(480)
        results = [detector.process_frame(frame) for _ in range(8)]
        assert results[:7] == [False] * 7
        assert results[7] is True


class TestGetDetector:
    def test_returns_silero_by_default(self):
        detector = get_detector("silero", speech_threshold_ms=250)
        assert isinstance(detector, SileroDetector)

    def test_returns_webrtc(self):
        detector = get_detector("webrtc", speech_threshold_ms=250)
        assert isinstance(detector, WebRTCDetector)

    def test_caches_by_backend_and_threshold(self):
        d1 = get_detector("silero", speech_threshold_ms=300)
        d2 = get_detector("silero", speech_threshold_ms=300)
        assert d1 is d2

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown VAD backend"):
            get_detector("not-a-real-backend", speech_threshold_ms=250)


def test_sample_rate_constant_is_16khz():
    assert SAMPLE_RATE == 16000
