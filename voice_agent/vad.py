"""Barge-in detection: the *when*, not the *what*. Runs continuously
during agent TTS playback; fires as soon as it detects sustained caller
speech (not a breath/cough -- accumulates consecutive speech-classified
frames past `settings.vad_speech_threshold_ms` before firing).

Deliberately NOT faster-whisper: partial transcripts have too much
latency (1-2s context) to be the interrupt signal itself -- see
CLAUDE.md "Open questions" and DECISIONS.md. `stt.py` only runs after
this module fires, to transcribe what the caller actually said.

Backend is selected via `settings.vad_backend` ("silero" primary,
"webrtc" as a lighter fallback -- see .env.example).
"""

import threading
from typing import Protocol

SAMPLE_RATE = 16000


class BargeInDetector(Protocol):
    """Interface every VAD backend implements. `frame_size_samples`
    documents the exact chunk size the backend's methods expect --
    backends have genuinely different native frame-size constraints (see
    below), so callers must chunk incoming PCM accordingly rather than
    assume one universal size."""

    frame_size_samples: int

    def classify_frame(self, audio_frame: bytes) -> bool:
        """Feed one 16-bit PCM mono frame at 16kHz, exactly
        `frame_size_samples` samples long. Returns the raw per-frame
        speech/silence classification, with no duration-accumulation --
        used directly by callers that need a *symmetric* signal (e.g.
        detecting the caller has gone silent again, not just that they
        started talking). `process_frame` builds sustained-speech-start
        detection on top of this; this method exists on the public
        interface too since it's a genuinely different, useful signal on
        its own."""
        ...

    def process_frame(self, audio_frame: bytes) -> bool:
        """Feed one frame (same format as `classify_frame`). Returns
        True the moment sustained speech (past the configured threshold)
        is detected -- i.e. `classify_frame` returning True for enough
        consecutive frames."""
        ...

    def reset(self) -> None:
        """Clear accumulated speech-frame state, e.g. after a barge-in
        fires or a new turn starts."""
        ...


class SileroDetector:
    """Primary backend. Model constraint verified directly (not assumed
    from docs): the JIT-scripted `silero_vad` model only accepts exactly
    512 samples (32ms) per call at 16kHz -- 256, 480, 1024, and 160 all
    raised at plan-stage spike time. `VADIterator` (Silero's own
    streaming utility) exists but fires on the very first frame crossing
    `threshold`, with no duration-accumulation knob -- too eager for this
    project's "sustained N ms" design, so the accumulation is implemented
    directly against the raw per-frame speech probability instead.
    """

    frame_size_samples = 512  # 32ms at 16kHz -- see docstring above

    def __init__(self, speech_threshold_ms: int, prob_threshold: float = 0.5) -> None:
        import torch

        self._torch = torch
        model, _utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True, onnx=False
        )
        self._model = model
        self._prob_threshold = prob_threshold
        frame_ms = 1000 * self.frame_size_samples / SAMPLE_RATE
        self._frames_needed = max(1, round(speech_threshold_ms / frame_ms))
        self._consecutive_speech_frames = 0

    def classify_frame(self, audio_frame: bytes) -> bool:
        if len(audio_frame) != self.frame_size_samples * 2:  # 16-bit = 2 bytes/sample
            raise ValueError(
                f"SileroDetector requires exactly {self.frame_size_samples} samples "
                f"({self.frame_size_samples * 2} bytes) per frame, got {len(audio_frame)} bytes"
            )
        import numpy as np

        pcm = np.frombuffer(audio_frame, dtype=np.int16).astype(np.float32) / 32768.0
        tensor = self._torch.from_numpy(pcm)
        speech_prob = self._model(tensor, SAMPLE_RATE).item()
        return speech_prob >= self._prob_threshold

    def process_frame(self, audio_frame: bytes) -> bool:
        if self.classify_frame(audio_frame):
            self._consecutive_speech_frames += 1
        else:
            self._consecutive_speech_frames = 0

        return self._consecutive_speech_frames >= self._frames_needed

    def reset(self) -> None:
        self._consecutive_speech_frames = 0
        # Silero's model carries internal recurrent state across calls;
        # reset it too so a fresh turn doesn't inherit stale context.
        if hasattr(self._model, "reset_states"):
            self._model.reset_states()


class WebRTCDetector:
    """Lighter fallback -- no torch dependency, much cheaper per-frame.
    webrtcvad requires 10/20/30ms frames at 8/16/32/48kHz; 30ms @ 16kHz
    (480 samples) is used here, matching Silero's ~32ms cadence closely
    enough that `speech_threshold_ms` means roughly the same thing across
    both backends without the caller needing to know which is active.
    """

    frame_size_samples = 480  # 30ms at 16kHz

    def __init__(self, speech_threshold_ms: int, aggressiveness: int = 2) -> None:
        import webrtcvad

        self._vad = webrtcvad.Vad(aggressiveness)
        frame_ms = 1000 * self.frame_size_samples / SAMPLE_RATE
        self._frames_needed = max(1, round(speech_threshold_ms / frame_ms))
        self._consecutive_speech_frames = 0

    def classify_frame(self, audio_frame: bytes) -> bool:
        if len(audio_frame) != self.frame_size_samples * 2:
            raise ValueError(
                f"WebRTCDetector requires exactly {self.frame_size_samples} samples "
                f"({self.frame_size_samples * 2} bytes) per frame, got {len(audio_frame)} bytes"
            )
        return self._vad.is_speech(audio_frame, SAMPLE_RATE)

    def process_frame(self, audio_frame: bytes) -> bool:
        if self.classify_frame(audio_frame):
            self._consecutive_speech_frames += 1
        else:
            self._consecutive_speech_frames = 0

        return self._consecutive_speech_frames >= self._frames_needed

    def reset(self) -> None:
        self._consecutive_speech_frames = 0


_lock = threading.Lock()
_cache: dict[str, BargeInDetector] = {}


def get_detector(backend: str, speech_threshold_ms: int | None = None) -> BargeInDetector:
    """Factory -- returns the configured VAD backend, cached per backend
    name (model load is real work, not something to repeat per call)."""
    from voice_agent.config import settings

    threshold = speech_threshold_ms if speech_threshold_ms is not None else settings.vad_speech_threshold_ms
    cache_key = f"{backend}:{threshold}"
    if cache_key in _cache:
        return _cache[cache_key]
    with _lock:
        if cache_key in _cache:  # re-check inside the lock
            return _cache[cache_key]
        if backend == "silero":
            detector: BargeInDetector = SileroDetector(threshold)
        elif backend == "webrtc":
            detector = WebRTCDetector(threshold)
        else:
            raise ValueError(f"Unknown VAD backend: {backend!r} (expected 'silero' or 'webrtc')")
        _cache[cache_key] = detector
        return detector
