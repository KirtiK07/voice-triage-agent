"""Caller end-of-utterance detection: a symmetric problem to barge-in
detection (vad.py). vad.BargeInDetector answers "has the caller STARTED
talking" (used to interrupt agent playback); this module answers "has
the caller FINISHED talking" (sustained silence after speech), so the
WebSocket handler knows when to stop buffering microphone audio and hand
it to STT.

Deliberately reuses the same cached VAD instance rather than loading a
second model: `get_detector()` caches by backend+threshold (see vad.py),
so calling it again here returns the identical instance CallSession
already uses for barge-in -- no extra memory/compute, and no special
plumbing needed to share it, since the two uses are always sequential
(listening for a new utterance vs. listening for a barge-in during
playback), never concurrent for the same call.
"""

from voice_agent.vad import SAMPLE_RATE, BargeInDetector, get_detector


class UtteranceCapture:
    """Buffers microphone frames for one caller utterance and reports
    when it's complete (sustained trailing silence after speech has
    started)."""

    def __init__(
        self, vad_backend: str | None = None, silence_ms_to_end: int = 600
    ) -> None:
        from voice_agent.config import settings

        backend = vad_backend or settings.vad_backend
        self._detector: BargeInDetector = get_detector(backend)
        self.frame_size_samples = self._detector.frame_size_samples
        frame_ms = 1000 * self.frame_size_samples / SAMPLE_RATE
        self._frames_needed_for_end = max(1, round(silence_ms_to_end / frame_ms))
        self._buffer = bytearray()
        self._speech_started = False
        self._silence_frames = 0

    def feed(self, frame: bytes) -> bool:
        """Feed one frame (`frame_size_samples` samples of 16-bit PCM at
        16kHz -- matches the active VAD backend's requirement). Always
        buffers the frame, speech or silence, so trailing silence isn't
        clipped off the recording before STT gets a chance to include
        it. Returns True once end-of-utterance is detected -- only
        possible after speech has actually started, so silence alone
        (e.g. a caller who never speaks) never fires."""
        self._buffer.extend(frame)
        if self._detector.classify_frame(frame):
            self._speech_started = True
            self._silence_frames = 0
        elif self._speech_started:
            self._silence_frames += 1
        return self._speech_started and self._silence_frames >= self._frames_needed_for_end

    @property
    def audio(self) -> bytes:
        return bytes(self._buffer)

    def reset(self) -> None:
        """Start capturing a fresh utterance. Also resets the shared VAD
        detector's own accumulation state, since it may have been left
        mid-count by whichever phase (barge-in listening or a prior
        utterance capture) used it last."""
        self._buffer = bytearray()
        self._speech_started = False
        self._silence_frames = 0
        self._detector.reset()
