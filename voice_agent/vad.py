"""Barge-in detection: the *when*, not the *what*. Runs continuously
during agent TTS playback; fires as soon as it detects sustained caller
speech (not a breath/cough -- see `settings.vad_speech_threshold_ms`).

Deliberately NOT faster-whisper: partial transcripts have too much
latency (1-2s context) to be the interrupt signal itself -- see
CLAUDE.md "Open questions" and DECISIONS.md. `stt.py` only runs after
this module fires, to transcribe what the caller actually said.

Backend is selected via `settings.vad_backend` ("silero" primary,
"webrtc" as a lighter fallback -- see .env.example). Silero VAD's
import/inference path was verified working end-to-end on the actual
Vercel deploy target during the plan-stage spike (see DECISIONS.md).

Business logic not yet implemented -- scaffold stage only establishes
the interface.
"""

from typing import Protocol


class BargeInDetector(Protocol):
    """Interface every VAD backend implements."""

    def process_frame(self, audio_frame: bytes) -> bool:
        """Feed one audio frame; return True the moment sustained speech
        (past the configured threshold) is detected during playback."""
        ...

    def reset(self) -> None:
        """Clear any accumulated speech-frame state, e.g. after a
        barge-in fires or a new turn starts."""
        ...


def get_detector(backend: str) -> BargeInDetector:
    """Factory -- returns the configured VAD backend. Not yet
    implemented (build stage)."""
    raise NotImplementedError("build stage: wire up Silero/WebRTC VAD backends")
