"""Transcription: the *what*, not the *when*. Only invoked after
vad.py fires a barge-in (or at the start of a turn) -- never on the
hot interrupt-detection path, since faster-whisper's partial-transcript
latency (1-2s context) is too slow to serve as the interrupt trigger
itself. See vad.py and DECISIONS.md.

Backend: faster-whisper (CTranslate2-backed Whisper), CPU. Model size
via `settings.whisper_model_size`. Import/runtime path verified working
on the Vercel deploy target during the plan-stage spike.

Business logic not yet implemented -- scaffold stage only establishes
the interface.
"""


def transcribe(audio_bytes: bytes) -> str:
    """Transcribe a completed utterance. Not yet implemented (build stage)."""
    raise NotImplementedError("build stage: wire up faster-whisper")
