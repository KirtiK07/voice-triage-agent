"""Transcription: the *what*, not the *when*. Only invoked after
vad.py fires a barge-in (or at the start of a turn) -- never on the
hot interrupt-detection path, since faster-whisper's partial-transcript
latency (1-2s context) is too slow to serve as the interrupt trigger
itself. See vad.py and DECISIONS.md.

Backend: faster-whisper (CTranslate2-backed Whisper), CPU, int8
quantized for speed. Model size via `settings.whisper_model_size`.
"""

import threading

import numpy as np

_model = None
_model_lock = threading.Lock()


def _load_model():
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:  # re-check inside the lock
            return _model
        from faster_whisper import WhisperModel

        from voice_agent.config import settings

        _model = WhisperModel(settings.whisper_model_size, device="cpu", compute_type="int8")
        return _model


def warm() -> None:
    """Forces the model to load now, synchronously, on the calling
    thread -- see server.py's startup hook and DECISIONS.md "Model
    warm-up and the torch/onnxruntime deadlock" for why this matters:
    loading faster-whisper (which also touches ctranslate2) for the
    first time needs to happen deliberately at startup, not implicitly
    on a background thread during the first real request."""
    _load_model()


def transcribe(audio_bytes: bytes) -> str:
    """Transcribe a completed utterance. `audio_bytes` is 16-bit PCM
    mono at 16kHz (matching vad.py's SAMPLE_RATE -- the same audio
    stream, just handed here as a completed buffer instead of per-frame)."""
    model = _load_model()
    pcm = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    segments, _info = model.transcribe(pcm, language="en")
    return " ".join(segment.text.strip() for segment in segments).strip()
