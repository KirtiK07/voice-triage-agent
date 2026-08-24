"""Shared audio helpers. Currently just resampling -- promoted out of
tests/test_stt.py (where it started as a test-only fixture) because
server.py's `simulate_speech` debug/demo path needs the exact same
22050Hz-Piper-output -> 16kHz-Whisper/Silero-input conversion for real,
not just in a test.
"""

import numpy as np


def resample_int16(pcm_bytes: bytes, from_rate: int, to_rate: int) -> bytes:
    """Linear-interpolation resample -- adequate for this project's
    actual use cases (test fixtures, and the simulate_speech demo path,
    neither of which is the real caller-microphone-input path a
    production system would need to resample). Real microphone input
    never needs this: the browser client's AudioContext is constructed
    at 16kHz directly (see public/client.js), so the browser/OS handles
    proper resampling from hardware rate. This function only exists for
    the two cases where *synthesized* (not captured) audio needs to
    cross the 22050Hz Piper / 16kHz Whisper+Silero boundary.
    """
    if from_rate == to_rate:
        return pcm_bytes
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    duration = len(samples) / from_rate
    n_out = int(duration * to_rate)
    x_old = np.linspace(0, duration, num=len(samples), endpoint=False)
    x_new = np.linspace(0, duration, num=n_out, endpoint=False)
    resampled = np.interp(x_new, x_old, samples)
    return resampled.astype(np.int16).tobytes()
