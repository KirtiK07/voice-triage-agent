"""Orchestrates one call session: VAD listens continuously during TTS
playback, a barge-in cancels the in-flight LLM/TTS and restarts with the
new utterance, and every turn logs the three timestamps the benchmark
(see eval/) is built on:

  t_vad_fire        -- barge-in detected (voice_agent.vad fires)
  t_playback_stopped -- audio actually cut on the wire
  t_new_audio_start  -- the new response starts streaming back

Reported as two separate deltas, not one blended number, since they have
different failure modes -- see CLAUDE.md / DECISIONS.md:
  cutoff_latency   = t_playback_stopped - t_vad_fire   (buffering/cancellation)
  recovery_latency = t_new_audio_start  - t_vad_fire   (LLM+TTS pipeline)

Business logic not yet implemented -- scaffold stage only establishes
the interface and the timestamp data model.
"""

from dataclasses import dataclass


@dataclass
class TurnTimings:
    t_vad_fire: float | None = None
    t_playback_stopped: float | None = None
    t_new_audio_start: float | None = None

    @property
    def cutoff_latency_ms(self) -> float | None:
        if self.t_vad_fire is None or self.t_playback_stopped is None:
            return None
        return (self.t_playback_stopped - self.t_vad_fire) * 1000

    @property
    def recovery_latency_ms(self) -> float | None:
        if self.t_vad_fire is None or self.t_new_audio_start is None:
            return None
        return (self.t_new_audio_start - self.t_vad_fire) * 1000


class CallSession:
    """One caller's session: owns the VAD/STT/LLM/TTS wiring and the
    barge-in cancel/restart logic. Not yet implemented (build stage)."""

    def __init__(self) -> None:
        raise NotImplementedError("build stage: wire up VAD/STT/LLM/TTS orchestration")
