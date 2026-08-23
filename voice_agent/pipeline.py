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

A subtlety that shaped this design: t_vad_fire and t_playback_stopped
belong to the turn that got INTERRUPTED, but t_new_audio_start belongs to
the NEXT turn that replaces it -- so a single TurnTimings record has to
span the cancel/restart boundary rather than being recreated fresh per
`start_turn()` call. See `start_turn`'s `continuing` parameter.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

from voice_agent import llm, tts
from voice_agent.vad import BargeInDetector, get_detector


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


SendAudio = Callable[[bytes], Awaitable[None]]


class CallSession:
    """One caller's session: owns the VAD wiring and the barge-in
    cancel/restart logic. Does not own microphone capture or
    end-of-speech detection for the caller's own new utterance --  that's
    the WebSocket handler's job (audio I/O), this class's job is strictly
    the cancel/restart state machine and its timing.
    """

    def __init__(self, send_audio: SendAudio, vad_backend: str | None = None) -> None:
        from voice_agent.config import settings

        backend = vad_backend or settings.vad_backend
        self._detector: BargeInDetector = get_detector(backend)
        self._send_audio = send_audio
        self._playback_task: asyncio.Task | None = None
        self._pending_timings: TurnTimings | None = None
        self.turn_history: list[TurnTimings] = []

    @property
    def is_speaking(self) -> bool:
        return self._playback_task is not None and not self._playback_task.done()

    async def start_turn(self, transcript: str, continuing: TurnTimings | None = None) -> None:
        """Cancel any in-flight turn, then start a new one speaking the
        LLM's response to `transcript`.

        `continuing`: pass the TurnTimings from `feed_mic_frame`'s
        barge-in return value when this turn is the direct result of an
        interruption, so t_new_audio_start lands on the same record as
        that interruption's t_vad_fire/t_playback_stopped. Omit for a
        turn that starts fresh (e.g. the call's first turn).
        """
        await self._cancel_playback_if_active()
        timings = continuing if continuing is not None else TurnTimings()
        self._pending_timings = timings
        self._playback_task = asyncio.create_task(self._run_turn(transcript, timings))

    async def _run_turn(self, transcript: str, timings: TurnTimings) -> None:
        first_chunk_sent = False
        sentence_buffer = ""
        try:
            async for token in llm.stream_response(transcript):
                sentence_buffer += token
                if token and token[-1] in ".!?":
                    async for chunk in tts.synthesize_stream(sentence_buffer):
                        if not first_chunk_sent:
                            timings.t_new_audio_start = time.perf_counter()
                            first_chunk_sent = True
                        await self._send_audio(chunk)
                    sentence_buffer = ""
            if sentence_buffer.strip():
                async for chunk in tts.synthesize_stream(sentence_buffer):
                    if not first_chunk_sent:
                        timings.t_new_audio_start = time.perf_counter()
                        first_chunk_sent = True
                    await self._send_audio(chunk)
        except asyncio.CancelledError:
            timings.t_playback_stopped = time.perf_counter()
            raise
        finally:
            # Recorded once per barge-in event, not once per _run_turn call:
            # a `continuing` timings object is the *same* instance reused
            # across the cancel/restart boundary (see start_turn), so on a
            # rapid double-interrupt (turn N+1 itself gets cut off before
            # sending any audio) this method runs again for the same
            # object -- guard on identity, not just "has t_vad_fire", or
            # it would append a duplicate reference and the two entries
            # would silently diverge/reconverge in confusing ways once a
            # later turn finally gets audio out. Known remaining
            # limitation, not fully solved: a double-interrupt still
            # overwrites t_vad_fire/t_playback_stopped with the second
            # interruption's timestamps rather than preserving the
            # first -- acceptable for now since it requires two barge-ins
            # faster than one LLM+TTS round-trip, an edge case rare enough
            # not to block build stage on. See DECISIONS.md.
            if timings.t_vad_fire is not None and timings not in self.turn_history:
                self.turn_history.append(timings)

    async def feed_mic_frame(self, frame: bytes) -> TurnTimings | None:
        """Feed one VAD frame while the agent is speaking. Returns the
        TurnTimings record for the interrupted turn if this call
        triggered a barge-in -- pass it to the next `start_turn()`
        call's `continuing` param. Returns None otherwise (not speaking,
        or no speech detected yet).

        Deliberately `async`, not sync: `task.cancel()` only *schedules*
        cancellation -- the actual `CancelledError` delivery (and
        `_run_turn`'s recording of `t_playback_stopped`) happens whenever
        the event loop next reaches that task, not synchronously at the
        call site. Awaiting the task here guarantees
        `t_playback_stopped` is really set before this returns, rather
        than relying on incidental event-loop scheduling order between
        two different coroutines -- an earlier sync-returning version of
        this method had exactly that latent race.
        """
        if not self.is_speaking:
            return None
        fired = self._detector.process_frame(frame)
        if not fired:
            return None
        task = self._playback_task
        timings = self._pending_timings
        assert task is not None and timings is not None  # is_speaking guarantees both
        timings.t_vad_fire = time.perf_counter()
        self._detector.reset()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return timings

    async def _cancel_playback_if_active(self) -> None:
        if self._playback_task is not None and not self._playback_task.done():
            self._playback_task.cancel()
            try:
                await self._playback_task
            except asyncio.CancelledError:
                pass
