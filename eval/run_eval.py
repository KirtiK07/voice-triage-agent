"""Benchmark harness: runs a fixed set of scripted barge-in scenarios
against the real pipeline (real Piper synthesis standing in for the
caller's voice, real VAD, real STT, real Groq LLM, real Piper synthesis
for the response -- the same building blocks server.py's simulate_speech
uses, called directly here rather than over a WebSocket, since this is a
standalone script) and reports cutoff_latency_ms / recovery_latency_ms
p50/p95 across the set. See eval/README.md for what these two numbers
mean and why they're reported separately.

Each scenario: an opening utterance starts a turn: once the agent's
response is genuinely playing, a fixed "reaction delay" simulates the
time a caller takes to notice something worth interrupting about, then
an interrupting utterance is fed in frame-by-frame exactly like a real
barge-in, and the resulting TurnTimings is recorded.
"""

import asyncio
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path

from voice_agent import stt, tts
from voice_agent.audio_utils import resample_int16
from voice_agent.config import settings
from voice_agent.pipeline import CallSession, TurnTimings
from voice_agent.stt import transcribe
from voice_agent.tts import output_sample_rate, synthesize_stream
from voice_agent.turn_taking import UtteranceCapture
from voice_agent.vad import SAMPLE_RATE, get_detector

REACTION_DELAY_S = 0.5  # time after playback starts before the caller "interrupts"
TRAILING_SILENCE_MS = 800  # matches server.py's simulate_speech -- see DECISIONS.md

SCENARIOS = [
    (
        "My internet has been down for three days and I have a huge presentation tomorrow.",
        "Wait, actually never mind, I just got it working again.",
    ),
    (
        "I was charged twice for my subscription this month and I need a refund.",
        "Sorry, hold on, I see a third charge too, this is worse than I thought.",
    ),
    (
        "I can't log into my account, it keeps saying my password is wrong.",
        "Oh wait, caps lock was on, sorry, never mind, that fixed it.",
    ),
    (
        "The app keeps crashing every time I try to upload a photo.",
        "Actually it's not just photos, videos crash it too.",
    ),
    (
        "I need to change my shipping address before my order ships out today.",
        "Wait, actually can you just cancel the order entirely instead?",
    ),
]


async def _synthesize_16k(text: str) -> bytes:
    chunks = [chunk async for chunk in synthesize_stream(text)]
    audio = b"".join(chunks)
    pcm_16k = resample_int16(audio, output_sample_rate(), SAMPLE_RATE)
    silence_samples = int(SAMPLE_RATE * TRAILING_SILENCE_MS / 1000)
    return pcm_16k + b"\x00\x00" * silence_samples


async def _run_scenario(opening_text: str, interrupt_text: str) -> TurnTimings:
    sent_bytes = 0

    async def send_audio(chunk: bytes) -> None:
        nonlocal sent_bytes
        sent_bytes += len(chunk)

    session = CallSession(send_audio=send_audio)
    capture = UtteranceCapture()
    frame_bytes = capture.frame_size_samples * 2

    # Opening turn: same real capture -> STT -> start_turn path as a
    # genuine first utterance, no barge-in involved yet.
    opening_audio = await _synthesize_16k(opening_text)
    for i in range(0, len(opening_audio) - frame_bytes + 1, frame_bytes):
        capture.feed(opening_audio[i : i + frame_bytes])
    transcript = transcribe(capture.audio)
    capture.reset()
    await session.start_turn(transcript)

    # Wait for the response to genuinely start playing before interrupting --
    # a barge-in on a turn that hasn't started speaking yet isn't the
    # scenario being benchmarked here.
    while not session.is_speaking:
        await asyncio.sleep(0.01)
    await asyncio.sleep(REACTION_DELAY_S)

    # Feed the interrupting utterance frame-by-frame through the real
    # barge-in detector, exactly like a real caller talking over the agent.
    interrupt_audio = await _synthesize_16k(interrupt_text)
    timings = None
    interrupt_capture = UtteranceCapture()
    for i in range(0, len(interrupt_audio) - frame_bytes + 1, frame_bytes):
        frame = interrupt_audio[i : i + frame_bytes]
        if timings is None:
            fired = await session.feed_mic_frame(frame)
            if fired is not None:
                timings = fired
                interrupt_capture.feed(frame)
        else:
            interrupt_capture.feed(frame)

    if timings is None:
        raise RuntimeError(
            f"Interrupting utterance never triggered a barge-in for scenario: {interrupt_text!r}"
        )

    interrupt_transcript = transcribe(interrupt_capture.audio)
    await session.start_turn(interrupt_transcript, continuing=timings)
    await session.wait_for_turn()

    return timings


async def _main_async() -> dict:
    results = []
    for i, (opening, interrupt) in enumerate(SCENARIOS, start=1):
        print(f"[{i}/{len(SCENARIOS)}] {opening[:60]}...")
        t0 = time.perf_counter()
        timings = await _run_scenario(opening, interrupt)
        print(
            f"    cutoff={timings.cutoff_latency_ms:.0f}ms "
            f"recovery={timings.recovery_latency_ms:.0f}ms "
            f"(scenario took {time.perf_counter() - t0:.1f}s)"
        )
        results.append(asdict(timings))

    cutoffs = [r["t_playback_stopped"] * 1000 - r["t_vad_fire"] * 1000 for r in results]
    recoveries = [r["t_new_audio_start"] * 1000 - r["t_vad_fire"] * 1000 for r in results]

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "n_scenarios": len(SCENARIOS),
        "cutoff_latency_ms": {
            "p50": statistics.median(cutoffs),
            "p95": statistics.quantiles(cutoffs, n=20)[18] if len(cutoffs) >= 2 else cutoffs[0],
            "mean": statistics.mean(cutoffs),
        },
        "recovery_latency_ms": {
            "p50": statistics.median(recoveries),
            "p95": statistics.quantiles(recoveries, n=20)[18] if len(recoveries) >= 2 else recoveries[0],
            "mean": statistics.mean(recoveries),
        },
        "raw_runs": results,
    }
    return summary


def _warm_models() -> None:
    """Same reasoning as server.py's lifespan startup hook -- load all
    three models synchronously on the main thread before any scenario
    runs, to avoid a real first-load deadlock between torch (VAD) and
    onnxruntime (Piper) initializing concurrently on different threads.
    See DECISIONS.md "The end-to-end verification saga"."""
    print("Warming models (VAD, STT, TTS)...")
    get_detector(settings.vad_backend)
    stt.warm()
    tts.warm()
    print("Models warm.\n")


def main() -> None:
    _warm_models()
    summary = asyncio.run(_main_async())

    print()
    print(f"Cutoff latency   (playback stopped - VAD fired):  p50={summary['cutoff_latency_ms']['p50']:.0f}ms  p95={summary['cutoff_latency_ms']['p95']:.0f}ms")
    print(f"Recovery latency (new audio started - VAD fired): p50={summary['recovery_latency_ms']['p50']:.0f}ms  p95={summary['recovery_latency_ms']['p95']:.0f}ms")

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "latest.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
