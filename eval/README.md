# Eval harness

Benchmarks cutoff latency and recovery latency (see
`voice_agent/pipeline.py`'s `TurnTimings`), reported separately since they
have different failure modes:

- **Cutoff latency** (`t_playback_stopped - t_vad_fire`) — a
  buffering/cancellation problem: how fast the agent actually stops
  talking once a barge-in is detected.
- **Recovery latency** (`t_new_audio_start - t_vad_fire`) — an
  LLM+TTS pipeline problem: how fast a new response starts streaming
  back after the interruption.

## Protocol

Five scripted scenarios (`SCENARIOS` in `run_eval.py`), each an opening
support-ticket-style utterance plus an interrupting line a caller might
plausibly cut in with. For each: the opening line starts a real turn
(real STT → real Groq → real Piper), a fixed 500ms "reaction delay" waits
for the response to genuinely start playing, then the interrupting line
is synthesized and fed frame-by-frame through the real barge-in detector
— the exact same code path a real caller's voice takes, not a mocked
shortcut. `TurnTimings` is recorded per scenario; p50/p95/mean are
reported across the set.

Caller audio (both the opening and interrupting lines) is synthesized
with the same Piper voice used for agent responses — see
`server.py`'s `simulate_speech` docstring and DECISIONS.md for why this
stands in for a human caller here, and its one real caveat (Piper's
output needs trailing silence appended before end-of-speech detection
will fire on it, since Piper doesn't naturally pad its own output).

## Run it

```
python -m eval.run_eval
```

Needs a real `GROQ_API_KEY` in `.env` — every scenario makes a real LLM
call. Takes a few seconds per scenario (real STT/LLM/TTS round-trips
twice per scenario). Writes `eval/results/latest.json`.

## Results

Latest run (5 scenarios, real API calls, no simulated numbers):

| Metric | p50 | p95 | mean |
|---|---|---|---|
| Cutoff latency (audio actually stops) | 77 ms | 108 ms | 60 ms |
| Recovery latency (new response starts) | 2207 ms | 3155 ms | 2272 ms |

Cutoff is fast and consistent — cancelling an in-flight Piper stream and
stopping playback doesn't have much room to vary. Recovery is
dominated by the real STT → LLM → TTS round-trip for the interrupting
utterance, which is the actual bottleneck worth optimizing further (see
the project README's "What I'd do with more time").

Raw numbers: [`results/latest.json`](./results/latest.json).
