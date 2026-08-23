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

## Protocol (not yet implemented — build stage)

A fixed set of simulated audio inputs with scripted interruption points at
known offsets, run against the real pipeline, timings logged per turn and
summarized (p50/p95) across the set. Not yet designed in detail — needs a
small hand-built test protocol at build stage, per CLAUDE.md "Open
questions".

## Run it

```
python -m eval.run_eval
```

(Not yet implemented — see `run_eval.py`.)
