# Low-Latency Interruptible Voice Agent for Support Ticket Triage — CLAUDE.md

## What this project is
A voice interface for triaging incoming support tickets: a caller describes
their issue by voice, the agent classifies urgency/category in real time,
and — the actual hard problem this project is about — the caller can
interrupt (barge in) mid-response and the agent gracefully stops and
reprocesses instead of talking over them or ignoring the interruption.
Most voice-agent demos can't be interrupted, which makes them unusable for
real conversation; this is a genuine, current hard problem in voice AI.
Built for a portfolio audience (AI/ML hiring managers) as a technically
ambitious, non-wrapper voice AI project with a benchmarked optimization
axis, per `../GitHub_Profile_Revamp_Plan.md` §4. Runs at $0 ongoing cost
(local/free-tier only) — a constraint, not just a feature, matching the
discipline of the prior `llm-cost-router` project.

## Stack
- **STT:** local, streaming-capable — `faster-whisper` (CTranslate2-backed
  Whisper, runs on CPU) is the leading candidate; confirm streaming/partial-
  transcript support works well enough for real barge-in detection before
  committing, or fall back to chunked short-window transcription if true
  streaming turns out impractical on CPU.
- **LLM:** Groq (fast, free-tier, matches the $0 constraint and the prior
  project's provider choice) — needs streaming output so response
  generation can be interrupted mid-stream, not just mid-playback.
- **TTS:** local or free-tier, streaming-capable — needs research at plan
  stage (see Open questions); options include a local model (e.g. Piper)
  or a free tier of a hosted streaming TTS API, evaluated the same
  research-before-assuming way Redis/hosting options were evaluated on the
  prior project.
- **Pipeline:** WebSocket-based streaming STT → LLM → streaming TTS, with
  barge-in handled by detecting new caller speech during agent playback and
  cancelling/restarting the in-flight response.
- **Eval harness:** custom — benchmarks time-to-first-audio-byte and
  interruption-recovery latency (the two metrics real voice AI teams care
  about, per the plan), not just a "does it work" demo.

## Workflow rules (inherited from parent workspace)
- Work in stages: plan → scaffold → build → test → document → deploy.
- Stop and summarize between stages before continuing.
- No stage is "done" without passing tests written alongside the code (not after).
- No hardcoded secrets — use `.env`, keep it in `.gitignore`.
- Keep DECISIONS.md updated in plain language for any non-obvious technical choice.
- README.md must cover: problem, architecture, how to run, benchmark/results (if applicable).
- **Do not create a GitHub repo or push until explicitly told to at the deploy stage.**
  Ask first, using this shape: "Should I create a GitHub repo called
  `KirtiK07/<project-name>` and push?"
- Git identity for this repo is set locally (not global) as:
  ```
  git config user.name "Kirti Kolare"
  git config user.email "kirtikolare15@gmail.com"
  ```

## Current stage
plan

## Open questions
- STT choice: does `faster-whisper` actually deliver low-enough-latency
  partial transcripts on CPU for barge-in detection to feel real, or does
  this need a different local model / a lighter VAD-based interrupt signal
  (detect "caller started talking" via voice-activity-detection alone,
  separately from full transcription)? Needs a quick spike before
  committing at scaffold stage.
- TTS choice: which free/local streaming TTS actually has usable streaming
  output (not just non-streaming synthesis) — needs the same
  research-before-assuming pass the prior project gave Redis/hosting.
- Eval set: what does a fair benchmark for "interruption-recovery latency"
  even look like — needs a small hand-designed test protocol (e.g.
  simulated audio inputs with scripted interruption points at fixed
  offsets), to be resolved at plan stage before scaffold begins.
- Demo format: a live hosted demo for a voice pipeline is a much bigger
  lift than the router project's Streamlit UI (WebSocket audio streaming
  needs a real backend host, not a $0 static/serverless option) — likely
  means a local-only demo + recorded GIF/video for the README, not a
  public live URL. Confirm this tradeoff explicitly before document stage.
