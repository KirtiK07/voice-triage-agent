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
- **Barge-in trigger:** VAD (Silero, WebRTC VAD as a lighter fallback) —
  fast enough to be the interrupt signal itself; `faster-whisper` only
  transcribes *after* a barge-in fires (the "what," not the "when," since
  its partials are too slow at 1-2s context to be the trigger).
- **Pipeline:** WebSocket-based streaming STT → LLM → streaming TTS, with
  barge-in handled by detecting new caller speech during agent playback and
  cancelling/restarting the in-flight response.
- **Deploy target:** Vercel Hobby (free) — native WebSocket support
  (public beta) over FastAPI/Python ASGI, verified end-to-end with a real
  scratch deploy of the actual dependency chain (see "Open questions"
  below for the full spike results). Needs `VERCEL_SUPPORT_LARGE_FUNCTIONS=1`
  set on the real project (bundle exceeds the 500MB default with
  torch+ctranslate2+onnxruntime). Client needs reconnect-with-backoff
  logic for the 300s hard connection limit on Hobby.
- **Eval harness:** custom — benchmarks cutoff latency (playback stopped −
  VAD fired) and recovery latency (new response started − VAD fired),
  reported separately since they have different failure modes
  (buffering/cancellation vs. LLM+TTS pipeline speed).

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
build — in progress. Scaffold complete (see below). All four pipeline
pieces implemented for real and tested: TTS (Piper, streaming),
VAD (Silero/WebRTC, both real), STT (faster-whisper, real round-trip
tested against real Piper output), LLM (Groq, same model as
llm-cost-router). `pipeline.py`'s `CallSession` ties them together with
the actual barge-in cancel/restart state machine and the three-timestamp
benchmark instrumentation — the project's core hard problem, now working
and tested (34/35 passing; 1 skipped pending a real `GROQ_API_KEY` in
`.env`, deferred to document stage same as the prior project's rhythm).
Several real bugs caught during this stage before they shipped — see
DECISIONS.md for the full list (exception-swallowing in TTS, a
timestamp-ownership design issue and an async/race bug in the pipeline,
two test-design races). **Not yet done:** `server.py`'s `/api/ws` handler
still only accepts+closes (scaffold stub) — needs real wiring to
`CallSession` plus the browser client's real microphone capture/
playback/barge-in UI (currently just a connectivity-check stub).

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
- Demo format: **partly resolved.** Live host + a recorded video, not
  recording alone — barge-in specifically needs to be *seen* live to be
  believed. Live-host choice still open (see below). The recording
  itself, however, is settled: stays **local-only**, saved into the
  project folder for the user's own reference, gitignored, never
  committed or deployed, and never referenced by a path in the README
  (which would break once it's untracked). See [[demo-media-local-only]]
  in Claude's memory — this is now a standing preference across projects,
  not a one-off for this repo.
- Live host for the WebSocket backend: **resolved — Vercel Hobby
  (free).** Fly.io confirmed dead (no free tier, credit card required) and
  Railway confirmed insufficient ($1/month usage credit — not enough for
  an always-on server) as of 2026-08-23. Oracle Cloud Always Free was the
  leading fallback but needs the user's own account/card setup with no
  guarantee of ARM capacity in-region.

  Then re-evaluated: Vercel added native WebSocket support in public beta
  (June 2026), works with FastAPI/Python over ASGI — this project's exact
  stack — and the user is already logged into an existing Vercel account
  (zero setup friction). **Verified with a real deploy, not just docs**
  (scratch spike, since discarded): a FastAPI WebSocket function bundling
  the actual planned dependency chain (`torch` CPU, `ctranslate2`,
  `faster-whisper`, `onnxruntime`, `piper-tts`) built and ran cleanly on
  Vercel Hobby. Real findings from that spike:
  - Uncompressed bundle came to **1.16GB**, over the *default* 500MB
    Python function limit — despite Vercel's own current docs describing
    5GB as available "on Fluid Compute" without flagging it as opt-in.
    Fixed with one project-level env var, `VERCEL_SUPPORT_LARGE_FUNCTIONS=1`
    (confirmed in the build log: "exceeds the standard size limit;
    enabling large functions (beta)") — a config change, not a code
    change, so this still counts as deploying the real stack "as-is."
  - All five heavy imports succeeded; a **real Silero VAD model was
    loaded via `torch.hub` and ran a real inference** (`speech_prob`
    correctly near-zero on a silence frame) — not just "imports work,"
    actually functionally correct.
  - **Peak memory: 315MB** — for comparison, Render's 512MB free tier
    already OOM'd on a *lighter* workload in the prior project; this
    leaves ~1.7GB of headroom on Vercel's 2GB Hobby limit.
  - Clean WebSocket lifecycle: open → 7 messages streamed → clean close
    (code 1000), no crashes, no timeouts (total runtime ~3.7s warm).
  - One dependency gap caught and fixed: Silero VAD needs `torchaudio`,
    not listed in the original plan — added to the real requirements.txt.

  Real caveats carried into scaffold, not hidden: 300s hard connection
  limit on Hobby (needs client reconnect logic — a documented Vercel
  pattern, not a workaround); Hobby is personal/non-commercial use only
  (fits this portfolio project); the beta status of native WebSocket
  support means this should be re-verified once the real app is built,
  not assumed permanent from one spike.
