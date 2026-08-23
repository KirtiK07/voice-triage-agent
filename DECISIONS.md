# Decisions

Plain-language log of non-obvious technical choices, filled in as they come up.

## Deploy target: Vercel Hobby (free), chosen over Oracle Cloud/Fly.io/Railway

**The starting assumption was wrong and got corrected before it cost real
setup time.** Fly.io and Railway were the obvious first candidates for a
free always-on WebSocket host, both already ruled out live (not assumed)
during the prior `llm-cost-router` project's own hosting research — Fly.io
now requires a credit card with no free allowance at all, Railway's "Free
Plan" is only $1/month in usage credits, nowhere near enough for an
always-on process. Oracle Cloud's Always Free tier was the next real
candidate (genuinely free forever, 2 ARM OCPUs/12GB RAM) but needs the
user's own account, credit card for identity verification, and carries a
known risk of ARM capacity being unavailable by region — none of which
Claude Code can do or guarantee on the user's behalf.

**Then the user pointed out they were already logged into Vercel**, which
prompted re-checking whether Vercel could actually serve this project
rather than assuming "Vercel = frontend hosting, no persistent backend."
Current Vercel docs (loaded via the `vercel:knowledge-update` and
`vercel:vercel-functions` skills, not from stale training-data assumptions)
say Vercel Functions added native WebSocket support in public beta (June
2026), working with FastAPI over ASGI with zero extra config — this
project's exact planned stack.

**Verified with an actual deploy, not just docs — the user explicitly
asked for this ("if it works properly ... go ahead") rather than a
docs-only answer.** Built a throwaway scratch project (a FastAPI WebSocket
function bundling the real dependency chain: CPU-only `torch`,
`ctranslate2`, `faster-whisper`, `onnxruntime`, `piper-tts`) and deployed
it for real to Vercel Hobby using the user's existing account.

Findings from the real deploy, not assumed:
- **First deploy failed**: uncompressed bundle was 1.16GB, over the
  *default* 500MB Python function limit. Vercel's own current docs
  describe "5GB on Fluid Compute" as available without clearly flagging
  it as an opt-in beta — the real behavior only matched that once
  `VERCEL_SUPPORT_LARGE_FUNCTIONS=1` was set at the project level (build
  log then explicitly said "exceeds the standard size limit; enabling
  large functions (beta)"). Recorded here so the real project's build
  doesn't repeat this same failed-deploy cycle from a docs
  misunderstanding.
- **All five heavy imports succeeded** on the redeploy: torch (CPU),
  ctranslate2, faster-whisper, onnxruntime, piper. Peak memory across all
  of them: 315MB — well inside Vercel Hobby's 2GB limit. Worth comparing
  directly to the prior project's Render experience: Render's free tier
  (512MB) OOM'd three separate times on a *lighter* workload (just
  sentence-transformers + ONNX, no torch.hub network load) — Vercel's
  headroom here is real, not marginal.
- **A real Silero VAD model was loaded via `torch.hub.load` (network
  fetch from GitHub) and ran one real inference** — `speech_prob` came
  back correctly near-zero (0.0017) on a silent dummy frame, confirming
  the model is functionally correct, not just importable.
- One real dependency gap caught by the spike, not guessed: Silero VAD
  needs `torchaudio`, which wasn't in the original plan's dependency list
  — now known before the real project's `requirements.txt` is written,
  rather than discovered mid-build.
- WebSocket lifecycle was clean end-to-end: open → 7 streamed
  per-step messages → clean close (code 1000), ~3.7s total warm runtime,
  no crashes, no timeouts.
- One test-script bug surfaced and got fixed along the way (unrelated to
  Vercel itself): an `async def step()` helper was called without
  `await`, silently producing an empty report — caught by noticing
  `steps: []` in the response rather than assuming the deploy itself was
  broken. Worth remembering as a general lesson: an unexpectedly-empty
  result is itself a signal to double check the test harness before
  blaming the platform.

**Caveats carried forward into scaffold, stated plainly rather than
glossed over:** native WebSocket support is in public beta and could
change; Hobby's 300s hard connection-duration limit needs real client
reconnect-with-backoff logic (a documented Vercel pattern, not a
workaround); Hobby is restricted to personal/non-commercial use (fine for
a portfolio project, would not fit a real product). The scratch project
used for this spike was deleted after the test (`vercel project rm`) —
nothing left behind in the user's Vercel account.

**Why not set up Oracle Cloud as a backup too:** considered and explicitly
rejected — no redundancy benefit for a single interview-demo link, doubles
the maintenance surface (two deploy configs, two secret sets) for zero
payoff, and Oracle still carries real capacity risk even if provisioned.
Vercel remains the sole deploy target unless a real blocker surfaces once
the actual app is built against it.

## Scaffold stage: two real bugs caught before they shipped

**Route-shadowing bug, caught by reasoning about Starlette's routing
before it ever ran wrong.** The first draft of `server.py` registered
`app.mount("/", StaticFiles(...))` *before* the `/api/health` and
`/api/ws` route decorators. Starlette dispatches on the first path-prefix
match in registration order, not by whether the matched route actually
returns a non-404 — so a `"/"` mount registered first would have silently
swallowed every request to `/api/*`, static files or not. Fixed by moving
the mount to the end of the file, after the API routes, with a comment
explaining why the order matters (so a future edit doesn't reintroduce
it). A regression test (`test_static_index_served_and_api_routes_not_shadowed`)
locks this in.

**`webrtcvad` has no prebuilt wheels on PyPI for any platform — verified
via the PyPI JSON API, not discovered by guessing.** Installing the
original `requirements.txt` failed locally with `Microsoft Visual C++
14.0 or greater is required` (no MSVC Build Tools on this Windows dev
machine). Before just "fixing the local machine," checked whether this
was a local-only problem or a real risk for the Vercel deploy too:
queried `pypi.org/pypi/webrtcvad/json` directly and confirmed the latest
release (2.0.10) ships **only an sdist** — meaning Vercel's Linux build
would face the same C-compiler dependency, not just this laptop. Checked
for a maintained alternative rather than vendoring a compiler toolchain:
`webrtcvad-wheels` is a drop-in fork (same `import webrtcvad` API, same
functionality) that publishes real prebuilt wheels for manylinux, macOS,
and Windows across Python 3.6-3.13 — confirmed via the same JSON-API
check before switching to it. Swapped in `requirements.txt` with a
comment recording why, so a future contributor doesn't "helpfully"
revert it to the more obviously-named original package. Full dependency
install and the 5-test suite (including a real `uvicorn server:app` boot
smoke test, not just the ASGI test client) verified clean after the
swap.

## Build stage: TTS (Piper)

Implemented against Piper's real API (`PiperVoice.load`, `.synthesize()`,
`piper.download_voices.download_voice`) inspected directly in the venv
rather than guessed from docs. `download_voice()` fetches from Hugging
Face (`rhasspy/piper-voices`) into `models/piper/` (gitignored, downloaded
at runtime like the classifier snapshot pattern in `llm-cost-router`).
Piper's own `synthesize()` is a blocking, CPU-bound sync generator (one
`AudioChunk` per sentence) -- run on a background thread, chunks handed
to the async caller through a queue, so breaking out of iteration early
(the barge-in case) doesn't block on further ONNX inference.

**Real bug caught by a genuine network flake, then fixed and covered by
a regression test:** the first version's background-thread helper had a
bare `finally: chunk_queue.put(_SENTINEL)` with no exception handling.
A transient network error downloading the voice's `.onnx.json` config
(`urlopen error [WinError 10065] A socket operation was attempted to an
unreachable host`) got silently swallowed -- the caller just saw an
empty audio stream (0 chunks) instead of an error, failing
`test_synthesize_stream_yields_real_audio` with no indication why. Fixed
by putting exceptions on the queue too and re-raising them on the
consumer side; `test_synthesize_stream_propagates_real_failures`
(monkeypatches `_load_voice` to force a failure) locks this in so a
future refactor can't silently reintroduce it. Re-ran after the fix with
the model already cached: 4/4 passed in 5.67s (vs. 33+ minutes on the
run that hit the network flake -- confirms it really was transient, not
a code path that's slow every time).

## Build stage: VAD (barge-in trigger)

**Silero's exact frame-size constraint was verified by testing, not
assumed from docs:** the JIT-scripted `silero_vad` model raises "Input
audio chunk is too short" for anything other than exactly 512 samples
(32ms) at 16kHz -- 256, 480, 1024, and 160 were all tried and all
failed, confirming this is a hard requirement, not a flexible minimum.
This matches Silero's own documented supported chunk sizes (512 @ 16kHz,
256 @ 8kHz), now recorded here so `SileroDetector.frame_size_samples`
isn't "mysteriously" hardcoded to a caller who hasn't read the spike
notes.

**Deliberately did not use Silero's own `VADIterator` streaming
utility**, despite it existing for exactly this use case: inspecting its
source showed it fires on the very first frame that crosses `threshold`,
with no duration-accumulation knob -- too eager for this project's
"sustained ~200-300ms" design (avoiding false triggers on breaths/
coughs, per the user's locked decision). Implemented the
consecutive-speech-frame accumulation directly against the model's raw
per-frame probability instead, in `SileroDetector`/`WebRTCDetector`.

**Testing strategy split deliberately in two:** a real negative control
(50 consecutive real-inference calls on true silence, asserting none
fire) proves the model and our threshold wiring aren't broken, without
needing to craft synthetic "speech" audio that reliably scores as
speech -- that's Silero's own concern, not something worth re-testing
here. The accumulation *logic itself* (fires at exactly the Nth
consecutive frame, not N-1 or N+1) is tested separately by monkeypatching
only the per-frame probability call, isolating "does our counting logic
work" from "does the model correctly classify this audio" -- two
different things that a single audio-based test would conflate.
