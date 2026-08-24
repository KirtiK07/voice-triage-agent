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

## Build stage: STT (faster-whisper) and the sample-rate mismatch

Piper's native output is **22050Hz**, but faster-whisper (and Silero VAD)
require **16kHz** -- checked directly (`voice.config.sample_rate`), not
assumed. This is a real pipeline-level concern, not just a test-fixture
detail: the browser client's microphone capture needs to request 16kHz
directly from `getUserMedia` so nothing server-side has to resample
incoming caller audio on the hot path; Piper's *output* (for playback
only, never fed back into VAD/STT) doesn't need resampling at all, since
browser audio playback handles arbitrary sample rates natively. Recorded
here so build-stage work on the browser client doesn't rediscover this
by trial and error.

Tested with a genuine round-trip, not canned audio: Piper synthesizes a
real sentence, a test-only linear-interpolation resampler (not claimed
production-quality, explicitly documented as such) converts it to
16kHz, and faster-whisper transcribes it -- asserting the actual content
words come back, not just "doesn't crash." This is the strongest
available end-to-end check that TTS output is intelligible enough for
the STT half of the round-trip to work, without needing a
human-recorded audio fixture in the repo.

## Build stage: LLM (Groq)

Same provider and model as `llm-cost-router` (`openai/gpt-oss-20b`,
Groq's free tier) -- deliberately not re-evaluating alternatives, since
that model is already verified working in production on the sibling
project; no reason to re-gamble on a different free-tier model's
availability/quality for this one. Streaming via `stream=True`, same
client-injection testing pattern as that project's `GroqProvider`
(`client` param so tests can fake the SDK instead of requiring a real
key).

**One real gap, stated plainly rather than hidden:** the one genuine
end-to-end integration test (`test_stream_response_real_groq_call`) is
`skipif`'d on `GROQ_API_KEY` not being set -- this project's `.env`
hasn't been filled in with real keys yet (that happens at document
stage, same rhythm as `llm-cost-router`). The fake-client tests cover
the actual logic this module owns (token ordering, empty-delta
filtering, early-stop-on-cancel), but the real Groq wire format hasn't
been exercised against *this* module yet, only inferred from the
sibling project's working code. Needs a real key before test stage can
call this fully verified.

## Build stage: pipeline.py -- the barge-in cancel/restart state machine

This is the project's actual hard problem, so it got the most design
scrutiny of anything built so far. Two real correctness issues were
caught and fixed before they became flaky-test or production bugs, not
after:

**Timestamp ownership spans the cancel/restart boundary.**
`t_vad_fire`/`t_playback_stopped` belong to the turn that got
interrupted; `t_new_audio_start` belongs to the turn that replaces it.
Naively creating a fresh `TurnTimings` per `start_turn()` call would
have made `recovery_latency_ms` uncomputable (the two halves would live
on different objects). Fixed by having `feed_mic_frame` return the
in-flight turn's `TurnTimings` object, and `start_turn`'s `continuing`
parameter accept it back in for the replacement turn -- same mutable
object, populated across the boundary. A known remaining gap: a rapid
*second* barge-in before the replacement turn sends any audio overwrites
`t_vad_fire`/`t_playback_stopped` with the second interruption's
timestamps rather than preserving the first. Not fixed -- it requires
two barge-ins faster than one LLM+TTS round-trip apart, judged rare
enough not to block build stage on, but recorded here rather than
silently left as a surprise.

**`feed_mic_frame` had to become `async`, not stay sync, to avoid a real
race.** `asyncio.Task.cancel()` only *schedules* cancellation --
`_run_turn`'s `except CancelledError` block (which records
`t_playback_stopped`) runs whenever the event loop next reaches that
task, not synchronously at the call site. A sync `feed_mic_frame`
returning a `TurnTimings` object right after calling `.cancel()` would
have returned it with `t_playback_stopped` still `None`, correct only by
incidental event-loop scheduling luck. Fixed by making the method
`async` and `await`-ing the cancelled task before returning, which
guarantees the timestamp is really set.

**Testing approach:** VAD/LLM/TTS are all faked with controllable,
deterministic timing (a `_FakeDetector` fired on demand, fake async
generators with an optional artificial delay) -- their own correctness
is covered separately in `test_vad.py`/`test_llm.py`/`test_tts.py`, so
these tests exercise only the orchestration logic: does a barge-in
actually cancel playback, do the three timestamps land in the right
place, does `turn_history` stay clean. Two of the six tests originally
failed for a genuine test-design reason, not a pipeline bug: the
"instant" fake generators (no artificial delay) could race to full
completion within a single `asyncio.sleep(0)`, so the test's assumption
that the task was "still running" after yielding control once wasn't
actually guaranteed. Fixed with a `slow_session` fixture (a deliberately
slowed-down fake TTS stream) for any test that needs to interrupt a turn
mid-flight, rather than relying on incidental timing.

## Build stage: end-of-utterance detection (turn_taking.py) and the WS handler

Barge-in detection (vad.py) answers "has the caller started talking";
knowing when to stop listening and actually respond needs the opposite
signal -- "has the caller finished talking" (sustained silence after
speech). Rather than load a second VAD model for this, `UtteranceCapture`
calls the same `get_detector()` factory vad.py already exposes -- since
it caches by backend+threshold, this returns the identical instance
`CallSession` uses for barge-in, with no extra memory/compute and no
special plumbing to share it. Safe because the two uses are always
sequential per call (listening for a new utterance vs. listening for a
barge-in during playback), never concurrent.

`/api/ws`'s state branching is deliberately simple: `session.is_speaking`
itself *is* the mode flag (feed frames to the session for barge-in
detection when true, to the utterance capture when false) -- no separate
state variable to keep in sync, since `CallSession` already tracks this
correctly (verified in test_pipeline.py).

**Real testing gap, stated plainly:** the browser client (`client.js`,
`mic-worklet.js`) has no automated test coverage -- there's no JS test
framework in this project, and the actual behavior that matters
(getUserMedia mic capture, AudioWorklet timing, gapless-but-interruptible
playback scheduling) can't be meaningfully unit-tested without a real
browser and a real microphone anyway. Syntax-checked with `node --check`
(catches typos, not behavior) and smoke-tested via browser automation up
to the point of the code correctly requesting microphone access -- the
page loads, all static assets resolve at the paths the client actually
references (`client.js`, `mic-worklet.js`), clicking "Start call"
disables the button and correctly triggers the `getUserMedia` flow with
the right status update, no console errors before that point. Automation
could not go further: `getUserMedia` opens a real native OS/browser
permission dialog that blocks the page's renderer (confirmed directly --
`Page.captureScreenshot` timed out twice in a row right after the
click), which is expected, correct behavior, not a bug, but means the
actual mic-capture -> WebSocket -> playback -> barge-in loop has **not**
been verified end-to-end yet. That needs a human, with a real
microphone, clicking through it -- flagged here rather than glossed
over. Also still blocked on a real `GROQ_API_KEY` (see the LLM section
above) for the same full-loop test.

## The end-to-end verification saga: three real bugs that all looked like "hangs"

Closing the "not yet verified end-to-end" gap (see the build-stage entries
above) required actually running the full VAD -> STT -> LLM -> TTS ->
barge-in loop for real, with a real `GROQ_API_KEY`. Since browser
automation can't grant a real microphone permission (confirmed directly
earlier), this needed a way to exercise the real pipeline without a mic
-- which is what `simulate_speech` (see `server.py`'s docstring) was
built for. Getting simulate_speech to actually work end-to-end surfaced
three real, unrelated bugs, each of which looked identical from the
outside (client waits forever, nothing happens) but had completely
different causes. Recorded here in the order found, since the diagnosis
process itself is worth keeping -- each false lead was ruled out with
evidence, not assumption, before moving to the next hypothesis.

**Bug 1 (real, fixed): first-load deadlock between torch and onnxruntime.**
The very first real WebSocket test hung with flat 0% CPU and no pending
network I/O (checked directly via `Get-Process` CPU deltas and `netstat`
-- ruled out "slow but working" before calling it a hang). `CallSession`'s
constructor loads Silero (torch) synchronously on the main thread at
connection time; `tts.synthesize_stream()`'s first call loads Piper
(onnxruntime) on a separate background thread shortly after. The two
libraries' first-time thread-pool initialization racing on a fresh
process is a known class of problem. **Fix:** added a FastAPI `lifespan`
startup hook that warms all three models (VAD, STT, TTS) synchronously
on the main thread, before any request is served -- removes *all*
background-thread involvement from first-time model loading, which is a
strictly safer fix than trying to pin down the exact race, and has the
side benefit of removing cold-start latency from the first real user
turn.

**Bug 2 (real, fixed): Piper's synthesized audio has no trailing silence,
so end-of-utterance never fires.** After Bug 1's fix, a fully-instrumented
trace (temporary debug prints through every step -- kept deliberately
crude and removed after, not left in production code) showed
`simulate_speech` completing entirely successfully: TTS synthesized real
audio, resampling worked, `process_pcm` fed every frame through the real
VAD/capture logic -- but no transcript event was ever sent. Piper ends
its output right after the last phoneme, with no natural trailing
silence, so `UtteranceCapture`'s end-of-speech detector (sustained
silence *after* speech, 600ms default -- see `turn_taking.py`) never
saw enough silence to fire. The pipeline wasn't stuck; it was correctly,
silently, waiting for a silence that would never come. **Fix:**
`simulate_speech` appends 800ms of real trailing silence to the
synthesized "caller" audio before feeding it in -- comfortably above the
threshold, and using the exact same code path a real caller's natural
pause would exercise, not a special case.

**Bug 3 (real, fixed, the actual root cause behind the scariest-looking
symptom): `GROQ_API_KEY` was never in the real process environment.**
With Bugs 1 and 2 fixed, transcription started working (real STT results
arrived correctly) but the agent's response never came -- flat CPU
again, plus an established HTTPS connection sitting idle (checked via
`netstat`, which turned out to be a red herring, not evidence of an
in-flight request). Isolated by adding a plain HTTP debug endpoint that
called `llm.stream_response()` directly, *outside* the
`asyncio.create_task()` context `CallSession` normally uses -- this
failed **instantly** with `groq.GroqError: The api_key client option
must be set`, not a hang at all. Root cause: `voice_agent.config.settings`
(pydantic-settings) parses `.env` into its own `Settings` instance, but
nothing ever loads `.env` into the real OS process environment when
running the actual server -- only `tests/conftest.py`'s `load_dotenv()`
call does that, and only for pytest. `llm.py`'s `_get_client()` called
bare `AsyncGroq()`, which falls back to reading `os.environ` directly --
so it silently had no key, every time, outside of tests. **Fix:**
`_get_client()` now explicitly passes `api_key=settings.groq_api_key`.

**Why bug 3 looked like a hang and not an error, which is arguably the
more important fix:** the failing call happened inside
`CallSession._run_turn`, which runs as a fire-and-forget
`asyncio.create_task()` that `start_turn()` deliberately never awaits
(so barge-in can cancel it later). Nothing was watching that task for
exceptions -- Python only surfaces an unretrieved task exception when the
task is garbage-collected, which during a live session holding a
reference to `self._playback_task` might be never. **Fix, independent of
the specific API-key bug:** `_run_turn` now has a real `except Exception`
clause (previously only `except asyncio.CancelledError` existed) that
always logs the error and optionally calls an `on_error` callback;
`server.py` wires this to send a `{"event": "error", ...}` message to
the client. This is a general robustness fix, not specific to Groq --
any future error in the LLM/TTS chain will now be visible instead of
silently vanishing. Two regression tests
(`test_run_turn_error_is_reported_via_on_error_not_swallowed`,
`test_run_turn_error_is_logged_even_without_on_error`) lock this in.

**Final verification, once all three were fixed:** a real end-to-end run
with the real Groq key -- simulated caller speech transcribed correctly,
agent response streamed back as real audio, and, critically, a second
simulated utterance sent mid-playback correctly triggered a real
`barge_in` event, cut the first response short (282KB delivered vs. the
~530-600KB a completed response normally runs), and started a new
response in ~2s. This is the project's actual core feature, verified
working for real, not assumed from unit tests alone.

## Deploy: pyproject.toml broke the real Vercel build

The first real production deploy (after the throwaway scratch spike
already validated the dependency chain) failed at the build step with
`uv lock ... error: No 'project' table found in: pyproject.toml`.
Root cause: this repo's `pyproject.toml` only ever held tool config
(`[tool.pytest.ini_options]`, `[tool.ruff]`) -- it was never meant to
declare project metadata or dependencies, since those live in
`requirements.txt`. Vercel's Python build detects `pyproject.toml` as a
possible dependency manifest and runs `uv lock` against it; without a
valid `[project]` table (PEP 621), that fails outright rather than
falling back to `requirements.txt`. The scratch spike from plan stage
never hit this because it never had a `pyproject.toml` at all.

**Fix:** moved the two tool configs to their own native files
(`pytest.ini`, `ruff.toml`) and deleted `pyproject.toml` entirely, rather
than adding a minimal-but-unused `[project]` table -- that would create
a second, easy-to-drift source of truth for dependencies alongside
`requirements.txt` for no real benefit. Verified locally before
redeploying: full test suite still finds `pytest.ini`'s config
(56/56 passing) -- `ruff.toml`'s syntax wasn't independently verified
(ruff isn't a project dependency), but it's Ruff's standard documented
format, not something worth installing a tool just to check two lines.

## Deploy: public/ was never actually in the deployed bundle

Second real deploy failure, after the pyproject.toml fix: the deployed
function crashed on import with `RuntimeError: Directory 'public' does
not exist`, at `server.py`'s `app.mount("/", StaticFiles(directory=
"public", ...))` line. Root cause: Vercel's Python bundler includes
files it can determine are "reachable" via static analysis of actual
Python imports -- `public/` is never imported by any Python code, it's
only referenced by a runtime string literal (`"public"`) passed to
`StaticFiles`, which the bundler has no way to trace. The scratch spike
never hit this because it had no static files at all.

**Fix:** added `includeFiles: "public/**"` to `vercel.json`'s function
config -- the explicit override Vercel's own docs describe for exactly
this situation (files a function needs that static analysis can't
discover on its own). Also updated `excludeFiles` to reference the real
current dev-only files (`pytest.ini`, `ruff.toml`) instead of the
now-deleted `pyproject.toml`.
