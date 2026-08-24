"""Vercel entrypoint (must be named app.py/index.py/server.py/main.py/
wsgi.py/asgi.py at the project root, defining a top-level `app` -- see
DECISIONS.md "Deploy target"). Local dev: `uvicorn server:app --reload`.

WebSocket protocol (see webapp/client.js for the browser side):
  Client -> server:
    binary frames: raw 16-bit PCM mono audio at 16kHz, any chunk size
      (server-side buffers into whatever frame size the active VAD
      backend needs -- see turn_taking.py/vad.py).
    text frames: JSON control messages --
      {"event": "simulate_speech", "text": "..."} synthesizes the given
      text into audio and feeds it through the exact same real
      VAD/capture/barge-in code path as genuine microphone frames (not a
      separate mocked path) -- see DECISIONS.md "Build stage:
      simulate_speech" for why this exists: it's the only way to
      exercise and demo the full pipeline without a real microphone
      (browser automation cannot grant a real OS mic-permission dialog),
      and doubles as a real interview-demo control for anyone without a
      working mic on hand.
  Server -> client: binary frames = raw 16-bit PCM audio for playback, at
    `output_sample_rate` (Piper's native rate, sent in the initial
    "ready" message -- never assumed by the client).
    Text frames = JSON events: {"event": "ready", ...},
    {"event": "listening"}, {"event": "transcript", "text": ...},
    {"event": "barge_in"}.
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState

from voice_agent import stt, tts
from voice_agent.audio_utils import resample_int16
from voice_agent.config import settings
from voice_agent.pipeline import CallSession
from voice_agent.turn_taking import UtteranceCapture
from voice_agent.vad import SAMPLE_RATE, get_detector

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Warm all three models synchronously on the main thread, deliberately
    # NOT via asyncio.to_thread -- this isn't just about avoiding cold-start
    # latency on the first real call, it fixes a genuine deadlock. Loading
    # Piper (onnxruntime) for the first time on a background thread, in a
    # process that already has torch loaded (from VAD), hung indefinitely
    # (confirmed directly: flat 0% CPU on both the request-handling thread
    # and the synthesis thread, no pending network I/O -- ruled out slow-
    # but-working before concluding "hung"). The exact root cause wasn't
    # pinned down further (likely conflicting first-time OpenMP/MKL
    # thread-pool init between the two libraries, a known class of issue),
    # but blocking the event loop here is safe -- no requests are being
    # served yet -- and it's a strictly stronger fix than moving the
    # background-thread work elsewhere: it removes *all* background-thread
    # involvement from first-time model loading, not just concurrent
    # background threads, so it doesn't depend on the theory being exactly
    # right. See DECISIONS.md "Model warm-up and the torch/onnxruntime
    # deadlock".
    logger.info("Warming models: VAD (%s), STT, TTS...", settings.vad_backend)
    get_detector(settings.vad_backend)
    stt.warm()
    tts.warm()
    logger.info("Models warm.")
    yield


app = FastAPI(title="Voice Triage Agent", lifespan=lifespan)


@app.get("/api/health")
async def health():
    return {"status": "ok", "vad_backend": settings.vad_backend}


@app.websocket("/api/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()

    async def send_audio(chunk: bytes) -> None:
        await websocket.send_bytes(chunk)

    async def on_error(exc: BaseException) -> None:
        # See CallSession.__init__ and DECISIONS.md: without this, a real
        # error in the LLM/TTS background task previously died completely
        # silently -- the client just waited forever with no message and
        # no visible symptom, looking exactly like a hang.
        await websocket.send_text(json.dumps({"event": "error", "message": str(exc)}))

    session = CallSession(send_audio=send_audio, on_error=on_error)
    capture = UtteranceCapture()
    frame_bytes = capture.frame_size_samples * 2  # 16-bit PCM = 2 bytes/sample
    mic_buffer = bytearray()
    pending_timings = None  # set when a barge-in fires; consumed by the next start_turn

    await websocket.send_text(
        json.dumps(
            {
                "event": "ready",
                "sample_rate_in": SAMPLE_RATE,
                "sample_rate_out": tts.output_sample_rate(),
                "frame_size_samples": capture.frame_size_samples,
            }
        )
    )

    async def process_pcm(pcm_bytes: bytes) -> None:
        """Runs incoming 16-bit PCM (real mic frames, or simulate_speech's
        synthesized audio -- same code path either way) through the
        real VAD/capture/barge-in logic. Shared so simulate_speech
        cannot silently diverge from the genuine microphone path."""
        nonlocal pending_timings
        mic_buffer.extend(pcm_bytes)

        while len(mic_buffer) >= frame_bytes:
            frame = bytes(mic_buffer[:frame_bytes])
            del mic_buffer[:frame_bytes]

            if session.is_speaking:
                timings = await session.feed_mic_frame(frame)
                if timings is not None:
                    await websocket.send_text(json.dumps({"event": "barge_in"}))
                    capture.reset()
                    capture.feed(frame)  # the frame that triggered it is real caller speech
                    pending_timings = timings
                continue

            utterance_complete = capture.feed(frame)
            if not utterance_complete:
                continue

            audio = capture.audio
            capture.reset()
            # Blocking CPU-bound inference -- offloaded to a thread so it
            # doesn't stall the event loop (and any other concurrent
            # WebSocket traffic) for the duration of transcription.
            transcript = await asyncio.to_thread(stt.transcribe, audio)
            await websocket.send_text(json.dumps({"event": "transcript", "text": transcript}))
            if transcript.strip():
                await session.start_turn(transcript, continuing=pending_timings)
            pending_timings = None

    async def simulate_speech(text: str) -> None:
        """Synthesizes `text` with the same Piper voice used for agent
        responses (standing in for a caller's voice, not claimed to sound
        like a distinct speaker) and feeds it through process_pcm exactly
        as if it were real microphone audio.

        Appends trailing silence before feeding it in: Piper's synthesized
        audio ends right after the last phoneme, with no real trailing
        silence, so without this UtteranceCapture's end-of-speech
        detection (sustained silence *after* speech, see turn_taking.py)
        never fires -- process_pcm runs to completion having fed every
        frame, but silently, with no transcript/response ever sent. Found
        by instrumenting every step with debug prints after ruling out an
        actual hang (an earlier version of this genuinely deadlocked
        loading Piper on a background thread post-Silero-load; fixed by
        the startup warm-up above) -- this looked identical from the
        outside (client waits forever for a message that never comes)
        but had a completely different real cause. See DECISIONS.md.
        """
        chunks = [chunk async for chunk in tts.synthesize_stream(text)]
        piper_audio = b"".join(chunks)
        pcm_16k = resample_int16(piper_audio, tts.output_sample_rate(), SAMPLE_RATE)
        trailing_silence_ms = 800  # comfortably above UtteranceCapture's default 600ms threshold
        silence_samples = int(SAMPLE_RATE * trailing_silence_ms / 1000)
        pcm_16k += b"\x00\x00" * silence_samples
        await process_pcm(pcm_16k)

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if (data := message.get("bytes")) is not None:
                await process_pcm(data)
            elif (text := message.get("text")) is not None:
                control = json.loads(text)
                if control.get("event") == "simulate_speech":
                    await simulate_speech(control["text"])
    except WebSocketDisconnect:
        pass
    finally:
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close()


# Registered last, deliberately: Starlette dispatches on the first
# path-prefix match, not by response code, so a "/" static mount
# registered *before* the /api/* routes above would shadow them
# entirely (matches every path, whether or not the file exists) --
# caught in scaffold review, not left as a live bug.
app.mount("/", StaticFiles(directory="webapp", html=True), name="webapp")
