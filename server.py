"""Vercel entrypoint (must be named app.py/index.py/server.py/main.py/
wsgi.py/asgi.py at the project root, defining a top-level `app` -- see
DECISIONS.md "Deploy target"). Local dev: `uvicorn server:app --reload`.

WebSocket protocol (see public/client.js for the browser side):
  Client -> server: raw 16-bit PCM mono audio at 16kHz, binary frames,
    any chunk size (server-side buffers into whatever frame size the
    active VAD backend needs -- see turn_taking.py/vad.py).
  Server -> client: binary frames = raw 16-bit PCM audio for playback, at
    `output_sample_rate` (Piper's native rate, sent in the initial
    "ready" message -- never assumed by the client).
    Text frames = JSON events: {"event": "ready", ...},
    {"event": "listening"}, {"event": "transcript", "text": ...},
    {"event": "barge_in"}.
"""

import asyncio
import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from voice_agent import stt, tts
from voice_agent.config import settings
from voice_agent.pipeline import CallSession
from voice_agent.turn_taking import UtteranceCapture
from voice_agent.vad import SAMPLE_RATE

app = FastAPI(title="Voice Triage Agent")


@app.get("/api/health")
async def health():
    return {"status": "ok", "vad_backend": settings.vad_backend}


@app.websocket("/api/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()

    async def send_audio(chunk: bytes) -> None:
        await websocket.send_bytes(chunk)

    session = CallSession(send_audio=send_audio)
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

    try:
        while True:
            message = await websocket.receive_bytes()
            mic_buffer.extend(message)

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
    except WebSocketDisconnect:
        pass


# Registered last, deliberately: Starlette dispatches on the first
# path-prefix match, not by response code, so a "/" static mount
# registered *before* the /api/* routes above would shadow them
# entirely (matches every path, whether or not the file exists) --
# caught in scaffold review, not left as a live bug.
app.mount("/", StaticFiles(directory="public", html=True), name="public")
