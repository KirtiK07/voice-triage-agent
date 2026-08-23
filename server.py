"""Vercel entrypoint (must be named app.py/index.py/server.py/main.py/
wsgi.py/asgi.py at the project root, defining a top-level `app` -- see
DECISIONS.md "Deploy target"). Local dev: `uvicorn server:app --reload`.

WebSocket route is a scaffold-stage stub (accepts, then closes) --
real barge-in orchestration is wired up at build stage via
voice_agent.pipeline.CallSession.
"""

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles

from voice_agent.config import settings

app = FastAPI(title="Voice Triage Agent")


@app.get("/api/health")
async def health():
    return {"status": "ok", "vad_backend": settings.vad_backend}


@app.websocket("/api/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    # Client needs reconnect-with-backoff logic for Vercel Hobby's 300s
    # hard connection limit -- see DECISIONS.md "Deploy target".
    # Real barge-in pipeline wiring happens at build stage.
    await websocket.close()


# Registered last, deliberately: Starlette dispatches on the first
# path-prefix match, not by response code, so a "/" static mount
# registered *before* the /api/* routes above would shadow them
# entirely (matches every path, whether or not the file exists) --
# caught in scaffold review, not left as a live bug.
app.mount("/", StaticFiles(directory="public", html=True), name="public")
