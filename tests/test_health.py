from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from server import app


async def test_health_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["vad_backend"] in ("silero", "webrtc")


def test_ws_endpoint_accepts_and_disconnects_cleanly():
    """Basic connectivity smoke test -- the real handler's actual
    protocol (ready message, audio framing, barge-in/transcript events)
    is tested in test_server_ws.py; this just confirms connecting and
    then disconnecting without sending anything doesn't raise or hang."""
    client = TestClient(app)
    with client.websocket_connect("/api/ws"):
        pass  # connection accepted without raising is the assertion


def test_static_index_served_and_api_routes_not_shadowed():
    """Regression test for the mount-ordering bug caught during
    scaffold: a "/" static mount registered before the /api/* routes
    would shadow them entirely."""
    client = TestClient(app)
    assert client.get("/api/health").status_code == 200
    assert "Voice Triage Agent" in client.get("/").text
