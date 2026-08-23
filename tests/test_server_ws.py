"""Tests for /api/ws's own glue logic (frame chunking, state branching
between "listening" and "speaking", event messages) -- CallSession's
cancel/restart correctness is test_pipeline.py's concern, UtteranceCapture's
end-of-speech logic is test_turn_taking.py's, so both are faked here with
controllable behavior, same isolation pattern used throughout this project.

Uses FastAPI's synchronous TestClient websocket support (not AsyncClient --
Starlette's WebSocketTestSession runs the ASGI app on a background thread
and exposes a sync interface), so these are regular `def test_...`
functions, not `async def`.
"""

import json

from fastapi.testclient import TestClient

import server


class _FakeTimings:
    pass


class _FakeCallSession:
    """Records calls; setting `.armed_barge_in = True` makes the *next*
    feed_mic_frame call report a barge-in."""

    #: shared across instances within a test so the test can reach in and
    #: arm/inspect behavior without needing to intercept construction.
    instances: list = []

    def __init__(self, send_audio, vad_backend=None):
        self.send_audio = send_audio
        self.is_speaking = False
        self.armed_barge_in = False
        self.start_turn_calls = []
        self.feed_mic_frame_calls = 0
        _FakeCallSession.instances.append(self)

    async def feed_mic_frame(self, frame):
        self.feed_mic_frame_calls += 1
        if self.armed_barge_in:
            self.armed_barge_in = False
            self.is_speaking = False
            return _FakeTimings()
        return None

    async def start_turn(self, transcript, continuing=None):
        self.start_turn_calls.append((transcript, continuing))
        self.is_speaking = True
        await self.send_audio(b"\x00\x01" * 10)  # simulate one TTS chunk going out
        self.is_speaking = False


class _FakeUtteranceCapture:
    """Setting the *class* attribute `.next_feed_completes = True`
    makes the next `feed()` call (on any instance) report the utterance
    as complete -- simpler than intercepting which instance the handler
    constructs internally, and sufficient since each test only has one
    handler connection alive at a time."""

    #: recorded so tests can inspect the single instance the handler
    #: constructs per connection (server.py creates one UtteranceCapture
    #: at connection start and reuses it via .reset(), never
    #: reconstructs it -- so "the instance" is unambiguous per test).
    instances: list = []

    frame_size_samples = 4  # tiny -- keeps test messages small
    next_feed_completes = False

    def __init__(self, vad_backend=None):
        self.fed_frames = []
        self._audio = b""
        _FakeUtteranceCapture.instances.append(self)

    def feed(self, frame):
        self.fed_frames.append(frame)
        self._audio += frame
        if _FakeUtteranceCapture.next_feed_completes:
            _FakeUtteranceCapture.next_feed_completes = False
            return True
        return False

    @property
    def audio(self):
        return self._audio

    def reset(self):
        self._audio = b""


def _install_fakes(monkeypatch):
    _FakeCallSession.instances = []
    _FakeUtteranceCapture.instances = []
    _FakeUtteranceCapture.next_feed_completes = False
    monkeypatch.setattr(server, "CallSession", _FakeCallSession)
    monkeypatch.setattr(server, "UtteranceCapture", _FakeUtteranceCapture)
    monkeypatch.setattr(server.tts, "output_sample_rate", lambda: 22050)


def test_ready_message_has_expected_fields(monkeypatch):
    _install_fakes(monkeypatch)
    with TestClient(server.app).websocket_connect("/api/ws") as ws:
        ready = json.loads(ws.receive_text())
        assert ready["event"] == "ready"
        assert ready["sample_rate_in"] == 16000
        assert ready["sample_rate_out"] == 22050
        assert ready["frame_size_samples"] == 4


def test_utterance_completes_transcribes_and_starts_a_turn(monkeypatch):
    _install_fakes(monkeypatch)
    monkeypatch.setattr(server.stt, "transcribe", lambda audio: "my internet is down")

    with TestClient(server.app).websocket_connect("/api/ws") as ws:
        ws.receive_text()  # ready

        _FakeUtteranceCapture.next_feed_completes = True
        ws.send_bytes(b"\x00\x00" * 4)  # exactly one 4-sample (8-byte) frame

        transcript_msg = json.loads(ws.receive_text())
        assert transcript_msg == {"event": "transcript", "text": "my internet is down"}

        audio_chunk = ws.receive_bytes()
        assert audio_chunk == b"\x00\x01" * 10

        session = _FakeCallSession.instances[-1]
        assert session.start_turn_calls == [("my internet is down", None)]


def test_empty_transcript_does_not_start_a_turn(monkeypatch):
    """A silent/unintelligible utterance shouldn't make the agent
    respond to nothing."""
    _install_fakes(monkeypatch)
    monkeypatch.setattr(server.stt, "transcribe", lambda audio: "   ")

    with TestClient(server.app).websocket_connect("/api/ws") as ws:
        ws.receive_text()  # ready

        _FakeUtteranceCapture.next_feed_completes = True
        ws.send_bytes(b"\x00\x00" * 4)

        transcript_msg = json.loads(ws.receive_text())
        assert transcript_msg == {"event": "transcript", "text": "   "}

        session = _FakeCallSession.instances[-1]
        assert session.start_turn_calls == []


def test_barge_in_sends_event_and_feeds_triggering_frame_to_fresh_capture(monkeypatch):
    _install_fakes(monkeypatch)

    with TestClient(server.app).websocket_connect("/api/ws") as ws:
        ws.receive_text()  # ready
        session = _FakeCallSession.instances[-1]
        session.is_speaking = True
        session.armed_barge_in = True

        trigger_frame = b"\x01\x02\x03\x04\x05\x06\x07\x08"  # 4 samples (8 bytes) -- matches frame_size_samples
        ws.send_bytes(trigger_frame)

        barge_in_msg = json.loads(ws.receive_text())
        assert barge_in_msg == {"event": "barge_in"}
        assert session.feed_mic_frame_calls == 1
        # The frame that triggered the barge-in is real caller speech --
        # must be fed into the (reset) capture, not discarded.
        capture = _FakeUtteranceCapture.instances[-1]
        assert capture.fed_frames == [trigger_frame]
        assert capture.audio == trigger_frame


def test_multiple_small_binary_messages_are_reassembled_into_frames(monkeypatch):
    """Client chunk size is independent of the VAD's required frame
    size -- the handler must buffer across receive_bytes() calls, not
    assume one message == one frame."""
    _install_fakes(monkeypatch)
    monkeypatch.setattr(server.stt, "transcribe", lambda audio: "test")

    with TestClient(server.app).websocket_connect("/api/ws") as ws:
        ws.receive_text()  # ready

        # frame_size_samples=4 -> 8 bytes/frame. Send it split across two
        # small messages, neither of which alone is a full frame.
        ws.send_bytes(b"\x01\x02\x03")
        ws.send_bytes(b"\x04\x05\x06\x07\x08")

        # No transcript/barge-in yet -- not a complete utterance, just
        # confirming no crash/desync from the split. Arm completion and
        # send one more full frame to confirm the buffer's state is sane.
        _FakeUtteranceCapture.next_feed_completes = True
        ws.send_bytes(b"\x00\x00" * 4)

        transcript_msg = json.loads(ws.receive_text())
        assert transcript_msg["event"] == "transcript"
