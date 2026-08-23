"""Fake-client unit tests for the streaming/cancellation logic (same
dependency-injection pattern as llm-cost-router's GroqProvider tests),
plus one real integration test against the actual Groq API -- a real,
tiny (~$0.0002-scale) call, same discipline as the prior project's
real-API tests. Skipped automatically if no API key is configured, so
the suite still runs clean without secrets.
"""

import os

import pytest

from voice_agent.llm import SYSTEM_PROMPT, stream_response


class _FakeDelta:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.delta = _FakeDelta(content)


class _FakeChunk:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeStream:
    def __init__(self, tokens):
        self._tokens = tokens

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for t in self._tokens:
            yield _FakeChunk(t)


class _FakeCompletions:
    def __init__(self, tokens):
        self._tokens = tokens

    async def create(self, **kwargs):
        assert kwargs["stream"] is True
        assert kwargs["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
        return _FakeStream(self._tokens)


class _FakeChat:
    def __init__(self, tokens):
        self.completions = _FakeCompletions(tokens)


class _FakeGroqClient:
    def __init__(self, tokens):
        self.chat = _FakeChat(tokens)


@pytest.mark.asyncio
async def test_stream_response_yields_tokens_in_order():
    client = _FakeGroqClient(["Got", " it", ",", " that's", " urgent."])
    result = [tok async for tok in stream_response("my payment failed", client=client)]
    assert "".join(result) == "Got it, that's urgent."


@pytest.mark.asyncio
async def test_stream_response_skips_empty_deltas():
    """Groq's real API sends a final chunk with delta.content=None -- must
    not yield that as a literal 'None' token."""
    client = _FakeGroqClient(["hello", None, " world"])
    result = [tok async for tok in stream_response("test", client=client)]
    assert result == ["hello", " world"]


@pytest.mark.asyncio
async def test_stream_response_can_be_stopped_early():
    """Simulates a barge-in cancelling generation mid-stream."""
    client = _FakeGroqClient(["a", "b", "c", "d", "e"])
    count = 0
    async for _tok in stream_response("test", client=client):
        count += 1
        if count == 2:
            break
    assert count == 2


@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="requires a real GROQ_API_KEY")
@pytest.mark.asyncio
async def test_stream_response_real_groq_call():
    """Real, tiny API call -- proves the actual streaming wire format
    matches what the fake-client tests above assume, not just that our
    own mock behaves as expected."""
    tokens = []
    async for tok in stream_response("My internet has been down for 3 days."):
        tokens.append(tok)
    full_response = "".join(tokens)
    assert len(tokens) > 1  # genuinely streamed, not one giant chunk
    assert len(full_response) > 0
