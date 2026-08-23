from voice_agent.config import Settings


def test_defaults_load_without_env_vars():
    s = Settings(_env_file=None)
    assert s.vad_backend == "silero"
    assert s.vad_speech_threshold_ms == 250
    assert s.whisper_model_size == "base"


def test_env_override():
    s = Settings(_env_file=None, vad_backend="webrtc", groq_api_key="test-key")
    assert s.vad_backend == "webrtc"
    assert s.groq_api_key == "test-key"
