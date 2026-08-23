"""Single source of truth for config, read from env vars / .env
(pydantic-settings) -- every other module imports `settings` from here
instead of reading os.environ directly. See DECISIONS.md for why each
default was chosen.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM provider
    groq_api_key: str = ""

    # Barge-in / VAD
    vad_backend: str = "silero"  # "silero" | "webrtc"
    vad_speech_threshold_ms: int = 250

    # STT (only invoked after a barge-in fires, not on the hot path)
    whisper_model_size: str = "base"

    # TTS
    piper_model_path: str = "models/piper/en_US-lessac-medium.onnx"

    # App
    log_level: str = "INFO"


settings = Settings()
