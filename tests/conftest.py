"""Loads .env into the real process environment for the whole test
session. Needed because pydantic-settings' Settings only reads .env
lazily when a Settings() instance is built -- it never populates
os.environ itself -- but tests like test_llm.py's real-Groq-call test
gate on os.getenv("GROQ_API_KEY") directly (skipif needs to evaluate at
collection time, before any Settings() would be constructed), and the
browser-facing app needs the same real env either way.
"""

from dotenv import load_dotenv

load_dotenv()
