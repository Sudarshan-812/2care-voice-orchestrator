import os

from dotenv import load_dotenv

# Load the real .env first (if present) so a real GROQ_API_KEY reaches
# Settings() for live evals. load_dotenv() never overwrites variables
# already present in os.environ.
load_dotenv()

# app.core.config.Settings validates eagerly on import and requires every field.
# The eval harness only exercises LLM/prompt behavior, so seed harmless defaults
# for the settings it doesn't touch. GROQ_API_KEY only gets a dummy fallback if
# no real key was found above, so plain `pytest` collection never crashes.
os.environ.setdefault("GROQ_API_KEY", "dummy_groq_key")
os.environ.setdefault("RETELL_API_KEY", "test-retell-key")
os.environ.setdefault("CLINIKO_API_KEY", "test-cliniko-key")
os.environ.setdefault("CLINIKO_SHARD", "au4")
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost/test")
os.environ.setdefault("LANGCHAIN_API_KEY", "test-langchain-key")
