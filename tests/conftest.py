import os

# app.core.config.Settings validates eagerly on import and requires every field.
# The eval harness only exercises LLM/prompt behavior, so seed harmless defaults
# for the settings it doesn't touch. OPENAI_API_KEY is deliberately left alone:
# the evals make real calls to the model, so a real key must come from the
# environment or a local .env file.
os.environ.setdefault("RETELL_API_KEY", "test-retell-key")
os.environ.setdefault("CLINIKO_API_KEY", "test-cliniko-key")
os.environ.setdefault("CLINIKO_SHARD", "au4")
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost/test")
os.environ.setdefault("LANGCHAIN_API_KEY", "test-langchain-key")
