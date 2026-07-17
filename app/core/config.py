from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    GROQ_API_KEY: str
    GEMINI_API_KEY: str | None = None
    RETELL_API_KEY: str
    CLINIKO_API_KEY: str
    CLINIKO_SHARD: str
    DATABASE_URL: str
    LANGCHAIN_API_KEY: str

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
