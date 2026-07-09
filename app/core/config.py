from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central app configuration. Pydantic reads these from environment
    variables (or a .env file) and validates types at startup.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    DATABASE_URL: str
    REDIS_URL: str

    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    ANTHROPIC_API_KEY: str = ""
    GROQ_API_KEY: str = ""

    ENVIRONMENT: str = "development"


settings = Settings()