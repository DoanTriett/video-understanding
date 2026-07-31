from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    app_name: str = "Video Understanding API"
    debug: bool = True

    # Redis job queue
    redis_url: str = "redis://localhost:6379/0"

    # Qdrant vector database
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # Upload settings
    upload_dir: str = "./uploads"
    max_file_size_mb: int = 500

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4.1-mini"

    # Hugging Face
    huggingface_token: str = Field(default="", env="HUGGINGFACE_TOKEN")

    # PostgreSQL
    POSTGRES_URL: str = Field(default="", env="POSTGRES_URL")

    device: str = "cuda"

    minio_endpoint: str = Field(default="", env="MINIO_ENDPOINT")
    minio_access_key: str = Field(default="", env="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(default="", env="MINIO_SECRET_KEY")
    minio_bucket: str = Field(default="videos", env="MINIO_BUCKET")

    # Rate limiting (slowapi, per IP).
    # Override via env: RATE_LIMIT_ASK="20/minute", RATE_LIMIT_UPLOAD="10/minute".
    # /ask is expensive (Qdrant + LLM); 10/min is a safe default for a single-user
    # local deployment. Raise if serving multiple users behind a proxy.
    rate_limit_ask: str = "10/minute"
    rate_limit_upload: str = "5/minute"

    # Set TESTING=true to disable rate limiting in the test suite.
    testing: bool = False

    @field_validator("debug", mode="before")
    @classmethod
    def _parse_debug(cls, value):
        if isinstance(value, str) and value.lower() in {"release", "prod", "production"}:
            return False
        return value

    class Config:
        env_file = ".env"
        extra = "ignore"


# Shared settings instance
settings = Settings()
