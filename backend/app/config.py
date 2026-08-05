from pydantic import AliasChoices, Field, field_validator
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

    # CORS — comma-separated exact origins + optional regex for Vercel previews.
    cors_origins: str = Field(
        default=(
            "http://localhost:3000,"
            "http://127.0.0.1:3000,"
            "https://video-understanding.vercel.app"
        ),
        env="CORS_ORIGINS",
    )
    cors_allow_origin_regex: str = Field(
        default=(
            r"https://video-understanding-[a-z0-9-]+" r"-doantriet2005-8192s-projects\.vercel\.app"
        ),
        env="CORS_ALLOW_ORIGIN_REGEX",
    )

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4.1-mini"

    # Hugging Face
    huggingface_token: str = Field(default="", env="HUGGINGFACE_TOKEN")

    # PostgreSQL (Neon often issues postgres:// — normalized below)
    POSTGRES_URL: str = Field(default="", env="POSTGRES_URL")

    device: str = "cuda"

    @field_validator("POSTGRES_URL", mode="before")
    @classmethod
    def _normalize_postgres_url(cls, value: object) -> object:
        if isinstance(value, str) and value.startswith("postgres://"):
            return "postgresql://" + value[len("postgres://") :]
        return value

    # S3-compatible object storage.
    # Leave MINIO_ENDPOINT empty to use real AWS S3 (boto3 default endpoints).
    # Set to host:port or https://... for MinIO / Cloudflare R2 / etc.
    minio_endpoint: str = Field(default="", env="MINIO_ENDPOINT")
    minio_access_key: str = Field(default="", env="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(default="", env="MINIO_SECRET_KEY")
    # Accept MINIO_BUCKET (legacy local) or S3_BUCKET (standard AWS naming).
    minio_bucket: str = Field(
        default="videos",
        validation_alias=AliasChoices("MINIO_BUCKET", "S3_BUCKET"),
    )
    aws_region: str = Field(default="", env="AWS_REGION")

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
