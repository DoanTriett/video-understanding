from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    app_name: str = "Video Understanding API"
    debug: bool = True

    # Redis (dùng cho job queue)
    redis_url: str = "redis://localhost:6379/0"

    # Qdrant (vector database)
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # Upload settings
    upload_dir: str = "./uploads"
    max_file_size_mb: int = 500

    # OpenAI
    openai_api_key: str = ""

    # Hugging Face
    huggingface_token: str = Field(default="", env="HUGGINGFACE_TOKEN")

    # PostgreSQL
    POSTGRES_URL: str = Field(default="", env="POSTGRES_URL")

    device: str = "cuda"

    minio_endpoint: str = Field(default="", env="MINIO_ENDPOINT")
    minio_access_key: str = Field(default="", env="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(default="", env="MINIO_SECRET_KEY")
    minio_bucket: str = Field(default="videos", env="MINIO_BUCKET")

    class Config:
        env_file = ".env"  # Đọc từ file .env


# Tạo 1 instance dùng chung toàn app
settings = Settings()
