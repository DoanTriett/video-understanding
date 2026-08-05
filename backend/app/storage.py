from io import BytesIO
from typing import Optional

import boto3
from botocore.client import Config

from app.config import settings


def _endpoint_url() -> Optional[str]:
    """Return S3 endpoint URL, or None for real AWS S3.

    - Empty MINIO_ENDPOINT  → None  (boto3 uses standard AWS endpoints)
    - host:port             → http://host:port   (MinIO, Cloudflare R2 local, etc.)
    - http(s)://...         → returned as-is
    """
    endpoint = settings.minio_endpoint.strip()
    if not endpoint:
        return None
    if endpoint.startswith(("http://", "https://")):
        return endpoint
    return f"http://{endpoint}"


def get_s3_client():
    endpoint = _endpoint_url()
    kwargs: dict = {"config": Config(signature_version="s3v4")}
    # Only pass credentials explicitly when configured via MINIO_ACCESS_KEY /
    # MINIO_SECRET_KEY. When those are empty, boto3 falls back to its standard
    # credential chain: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY env vars,
    # ~/.aws/credentials, IAM role, etc.
    if settings.minio_access_key:
        kwargs["aws_access_key_id"] = settings.minio_access_key
    if settings.minio_secret_key:
        kwargs["aws_secret_access_key"] = settings.minio_secret_key
    if endpoint is not None:
        kwargs["endpoint_url"] = endpoint
    if settings.aws_region:
        kwargs["region_name"] = settings.aws_region
    return boto3.client("s3", **kwargs)


def upload_fileobj(file_obj, object_key: str) -> None:
    client = get_s3_client()
    client.upload_fileobj(file_obj, settings.minio_bucket, object_key)


def upload_bytes(contents: bytes, object_key: str) -> None:
    upload_fileobj(BytesIO(contents), object_key)


def download_to_path(object_key: str, local_path: str) -> None:
    client = get_s3_client()
    client.download_file(settings.minio_bucket, object_key, local_path)


def upload_file(local_path: str, object_key: str) -> None:
    client = get_s3_client()
    client.upload_file(local_path, settings.minio_bucket, object_key)


def presigned_url(object_key: str, expires: int = 3600) -> str:
    client = get_s3_client()
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.minio_bucket, "Key": object_key},
        ExpiresIn=expires,
    )
