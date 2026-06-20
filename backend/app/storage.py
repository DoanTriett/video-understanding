from io import BytesIO

import boto3
from botocore.client import Config

from app.config import settings


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=f"http://{settings.minio_endpoint}",
        aws_access_key_id=settings.minio_access_key,
        aws_secret_access_key=settings.minio_secret_key,
        config=Config(signature_version="s3v4"),
    )


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
