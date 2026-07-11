"""Unit tests for app.storage — mock boto3 S3 client, no MinIO."""

from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from app import storage


@pytest.fixture
def s3_client():
    client = MagicMock()
    with patch.object(storage, "get_s3_client", return_value=client):
        yield client


def test_upload_bytes_calls_upload_fileobj(s3_client):
    storage.upload_bytes(b"video-data", "vid/source.mp4")

    s3_client.upload_fileobj.assert_called_once()
    file_obj, bucket, key = s3_client.upload_fileobj.call_args[0]
    assert isinstance(file_obj, BytesIO)
    assert file_obj.read() == b"video-data"
    assert bucket == storage.settings.minio_bucket
    assert key == "vid/source.mp4"


def test_upload_fileobj_happy_path(s3_client):
    payload = BytesIO(b"abc")
    storage.upload_fileobj(payload, "obj/key")

    s3_client.upload_fileobj.assert_called_once_with(
        payload, storage.settings.minio_bucket, "obj/key"
    )


def test_upload_bytes_propagates_client_error(s3_client):
    s3_client.upload_fileobj.side_effect = ClientError(
        {"Error": {"Code": "AccessDenied", "Message": "denied"}},
        "PutObject",
    )

    with pytest.raises(ClientError):
        storage.upload_bytes(b"x", "key")


def test_download_to_path_happy_path(s3_client):
    storage.download_to_path("vid/transcript.json", "/tmp/transcript.json")

    s3_client.download_file.assert_called_once_with(
        storage.settings.minio_bucket,
        "vid/transcript.json",
        "/tmp/transcript.json",
    )


def test_download_to_path_nosuchkey(s3_client):
    s3_client.download_file.side_effect = ClientError(
        {"Error": {"Code": "NoSuchKey", "Message": "missing"}},
        "GetObject",
    )

    with pytest.raises(ClientError) as exc_info:
        storage.download_to_path("missing.json", "/tmp/x.json")

    assert exc_info.value.response["Error"]["Code"] == "NoSuchKey"


def test_upload_file_happy_path(s3_client):
    storage.upload_file("/tmp/local.mp4", "vid/source.mp4")

    s3_client.upload_file.assert_called_once_with(
        "/tmp/local.mp4",
        storage.settings.minio_bucket,
        "vid/source.mp4",
    )


def test_presigned_url_happy_path(s3_client):
    s3_client.generate_presigned_url.return_value = "https://minio.example/signed"

    url = storage.presigned_url("vid/source.mp4", expires=7200)

    assert url == "https://minio.example/signed"
    s3_client.generate_presigned_url.assert_called_once_with(
        "get_object",
        Params={"Bucket": storage.settings.minio_bucket, "Key": "vid/source.mp4"},
        ExpiresIn=7200,
    )


def test_get_s3_client_uses_boto3():
    with patch("app.storage.boto3.client") as mock_boto_client:
        mock_boto_client.return_value = MagicMock()
        client = storage.get_s3_client()

    assert client is mock_boto_client.return_value
    mock_boto_client.assert_called_once()
    call_kwargs = mock_boto_client.call_args.kwargs
    assert call_kwargs["endpoint_url"] == f"http://{storage.settings.minio_endpoint}"
    assert call_kwargs["aws_access_key_id"] == storage.settings.minio_access_key
