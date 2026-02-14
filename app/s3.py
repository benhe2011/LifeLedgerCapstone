"""AWS S3 operations for image storage."""
import os
import uuid
from datetime import datetime

import boto3
from botocore.exceptions import ClientError


def get_s3_client():
    """Get S3 client."""
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "us-west-2"),
    )


def get_bucket_name() -> str:
    """Get S3 bucket name from environment."""
    bucket = os.getenv("AWS_S3_BUCKET")
    if not bucket:
        raise ValueError("AWS_S3_BUCKET environment variable not set")
    return bucket


async def upload_to_s3(content: bytes, user_id: str, filename: str) -> str:
    """Upload file to S3 and return the key."""
    s3 = get_s3_client()
    bucket = get_bucket_name()

    # Generate unique key: user_id/date/uuid_filename
    date_prefix = datetime.utcnow().strftime("%Y/%m/%d")
    file_id = str(uuid.uuid4())[:8]
    ext = filename.rsplit(".", 1)[-1] if "." in filename else "jpg"
    s3_key = f"{user_id}/{date_prefix}/{file_id}.{ext}"

    s3.put_object(
        Bucket=bucket,
        Key=s3_key,
        Body=content,
        ContentType=f"image/{ext}",
    )

    return s3_key


async def download_from_s3(s3_key: str) -> bytes:
    """Download file from S3."""
    s3 = get_s3_client()
    bucket = get_bucket_name()

    response = s3.get_object(Bucket=bucket, Key=s3_key)
    return response["Body"].read()


async def generate_presigned_url(s3_key: str, expiration: int = 3600) -> str:
    """Generate a presigned URL for temporary access."""
    s3 = get_s3_client()
    bucket = get_bucket_name()

    try:
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": s3_key},
            ExpiresIn=expiration,
        )
        return url
    except ClientError:
        return ""


async def delete_from_s3(s3_key: str) -> bool:
    """Delete file from S3."""
    s3 = get_s3_client()
    bucket = get_bucket_name()

    try:
        s3.delete_object(Bucket=bucket, Key=s3_key)
        return True
    except ClientError:
        return False
