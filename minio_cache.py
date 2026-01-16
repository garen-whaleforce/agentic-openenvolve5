"""MinIO-based cache implementation to replace Redis."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from io import BytesIO
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    from minio import Minio
    from minio.error import S3Error
    HAS_MINIO = True
except ImportError:
    HAS_MINIO = False
    Minio = None  # type: ignore
    S3Error = Exception  # type: ignore

_minio_client: Optional["Minio"] = None
_cache_bucket: str = "llm-cache"


def _get_minio_config() -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Get MinIO configuration from environment."""
    endpoint = os.getenv("MINIO_ENDPOINT", "").replace("https://", "").replace("http://", "")
    access_key = os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("MINIO_SECRET_KEY")
    return endpoint, access_key, secret_key


def _create_minio_client() -> Optional["Minio"]:
    """Create MinIO client."""
    if not HAS_MINIO:
        logger.warning("minio library is not installed; MinIO cache disabled")
        return None

    endpoint, access_key, secret_key = _get_minio_config()

    if not endpoint or not access_key or not secret_key:
        logger.info("MinIO configuration incomplete; MinIO cache disabled")
        return None

    try:
        # Check if endpoint uses HTTPS
        raw_endpoint = os.getenv("MINIO_ENDPOINT", "")
        secure = raw_endpoint.startswith("https://")

        client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )

        # Ensure cache bucket exists
        global _cache_bucket
        _cache_bucket = os.getenv("MINIO_CACHE_BUCKET", "llm-cache")

        if not client.bucket_exists(_cache_bucket):
            client.make_bucket(_cache_bucket)
            logger.info(f"Created MinIO cache bucket: {_cache_bucket}")

        logger.info(f"MinIO client initialized, using bucket: {_cache_bucket}")
        return client
    except Exception as exc:
        logger.warning(f"Failed to initialize MinIO client: {exc}")
        return None


def get_minio_client(force_reconnect: bool = False) -> Optional["Minio"]:
    """Get or create MinIO client singleton."""
    global _minio_client

    if not HAS_MINIO:
        return None

    if force_reconnect:
        _minio_client = None

    if _minio_client is None:
        _minio_client = _create_minio_client()

    return _minio_client


def _key_to_object_name(key: str) -> str:
    """Convert cache key to MinIO object name (with hash for safety)."""
    # Use hash to avoid special characters in object names
    key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
    # Keep some readable part of the key
    safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in key[:50])
    return f"cache/{safe_key}_{key_hash}.json"


async def cache_get_json(key: str) -> Optional[Any]:
    """
    Get cached JSON value from MinIO.

    Returns None if not found, expired, or error.
    """
    client = get_minio_client()
    if client is None:
        return None

    object_name = _key_to_object_name(key)

    try:
        response = client.get_object(_cache_bucket, object_name)
        data = json.loads(response.read().decode("utf-8"))
        response.close()
        response.release_conn()

        # Check TTL expiration
        expires_at = data.get("_expires_at", 0)
        if expires_at > 0 and time.time() > expires_at:
            # Expired - delete and return None
            try:
                client.remove_object(_cache_bucket, object_name)
            except Exception:
                pass
            return None

        return data.get("value")
    except S3Error as exc:
        if exc.code == "NoSuchKey":
            return None  # Cache miss
        logger.warning(f"MinIO GET error for key {key}: {exc}")
        return None
    except Exception as exc:
        logger.warning(f"MinIO GET failed for key {key}: {exc}")
        return None


async def cache_set_json(key: str, value: Any, ttl_seconds: int) -> None:
    """
    Store JSON value in MinIO with TTL metadata.

    TTL is stored in the object itself and checked on read.
    """
    client = get_minio_client()
    if client is None:
        return

    object_name = _key_to_object_name(key)

    try:
        # Wrap value with expiration metadata
        data = {
            "value": value,
            "_expires_at": time.time() + ttl_seconds if ttl_seconds > 0 else 0,
            "_created_at": time.time(),
        }

        payload = json.dumps(data).encode("utf-8")
        payload_stream = BytesIO(payload)

        client.put_object(
            _cache_bucket,
            object_name,
            payload_stream,
            length=len(payload),
            content_type="application/json",
        )
    except Exception as exc:
        logger.warning(f"MinIO SET failed for key {key}: {exc}")


async def cache_delete(key: str) -> None:
    """Delete a cache entry."""
    client = get_minio_client()
    if client is None:
        return

    object_name = _key_to_object_name(key)

    try:
        client.remove_object(_cache_bucket, object_name)
    except Exception as exc:
        logger.warning(f"MinIO DELETE failed for key {key}: {exc}")


def check_minio_connection() -> bool:
    """Check if MinIO is accessible."""
    client = get_minio_client()
    if client is None:
        return False

    try:
        # Try to list buckets as a health check
        client.list_buckets()
        return True
    except Exception as exc:
        logger.warning(f"MinIO health check failed: {exc}")
        return False
