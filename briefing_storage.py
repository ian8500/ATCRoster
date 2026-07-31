"""Private, provider-neutral storage for controlled briefing documents."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


class BriefingStorageError(RuntimeError):
    """Raised when controlled-document storage is unavailable."""


class BriefingStorage:
    provider_name = "unknown"

    def put(self, key: str, content: bytes, content_type: str, sha256: str) -> None:
        raise NotImplementedError

    def get(self, key: str) -> bytes:
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        raise NotImplementedError

    def health(self) -> tuple[bool, str]:
        raise NotImplementedError


@dataclass
class LocalBriefingStorage(BriefingStorage):
    root: Path
    provider_name = "local"

    def _path(self, key: str) -> Path:
        candidate = (self.root / key).resolve()
        root = self.root.resolve()
        if root not in candidate.parents:
            raise BriefingStorageError("Invalid briefing object key.")
        return candidate

    def put(self, key: str, content: bytes, content_type: str, sha256: str) -> None:
        path = self._path(key)
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            with path.open("xb") as handle:
                handle.write(content)
            path.chmod(0o600)
        except OSError as exc:
            raise BriefingStorageError(
                "Private briefing storage could not save the document."
            ) from exc

    def get(self, key: str) -> bytes:
        try:
            return self._path(key).read_bytes()
        except OSError as exc:
            raise BriefingStorageError(
                "The briefing document is unavailable."
            ) from exc

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def health(self) -> tuple[bool, str]:
        try:
            self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
            return True, "Local private storage is available"
        except OSError:
            return False, "Local private storage is unavailable"


class S3BriefingStorage(BriefingStorage):
    provider_name = "s3"

    def __init__(
        self, *, bucket: str, endpoint: str, region: str,
        access_key: str, secret_key: str,
    ):
        if not all((bucket, endpoint, access_key, secret_key)):
            raise BriefingStorageError(
                "S3 briefing storage configuration is incomplete."
            )
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:
            raise BriefingStorageError(
                "S3 briefing storage support is not installed."
            ) from exc
        self.bucket = bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            region_name=region or "auto",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": "virtual"},
                connect_timeout=3,
                read_timeout=10,
                retries={"max_attempts": 2, "mode": "standard"},
            ),
        )

    def put(self, key: str, content: bytes, content_type: str, sha256: str) -> None:
        try:
            self.client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=content,
                ContentType=content_type,
                Metadata={"sha256": sha256},
            )
            response = self.client.head_object(Bucket=self.bucket, Key=key)
        except Exception as exc:
            raise BriefingStorageError(
                "Private briefing storage could not save the document."
            ) from exc
        stored_digest = (response.get("Metadata") or {}).get("sha256")
        if stored_digest != sha256:
            raise BriefingStorageError(
                "The stored document failed checksum verification."
            )

    def get(self, key: str) -> bytes:
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            return response["Body"].read()
        except Exception as exc:
            raise BriefingStorageError(
                "The briefing document is unavailable."
            ) from exc

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False

    def health(self) -> tuple[bool, str]:
        try:
            self.client.head_bucket(Bucket=self.bucket)
            return True, "Railway private bucket is available"
        except Exception:
            return False, "Railway private bucket is unavailable"


def configured_briefing_storage(instance_path: str) -> BriefingStorage:
    provider = os.environ.get("BRIEFING_STORAGE_PROVIDER", "local").strip().lower()
    production = (
        os.environ.get("ATCROSTER_ENVIRONMENT", "development").strip().lower()
        == "production"
    )
    if provider in {"local", "mounted"}:
        if production:
            root = os.environ.get(
                "ATCROSTER_BRIEFING_DURABLE_DIR", ""
            ).strip()
            if not root:
                raise BriefingStorageError(
                    "Production local briefing storage requires an explicit "
                    "ATCROSTER_BRIEFING_DURABLE_DIR mounted on durable storage."
                )
            if not Path(root).is_absolute():
                raise BriefingStorageError(
                    "ATCROSTER_BRIEFING_DURABLE_DIR must be an absolute path."
                )
        else:
            root = os.environ.get(
                "ATCROSTER_BRIEFING_UPLOAD_DIR",
                os.path.join(instance_path, "briefing_uploads"),
            )
        return LocalBriefingStorage(Path(root))
    if provider == "s3":
        return S3BriefingStorage(
            bucket=os.environ.get("BRIEFING_STORAGE_BUCKET", ""),
            endpoint=os.environ.get("BRIEFING_STORAGE_ENDPOINT", ""),
            region=os.environ.get("BRIEFING_STORAGE_REGION", "auto"),
            access_key=os.environ.get("BRIEFING_STORAGE_ACCESS_KEY", ""),
            secret_key=os.environ.get("BRIEFING_STORAGE_SECRET_KEY", ""),
        )
    raise BriefingStorageError("Unknown briefing storage provider.")
