from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO8601 with Z suffix."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def parse_utc_iso(value: str) -> datetime:
    """Parse an ISO8601 timestamp and normalize it to UTC."""
    if value.endswith("Z"):
        value = f"{value[:-1]}+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def canonical_json_dumps(value: Any) -> str:
    """Serialize JSON in a canonical, deterministic format."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_json_loads(value: str) -> Any:
    return json.loads(value)


def sha256_hexdigest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def digest_json_value(value: Any) -> str:
    return sha256_hexdigest(canonical_json_dumps(value).encode("utf-8"))


def digest_content(value: Any) -> str:
    if isinstance(value, bytes):
        return sha256_hexdigest(value)
    if isinstance(value, str):
        return sha256_hexdigest(value.encode("utf-8"))
    return digest_json_value(value)


def new_uuid() -> str:
    return str(uuid.uuid4())


def content_addressed_op_id(
    *,
    op_type: str,
    replica_id: str,
    timestamp: str,
    payload: Any,
) -> str:
    digest = digest_json_value(
        {
            "op_type": op_type,
            "replica_id": replica_id,
            "timestamp": timestamp,
            "payload": payload,
        }
    )
    return f"sha256:{digest}"
