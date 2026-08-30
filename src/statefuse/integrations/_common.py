from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..model import JSONValue
from .errors import (
    AdapterAuthenticationError,
    AdapterError,
    AdapterProtocolError,
    AdapterUnavailableError,
)
from .models import RetrievalRecord, SearchHit

_PROJECTION_ID = "statefuse_projection_id"
_NAMESPACE = "statefuse_namespace"
_CLAIM_IDS = "statefuse_claim_ids"
_CONFLICT_IDS = "statefuse_conflict_ids"
_VERSION = "statefuse_projection_version"
_METADATA = "statefuse_metadata"


def record_metadata(record: RetrievalRecord) -> dict[str, JSONValue]:
    return {
        _PROJECTION_ID: record.projection_id,
        _NAMESPACE: record.namespace,
        _CLAIM_IDS: list(record.claim_ids),
        _CONFLICT_IDS: list(record.conflict_ids),
        _VERSION: record.projection_version,
        _METADATA: dict(record.metadata),
    }


def record_payload(record: RetrievalRecord) -> dict[str, JSONValue]:
    return {"text": record.text, **record_metadata(record)}


def normalized_metadata(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def item_value(item: object, key: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def metadata_hit(
    *,
    external_id: object,
    text: object,
    score: object,
    metadata: object,
) -> SearchHit:
    if not isinstance(external_id, str) or not external_id:
        raise AdapterProtocolError("Repository search result is missing an external ID.")
    values = normalized_metadata(metadata)
    return SearchHit(
        external_id=external_id,
        projection_id=_string_or_none(values.get(_PROJECTION_ID)),
        text=text if isinstance(text, str) else "",
        score=float(score) if isinstance(score, int | float) else None,
        claim_ids=_string_tuple(values.get(_CLAIM_IDS)),
        conflict_ids=_string_tuple(values.get(_CONFLICT_IDS)),
        metadata=normalized_metadata(values.get(_METADATA)),
    )


def payload_matches_record(payload: object, record: RetrievalRecord) -> bool:
    values = normalized_metadata(payload)
    return values == record_payload(record)


def metadata_matches_filters(metadata: object, filters: Mapping[str, JSONValue]) -> bool:
    values = normalized_metadata(metadata)
    return all(values.get(key) == expected for key, expected in filters.items())


def protocol_error(message: str) -> AdapterProtocolError:
    return AdapterProtocolError(message)


def translated_error(error: Exception, fallback: type[AdapterError]) -> AdapterError:
    if isinstance(error, AdapterError):
        return error
    status = getattr(error, "status_code", None)
    response = getattr(error, "response", None)
    if status is None and response is not None:
        status = getattr(response, "status_code", None)
    name = type(error).__name__.casefold()
    message = str(error) or type(error).__name__
    if status in {401, 403} or any(token in name for token in ("auth", "permission")):
        return AdapterAuthenticationError(message)
    if (
        isinstance(error, ConnectionError | TimeoutError)
        or status == 429
        or isinstance(status, int) and status >= 500
        or any(token in name for token in ("connection", "timeout", "unavailable"))
    ):
        return AdapterUnavailableError(message)
    return fallback(message)


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item)
