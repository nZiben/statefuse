from __future__ import annotations

from typing import Any

from ..utils import digest_json_value
from ._common import (
    item_value,
    metadata_hit,
    normalized_metadata,
    record_metadata,
    record_payload,
    translated_error,
)
from .errors import (
    AdapterConfigurationError,
    AdapterProtocolError,
    AdapterSearchError,
    AdapterWriteError,
)
from .models import ExternalWriteResult, RetrievalRecord, SearchHit, SearchRequest

_INSTALL = 'Install the Letta integration with:\npip install "statefuse[letta]"'


class LettaAdapter:
    """Synchronous adapter for Letta archive passages."""

    name = "letta"

    def __init__(self, archive_id: str, client: Any | None = None, **client_options: Any) -> None:
        if not archive_id:
            raise AdapterConfigurationError("Letta archive_id is required.")
        if client is None:
            try:
                from letta_client import Letta
            except ImportError as error:
                raise AdapterConfigurationError(_INSTALL) from error
            try:
                client = Letta(**client_options)
            except Exception as error:
                raise AdapterConfigurationError(str(error)) from error
        self.archive_id = archive_id
        self.client = client
        self._known: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}

    def upsert(self, record: RetrievalRecord) -> ExternalWriteResult:
        try:
            key = (record.namespace, record.projection_id)
            payload = record_payload(record)
            existing = self._find(
                record.projection_id,
                record.namespace,
                query=record.text,
            )
            matching = next(
                (
                    item
                    for item in existing
                    if _passage_payload(item) == payload
                ),
                None,
            )
            if matching is not None:
                keep_id = _passage_id(matching)
                for item in existing:
                    if _passage_id(item) != keep_id:
                        self.client.archives.passages.delete(
                            _passage_id(item), archive_id=self.archive_id
                        )
                self._known[key] = (keep_id, payload)
                return _write_result(record, keep_id, created=False)

            known = self._known.get(key)
            if known is not None and known[1] == payload:
                return _write_result(record, known[0], created=False)

            created = self.client.archives.passages.create(
                self.archive_id,
                text=record.text,
                metadata=record_metadata(record),
                tags=_tags(record),
            )
            external_id = _passage_id(created)
            stale_ids = {_passage_id(item) for item in existing}
            if known is not None:
                stale_ids.add(known[0])
            for stale_id in stale_ids - {external_id}:
                self.client.archives.passages.delete(stale_id, archive_id=self.archive_id)
            self._known[key] = (external_id, payload)
            return _write_result(record, external_id, created=not stale_ids)
        except Exception as error:
            raise translated_error(error, AdapterWriteError) from error

    def search(self, request: SearchRequest) -> list[SearchHit]:
        try:
            results = self.client.passages.search(
                archive_id=self.archive_id,
                query=request.query,
                # Letta 0.16 filters tags after applying its result cap. Asking for
                # the maximum candidate set preserves namespace isolation as far as
                # that API allows, then StateFuse applies the requested limit.
                limit=100,
                tags=[
                    _namespace_tag(request.namespace),
                    *(
                        _filter_tag(key, value)
                        for key, value in request.filters.items()
                    ),
                ],
                tag_match_mode="all",
            )
            return [_search_hit(item) for item in results[: request.limit]]
        except Exception as error:
            raise translated_error(error, AdapterSearchError) from error

    def delete(self, projection_id: str, namespace: str) -> bool:
        try:
            existing = self._find(projection_id, namespace)
            key = (namespace, projection_id)
            external_ids = {_passage_id(item) for item in existing}
            known = self._known.pop(key, None)
            if known is not None:
                external_ids.add(known[0])
            for external_id in external_ids:
                self.client.archives.passages.delete(external_id, archive_id=self.archive_id)
            return bool(external_ids)
        except Exception as error:
            raise translated_error(error, AdapterWriteError) from error

    def healthcheck(self) -> bool:
        try:
            self.client.passages.search(archive_id=self.archive_id, limit=1)
        except Exception:
            return False
        return True

    def _find(
        self,
        projection_id: str,
        namespace: str,
        *,
        query: str | None = None,
    ) -> list[Any]:
        results = self.client.passages.search(
            archive_id=self.archive_id,
            query=query,
            limit=100,
            tags=[_namespace_tag(namespace), _projection_tag(projection_id)],
            tag_match_mode="all",
        )
        return [_passage(item) for item in results]


def _passage(item: object) -> object:
    return item_value(item, "passage", item)


def _passage_id(item: object) -> str:
    value = item_value(_passage(item), "id")
    if not isinstance(value, str) or not value:
        raise AdapterProtocolError("Letta passage is missing an ID.")
    return value


def _passage_payload(item: object) -> dict[str, Any]:
    passage = _passage(item)
    return {
        "text": item_value(passage, "text", ""),
        **normalized_metadata(item_value(passage, "metadata")),
    }


def _search_hit(item: object) -> SearchHit:
    passage = _passage(item)
    return metadata_hit(
        external_id=_passage_id(passage),
        text=item_value(passage, "text", ""),
        score=item_value(item, "score"),
        metadata=item_value(passage, "metadata"),
    )


def _tags(record: RetrievalRecord) -> list[str]:
    return [
        _namespace_tag(record.namespace),
        _projection_tag(record.projection_id),
        *(_filter_tag(key, value) for key, value in record.metadata.items()),
    ]


def _namespace_tag(namespace: str) -> str:
    return f"statefuse-namespace-{digest_json_value(namespace)}"


def _projection_tag(projection_id: str) -> str:
    return f"statefuse-projection-{digest_json_value(projection_id)}"


def _filter_tag(key: str, value: object) -> str:
    return f"statefuse-filter-{digest_json_value({'key': key, 'value': value})}"


def _write_result(
    record: RetrievalRecord, external_id: str, *, created: bool
) -> ExternalWriteResult:
    return ExternalWriteResult(
        LettaAdapter.name,
        record.projection_id,
        external_id,
        created,
        dict(record.metadata),
    )
