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

_INSTALL = 'Install the Mem0 integration with:\npip install "statefuse[mem0]"'


class Mem0Adapter:
    """Synchronous adapter for Mem0 OSS ``Memory`` clients."""

    name = "mem0"

    def __init__(self, client: Any | None = None, *, config: dict[str, Any] | None = None) -> None:
        if client is None:
            try:
                from mem0 import Memory
            except ImportError as error:
                raise AdapterConfigurationError(_INSTALL) from error
            try:
                client = Memory.from_config(config) if config is not None else Memory()
            except Exception as error:
                raise AdapterConfigurationError(str(error)) from error
        self.client = client

    def upsert(self, record: RetrievalRecord) -> ExternalWriteResult:
        try:
            existing = self._find(record.projection_id, record.namespace)
            if len(existing) > 1:
                raise AdapterProtocolError(
                    f"Mem0 returned duplicate projections for {record.projection_id}."
                )
            if existing:
                item = existing[0]
                external_id = _required_id(item)
                if _mem0_payload(item) != record_payload(record):
                    self.client.update(
                        external_id,
                        text=record.text,
                        metadata=_mem0_update_metadata(record, item),
                    )
                return _write_result(record, external_id, created=False)

            response = self.client.add(
                [{"role": "user", "content": record.text}],
                user_id=record.namespace,
                metadata=_mem0_metadata(record),
                infer=False,
            )
            results = _results(response)
            if len(results) != 1:
                raise AdapterProtocolError("Mem0 direct import did not return exactly one memory.")
            return _write_result(record, _required_id(results[0]), created=True)
        except Exception as error:
            raise translated_error(error, AdapterWriteError) from error

    def search(self, request: SearchRequest) -> list[SearchHit]:
        try:
            response = self.client.search(
                request.query,
                top_k=request.limit,
                filters={
                    "user_id": request.namespace,
                    **{
                        _filter_key(key): value
                        for key, value in request.filters.items()
                    },
                },
            )
            return [_mem0_hit(item) for item in _results(response)]
        except Exception as error:
            raise translated_error(error, AdapterSearchError) from error

    def delete(self, projection_id: str, namespace: str) -> bool:
        try:
            existing = self._find(projection_id, namespace)
            for item in existing:
                self.client.delete(_required_id(item))
            return bool(existing)
        except Exception as error:
            raise translated_error(error, AdapterWriteError) from error

    def healthcheck(self) -> bool:
        try:
            self.client.get_all(filters={"user_id": "statefuse-healthcheck"}, top_k=1)
        except Exception:
            return False
        return True

    def _find(self, projection_id: str, namespace: str) -> list[Any]:
        return _results(
            self.client.get_all(
                filters={
                    "user_id": namespace,
                    "statefuse_projection_id": projection_id,
                },
                top_k=2,
            )
        )


class AsyncMem0Adapter:
    """Native async adapter for Mem0 OSS ``AsyncMemory`` clients."""

    name = "mem0"

    def __init__(self, client: Any | None = None, *, config: dict[str, Any] | None = None) -> None:
        if client is None:
            try:
                from mem0 import AsyncMemory
            except ImportError as error:
                raise AdapterConfigurationError(_INSTALL) from error
            try:
                client = AsyncMemory.from_config(config) if config is not None else AsyncMemory()
            except Exception as error:
                raise AdapterConfigurationError(str(error)) from error
        self.client = client

    async def aupsert(self, record: RetrievalRecord) -> ExternalWriteResult:
        try:
            existing = await self._find(record.projection_id, record.namespace)
            if len(existing) > 1:
                raise AdapterProtocolError(
                    f"Mem0 returned duplicate projections for {record.projection_id}."
                )
            if existing:
                item = existing[0]
                external_id = _required_id(item)
                if _mem0_payload(item) != record_payload(record):
                    await self.client.update(
                        external_id,
                        text=record.text,
                        metadata=_mem0_update_metadata(record, item),
                    )
                return _write_result(record, external_id, created=False)
            response = await self.client.add(
                [{"role": "user", "content": record.text}],
                user_id=record.namespace,
                metadata=_mem0_metadata(record),
                infer=False,
            )
            results = _results(response)
            if len(results) != 1:
                raise AdapterProtocolError("Mem0 direct import did not return exactly one memory.")
            return _write_result(record, _required_id(results[0]), created=True)
        except Exception as error:
            raise translated_error(error, AdapterWriteError) from error

    async def asearch(self, request: SearchRequest) -> list[SearchHit]:
        try:
            response = await self.client.search(
                request.query,
                top_k=request.limit,
                filters={
                    "user_id": request.namespace,
                    **{
                        _filter_key(key): value
                        for key, value in request.filters.items()
                    },
                },
            )
            return [_mem0_hit(item) for item in _results(response)]
        except Exception as error:
            raise translated_error(error, AdapterSearchError) from error

    async def adelete(self, projection_id: str, namespace: str) -> bool:
        try:
            existing = await self._find(projection_id, namespace)
            for item in existing:
                await self.client.delete(_required_id(item))
            return bool(existing)
        except Exception as error:
            raise translated_error(error, AdapterWriteError) from error

    async def ahealthcheck(self) -> bool:
        try:
            await self.client.get_all(filters={"user_id": "statefuse-healthcheck"}, top_k=1)
        except Exception:
            return False
        return True

    async def _find(self, projection_id: str, namespace: str) -> list[Any]:
        response = await self.client.get_all(
            filters={"user_id": namespace, "statefuse_projection_id": projection_id},
            top_k=2,
        )
        return _results(response)


def _results(response: object) -> list[Any]:
    values = item_value(response, "results")
    if not isinstance(values, list):
        raise AdapterProtocolError("Mem0 response is missing a results list.")
    return values


def _required_id(item: object) -> str:
    value = item_value(item, "id")
    if not isinstance(value, str) or not value:
        raise AdapterProtocolError("Mem0 response is missing a memory ID.")
    return value


def _mem0_payload(item: object) -> dict[str, Any]:
    metadata = normalized_metadata(item_value(item, "metadata"))
    canonical = {
        key: metadata.get(key)
        for key in (
            "statefuse_projection_id",
            "statefuse_namespace",
            "statefuse_claim_ids",
            "statefuse_conflict_ids",
            "statefuse_projection_version",
            "statefuse_metadata",
        )
    }
    return {"text": item_value(item, "memory", ""), **canonical}


def _mem0_hit(item: object) -> SearchHit:
    return metadata_hit(
        external_id=_required_id(item),
        text=item_value(item, "memory", ""),
        score=item_value(item, "score"),
        metadata=item_value(item, "metadata"),
    )


def _write_result(
    record: RetrievalRecord, external_id: str, *, created: bool
) -> ExternalWriteResult:
    return ExternalWriteResult(
        repository=Mem0Adapter.name,
        projection_id=record.projection_id,
        external_id=external_id,
        created=created,
        metadata=dict(record.metadata),
    )


def _mem0_metadata(record: RetrievalRecord) -> dict[str, Any]:
    return {
        **record_metadata(record),
        **{_filter_key(key): value for key, value in record.metadata.items()},
    }


def _mem0_update_metadata(record: RetrievalRecord, item: object) -> dict[str, Any]:
    previous = normalized_metadata(item_value(item, "metadata"))
    current = _mem0_metadata(record)
    removed_filters = {
        key: None
        for key in previous
        if key.startswith("statefuse_filter_") and key not in current
    }
    return {**removed_filters, **current}


def _filter_key(key: str) -> str:
    return f"statefuse_filter_{digest_json_value(key)}"
