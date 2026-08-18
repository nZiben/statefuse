from __future__ import annotations

from typing import Any

from ..utils import digest_json_value
from ._common import (
    item_value,
    metadata_hit,
    payload_matches_record,
    record_payload,
    translated_error,
)
from .errors import AdapterConfigurationError, AdapterSearchError, AdapterWriteError
from .models import ExternalWriteResult, RetrievalRecord, SearchHit, SearchRequest

_INSTALL = 'Install the LangMem integration with:\npip install "statefuse[langmem]"'


class LangGraphStoreAdapter:
    """Adapter for LangMem-compatible LangGraph ``BaseStore`` implementations."""

    name = "langmem"

    def __init__(self, store: Any | None = None, *, prefix: tuple[str, ...] = ("statefuse",)):
        if store is None:
            try:
                from langgraph.store.memory import InMemoryStore
            except ImportError as error:
                raise AdapterConfigurationError(_INSTALL) from error
            store = InMemoryStore()
        self.store = store
        self.prefix = tuple(prefix)

    def upsert(self, record: RetrievalRecord) -> ExternalWriteResult:
        try:
            namespace = self._namespace(record.namespace)
            external_id = _external_id(record.namespace, record.projection_id)
            existing = self.store.get(namespace, external_id)
            if existing is None or not payload_matches_record(
                item_value(existing, "value"), record
            ):
                self.store.put(namespace, external_id, record_payload(record), index=["text"])
            return ExternalWriteResult(
                self.name,
                record.projection_id,
                external_id,
                existing is None,
                dict(record.metadata),
            )
        except Exception as error:
            raise translated_error(error, AdapterWriteError) from error

    def search(self, request: SearchRequest) -> list[SearchHit]:
        try:
            items = self.store.search(
                self._namespace(request.namespace),
                query=request.query,
                limit=request.limit,
                filter={
                    f"statefuse_metadata.{key}": value
                    for key, value in request.filters.items()
                }
                or None,
            )
            return [_store_hit(item) for item in items]
        except Exception as error:
            raise translated_error(error, AdapterSearchError) from error

    def delete(self, projection_id: str, namespace: str) -> bool:
        try:
            store_namespace = self._namespace(namespace)
            external_id = _external_id(namespace, projection_id)
            if self.store.get(store_namespace, external_id) is None:
                return False
            self.store.delete(store_namespace, external_id)
            return True
        except Exception as error:
            raise translated_error(error, AdapterWriteError) from error

    def healthcheck(self) -> bool:
        try:
            self.store.list_namespaces(prefix=self.prefix, limit=1)
        except Exception:
            return False
        return True

    def _namespace(self, namespace: str) -> tuple[str, ...]:
        return (*self.prefix, namespace)


class AsyncLangGraphStoreAdapter:
    """Native async adapter for LangGraph stores."""

    name = "langmem"

    def __init__(self, store: Any, *, prefix: tuple[str, ...] = ("statefuse",)):
        if store is None:
            raise AdapterConfigurationError("An async LangGraph store is required.\n" + _INSTALL)
        self.store = store
        self.prefix = tuple(prefix)

    async def aupsert(self, record: RetrievalRecord) -> ExternalWriteResult:
        try:
            namespace = self._namespace(record.namespace)
            external_id = _external_id(record.namespace, record.projection_id)
            existing = await self.store.aget(namespace, external_id)
            if existing is None or not payload_matches_record(
                item_value(existing, "value"), record
            ):
                await self.store.aput(
                    namespace, external_id, record_payload(record), index=["text"]
                )
            return ExternalWriteResult(
                self.name,
                record.projection_id,
                external_id,
                existing is None,
                dict(record.metadata),
            )
        except Exception as error:
            raise translated_error(error, AdapterWriteError) from error

    async def asearch(self, request: SearchRequest) -> list[SearchHit]:
        try:
            items = await self.store.asearch(
                self._namespace(request.namespace),
                query=request.query,
                limit=request.limit,
                filter={
                    f"statefuse_metadata.{key}": value
                    for key, value in request.filters.items()
                }
                or None,
            )
            return [_store_hit(item) for item in items]
        except Exception as error:
            raise translated_error(error, AdapterSearchError) from error

    async def adelete(self, projection_id: str, namespace: str) -> bool:
        try:
            store_namespace = self._namespace(namespace)
            external_id = _external_id(namespace, projection_id)
            if await self.store.aget(store_namespace, external_id) is None:
                return False
            await self.store.adelete(store_namespace, external_id)
            return True
        except Exception as error:
            raise translated_error(error, AdapterWriteError) from error

    async def ahealthcheck(self) -> bool:
        try:
            await self.store.alist_namespaces(prefix=self.prefix, limit=1)
        except Exception:
            return False
        return True

    def _namespace(self, namespace: str) -> tuple[str, ...]:
        return (*self.prefix, namespace)


LangMemAdapter = LangGraphStoreAdapter
AsyncLangMemAdapter = AsyncLangGraphStoreAdapter


def _external_id(namespace: str, projection_id: str) -> str:
    payload = {"namespace": namespace, "projection_id": projection_id}
    return f"statefuse-{digest_json_value(payload)}"


def _store_hit(item: object) -> SearchHit:
    payload = item_value(item, "value", {})
    return metadata_hit(
        external_id=item_value(item, "key", ""),
        text=item_value(payload, "text", ""),
        score=item_value(item, "score"),
        metadata=payload,
    )
