# Build a custom adapter

Implement the four synchronous methods in `MemoryRepositoryAdapter` and normalize every SDK
response into StateFuse integration models. Keep SDK types inside the connector module.

```python
from statefuse.integrations import (
    AdapterSearchError,
    AdapterUnavailableError,
    AdapterWriteError,
    ExternalWriteResult,
    RetrievalRecord,
    SearchHit,
    SearchRequest,
)


class AcmeAdapter:
    name = "acme"

    def __init__(self, client):
        self.client = client

    def upsert(self, record: RetrievalRecord) -> ExternalWriteResult:
        try:
            item = self.client.upsert(
                key=record.projection_id,
                namespace=record.namespace,
                text=record.text,
                metadata={
                    **record.metadata,
                    "statefuse_claim_ids": list(record.claim_ids),
                    "statefuse_conflict_ids": list(record.conflict_ids),
                    "statefuse_projection_version": record.projection_version,
                },
            )
        except Exception as error:
            raise AdapterWriteError(str(error)) from error
        return ExternalWriteResult(
            repository=self.name,
            projection_id=record.projection_id,
            external_id=str(item["id"]),
            created=bool(item["created"]),
            metadata=dict(item.get("metadata", {})),
        )

    def search(self, request: SearchRequest) -> list[SearchHit]:
        try:
            items = self.client.search(
                query=request.query,
                namespace=request.namespace,
                limit=request.limit,
                filters=request.filters,
            )
        except Exception as error:
            raise AdapterSearchError(str(error)) from error
        return [
            SearchHit(
                external_id=str(item["id"]),
                projection_id=None if item.get("key") is None else str(item["key"]),
                text=str(item.get("text", "")),
                score=None if item.get("score") is None else float(item["score"]),
                claim_ids=tuple(item.get("metadata", {}).get("statefuse_claim_ids", ())),
                conflict_ids=tuple(
                    item.get("metadata", {}).get("statefuse_conflict_ids", ())
                ),
                metadata=dict(item.get("metadata", {})),
            )
            for item in items
        ]

    def delete(self, projection_id: str, namespace: str) -> bool:
        try:
            return bool(self.client.delete(key=projection_id, namespace=namespace))
        except Exception as error:
            raise AdapterWriteError(str(error)) from error

    def healthcheck(self) -> bool:
        try:
            return bool(self.client.ping())
        except Exception as error:
            raise AdapterUnavailableError(str(error)) from error
```

Use it after committing StateFuse data:

```python
from statefuse.integrations import InMemoryExternalReferenceStore, ProjectionService

service = ProjectionService(memory, AcmeAdapter(client), InMemoryExternalReferenceStore())
report = service.synchronize("project")
```

Do not call the adapter before the StateFuse operation is committed. Do not reconstruct canonical
claim activity, conflicts, or resolutions from external metadata. For an async-only SDK, implement
`AsyncMemoryRepositoryAdapter` and use `AsyncProjectionService`:

```python
from statefuse.integrations import AsyncProjectionService

service = AsyncProjectionService(memory, async_adapter, reference_store)
report = await service.synchronize("project")
context = await service.search(SearchRequest("submission deadline", "project"))
```

A connector may add a documented sync wrapper if its users need one, but a wrapper must not call
`asyncio.run()` from an already-running event loop.
