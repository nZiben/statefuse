from __future__ import annotations

import asyncio

import pytest

from statefuse.integrations import (
    AdapterUnavailableError,
    FakeMemoryRepositoryAdapter,
    RetrievalRecord,
    SearchRequest,
)


class AdapterContract:
    """Reusable connector contract with adapter-specific outage controls."""

    def make_adapter(self):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def make_unavailable(self, adapter) -> None:  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def make_available(self, adapter) -> None:  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def test_new_projection_and_idempotent_upsert(self) -> None:
        adapter = self.make_adapter()
        record = _record()

        created = adapter.upsert(record)
        repeated = adapter.upsert(record)

        assert created.created is True
        assert repeated.created is False
        assert repeated.external_id == created.external_id
        assert len(adapter.search(SearchRequest("deadline", "project"))) == 1

    def test_projection_update(self) -> None:
        adapter = self.make_adapter()
        adapter.upsert(_record())

        result = adapter.upsert(_record(text="Project Alpha deadline is May 15."))

        assert result.created is False
        assert adapter.search(SearchRequest("May 15", "project"))[0].text.endswith("May 15.")

    def test_search_normalizes_metadata_and_statefuse_ids(self) -> None:
        adapter = self.make_adapter()
        record = _record()
        result = adapter.upsert(record)

        hit = adapter.search(SearchRequest("deadline", "project"))[0]

        assert hit.external_id == result.external_id
        assert hit.projection_id == record.projection_id
        assert hit.claim_ids == ("cl_1",)
        assert hit.conflict_ids == ("cf_1",)
        assert hit.metadata == {"kind": "claim", "rank": 1}

    def test_projection_deletion(self) -> None:
        adapter = self.make_adapter()
        adapter.upsert(_record())

        assert adapter.delete("statefuse:claim:cl_1", "project") is True
        assert adapter.delete("statefuse:claim:cl_1", "project") is False
        assert adapter.search(SearchRequest("deadline", "project")) == []

    def test_namespace_isolation(self) -> None:
        adapter = self.make_adapter()
        first = adapter.upsert(_record(namespace="project-a"))
        second = adapter.upsert(_record(namespace="project-b"))

        assert first.external_id != second.external_id
        assert len(adapter.search(SearchRequest("deadline", "project-a"))) == 1
        assert len(adapter.search(SearchRequest("deadline", "project-b"))) == 1

    def test_outage_and_recovery(self) -> None:
        adapter = self.make_adapter()
        self.make_unavailable(adapter)

        with pytest.raises(AdapterUnavailableError):
            adapter.upsert(_record())
        assert adapter.healthcheck() is False

        self.make_available(adapter)
        assert adapter.healthcheck() is True
        assert adapter.upsert(_record()).created is True


class AsyncAdapterContract:
    """Reusable version of the connector contract for native async adapters."""

    def make_adapter(self):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def make_unavailable(self, adapter) -> None:  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def make_available(self, adapter) -> None:  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def test_new_projection_and_idempotent_upsert(self) -> None:
        async def scenario() -> None:
            adapter = self.make_adapter()
            record = _record()
            created = await adapter.aupsert(record)
            repeated = await adapter.aupsert(record)
            assert created.created is True
            assert repeated.created is False
            assert repeated.external_id == created.external_id
            assert len(await adapter.asearch(SearchRequest("deadline", "project"))) == 1

        asyncio.run(scenario())

    def test_projection_update(self) -> None:
        async def scenario() -> None:
            adapter = self.make_adapter()
            await adapter.aupsert(_record())
            result = await adapter.aupsert(
                _record(text="Project Alpha deadline is May 15.")
            )
            assert result.created is False
            hits = await adapter.asearch(SearchRequest("May 15", "project"))
            assert hits[0].text.endswith("May 15.")

        asyncio.run(scenario())

    def test_search_normalizes_metadata_and_statefuse_ids(self) -> None:
        async def scenario() -> None:
            adapter = self.make_adapter()
            record = _record()
            result = await adapter.aupsert(record)
            hit = (await adapter.asearch(SearchRequest("deadline", "project")))[0]
            assert hit.external_id == result.external_id
            assert hit.projection_id == record.projection_id
            assert hit.claim_ids == ("cl_1",)
            assert hit.conflict_ids == ("cf_1",)
            assert hit.metadata == {"kind": "claim", "rank": 1}

        asyncio.run(scenario())

    def test_projection_deletion(self) -> None:
        async def scenario() -> None:
            adapter = self.make_adapter()
            await adapter.aupsert(_record())
            assert await adapter.adelete("statefuse:claim:cl_1", "project") is True
            assert await adapter.adelete("statefuse:claim:cl_1", "project") is False
            assert await adapter.asearch(SearchRequest("deadline", "project")) == []

        asyncio.run(scenario())

    def test_namespace_isolation(self) -> None:
        async def scenario() -> None:
            adapter = self.make_adapter()
            first = await adapter.aupsert(_record(namespace="project-a"))
            second = await adapter.aupsert(_record(namespace="project-b"))
            assert first.external_id != second.external_id
            assert len(await adapter.asearch(SearchRequest("deadline", "project-a"))) == 1
            assert len(await adapter.asearch(SearchRequest("deadline", "project-b"))) == 1

        asyncio.run(scenario())

    def test_outage_and_recovery(self) -> None:
        async def scenario() -> None:
            adapter = self.make_adapter()
            self.make_unavailable(adapter)
            with pytest.raises(AdapterUnavailableError):
                await adapter.aupsert(_record())
            assert await adapter.ahealthcheck() is False
            self.make_available(adapter)
            assert await adapter.ahealthcheck() is True
            assert (await adapter.aupsert(_record())).created is True

        asyncio.run(scenario())


class TestFakeAdapterContract(AdapterContract):
    def make_adapter(self) -> FakeMemoryRepositoryAdapter:
        return FakeMemoryRepositoryAdapter()

    def make_unavailable(self, adapter: FakeMemoryRepositoryAdapter) -> None:
        adapter.available = False

    def make_available(self, adapter: FakeMemoryRepositoryAdapter) -> None:
        adapter.available = True


def _record(*, text: str = "Project Alpha deadline is May 12.", namespace: str = "project"):
    return RetrievalRecord(
        projection_id="statefuse:claim:cl_1",
        text=text,
        namespace=namespace,
        claim_ids=("cl_1",),
        conflict_ids=("cf_1",),
        metadata={"kind": "claim", "rank": 1},
    )
