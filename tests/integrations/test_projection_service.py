from __future__ import annotations

import asyncio

from statefuse import Memory
from statefuse.integrations import (
    AsyncProjectionService,
    FakeMemoryRepositoryAdapter,
    InMemoryExternalReferenceStore,
    ProjectionService,
    SearchRequest,
    project_state,
)

from .test_connectors import _GraphitiClient


def test_repeated_synchronization_is_idempotent() -> None:
    memory = _memory_with_claim("cl_1", "May 12")
    adapter = FakeMemoryRepositoryAdapter()
    references = InMemoryExternalReferenceStore()
    service = ProjectionService(memory, adapter, references)

    first = service.synchronize("project")
    second = service.synchronize("project")

    assert first.created == ("statefuse:claim:cl_1",)
    assert second.unchanged == first.created
    assert adapter.record_count == 1
    assert adapter.write_count == 1
    reference = references.get("fake", "project", "statefuse:claim:cl_1")
    assert reference is not None
    assert references.get_by_external_id("fake", "project", reference.external_id) == reference


def test_changed_and_removed_projections_are_updated_and_deleted() -> None:
    memory = _memory_with_claim("cl_1", "May 12")
    adapter = FakeMemoryRepositoryAdapter()
    service = ProjectionService(memory, adapter, InMemoryExternalReferenceStore())
    service.synchronize("project")
    memory.add_claim(
        namespace="project",
        subject="Project Alpha submission deadline",
        predicate="date",
        value="May 15",
        confidence=0.9,
        evidence_ids=(),
        claim_id="cl_2",
    )

    conflicted = service.synchronize("project")
    conflict_id = memory.materialize().conflicts[0].conflict_id
    memory.retract_claim(target_claim_id="cl_1", evidence_ids=(), reason="Superseded.")
    retracted = service.synchronize("project")

    assert "statefuse:claim:cl_1" in conflicted.updated
    assert f"statefuse:conflict:{conflict_id}" in conflicted.created
    assert set(retracted.deleted) == {
        "statefuse:claim:cl_1",
        f"statefuse:conflict:{conflict_id}",
    }


def test_claim_conflict_and_resolution_projection_ids_are_deterministic() -> None:
    memory = _memory_with_claim("cl_1", "May 12")
    memory.add_claim(
        namespace="project",
        subject="Project Alpha submission deadline",
        predicate="date",
        value="May 15",
        confidence=0.9,
        evidence_ids=(),
        claim_id="cl_2",
    )
    conflict = memory.materialize().conflicts[0]
    memory.add_resolution(
        conflict_ref=conflict.conflict_ref,
        observed_conflict_id=conflict.conflict_id,
        selected_claim_ids=("cl_2",),
        rejected_claim_ids=("cl_1",),
        resolution_type="human",
        reason="Confirmed by the official organizer.",
        actor_id="reviewer",
        resolution_id="rs_1",
    )

    records = project_state(memory.materialize(), "project")
    ids = {record.projection_id for record in records}

    assert ids == {
        "statefuse:claim:cl_1",
        "statefuse:claim:cl_2",
        f"statefuse:conflict:{conflict.conflict_id}",
        "statefuse:resolution:rs_1",
    }
    assert all(record.projection_version == 1 for record in records)


def test_search_is_hydrated_through_current_state() -> None:
    memory = _memory_with_claim("cl_1", "May 12")
    adapter = FakeMemoryRepositoryAdapter()
    service = ProjectionService(memory, adapter, InMemoryExternalReferenceStore())
    service.synchronize("project")

    context = service.search(SearchRequest("deadline", "project"))

    assert [claim.claim_id for claim in context.claims] == ["cl_1"]
    assert context.claim_statuses == {"cl_1": "active"}


def test_async_projection_service_is_idempotent_and_hydrates() -> None:
    async def scenario() -> None:
        from types import SimpleNamespace

        from statefuse.integrations import AsyncGraphitiAdapter

        memory = _memory_with_claim("cl_1", "May 12")
        adapter = AsyncGraphitiAdapter(
            _GraphitiClient(),
            episode_type="json",
            search_config_factory=lambda limit: SimpleNamespace(limit=limit),
        )
        service = AsyncProjectionService(
            memory, adapter, InMemoryExternalReferenceStore()
        )
        first = await service.synchronize("project")
        second = await service.synchronize("project")
        context = await service.search(SearchRequest("deadline", "project"))
        assert first.created == ("statefuse:claim:cl_1",)
        assert second.unchanged == first.created
        assert context.claim_statuses == {"cl_1": "active"}

    asyncio.run(scenario())


def _memory_with_claim(claim_id: str, value: str) -> Memory:
    memory = Memory(replica_id="test")
    memory.add_claim(
        namespace="project",
        subject="Project Alpha submission deadline",
        predicate="date",
        value=value,
        confidence=0.8,
        evidence_ids=("ev_21",),
        claim_id=claim_id,
    )
    return memory
