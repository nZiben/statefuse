from __future__ import annotations

import asyncio

from statefuse import Memory
from statefuse.integrations import (
    AsyncProjectionService,
    FakeMemoryRepositoryAdapter,
    InMemoryExternalReferenceStore,
    InMemoryLocalProjectionDeltaStore,
    ProjectionService,
    SearchHit,
    SearchRequest,
    hydrate_search_hits,
)


def test_committed_unsynchronized_claim_is_searchable_locally() -> None:
    memory = _memory_with_claim("c1", "May 12")
    service = ProjectionService(
        memory,
        FakeMemoryRepositoryAdapter(),
        InMemoryExternalReferenceStore(),
    )

    context = service.search(SearchRequest("deadline", "project"))

    assert [claim.claim_id for claim in context.claims] == ["c1"]
    assert context.search_hits[0].metadata["freshness_source"] == "local_pending"


def test_projection_cursors_are_independent_per_namespace() -> None:
    memory = _memory_with_claim("c1", "May 12")
    memory.add_claim(
        namespace="profile",
        subject="user",
        predicate="city",
        value="Shenzhen",
        confidence=0.8,
        evidence_ids=(),
        claim_id="profile-1",
    )
    service = ProjectionService(
        memory,
        FakeMemoryRepositoryAdapter(),
        InMemoryExternalReferenceStore(),
    )

    project = service.synchronize("project")
    profile = service.synchronize("profile")

    assert project.created == ("statefuse:claim:c1",)
    assert profile.created == ("statefuse:claim:profile-1",)


def test_stale_external_hit_expands_to_current_conflict_competitor_and_resolution() -> None:
    memory = _memory_with_claim("c1", "May 12")
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
        claim_id="c2",
    )
    conflict = memory.materialize().conflicts[0]
    memory.add_resolution(
        conflict_ref=conflict.conflict_ref,
        observed_conflict_id=conflict.conflict_id,
        selected_claim_ids=("c2",),
        rejected_claim_ids=("c1",),
        resolution_type="human",
        reason="Confirmed",
        actor_id="reviewer",
        resolution_id="r1",
    )

    context = service.search(SearchRequest("deadline", "project"))

    assert {claim.claim_id for claim in context.claims} == {"c1", "c2"}
    assert [item.conflict_id for item in context.conflicts] == [conflict.conflict_id]
    assert [item.resolution_id for item in context.resolutions] == ["r1"]
    assert context.resolution_statuses == {"r1": "resolved"}
    c1_hit = next(hit for hit in context.search_hits if hit.claim_ids == ("c1",))
    assert c1_hit.metadata["freshness_source"] == "local_pending"
    assert c1_hit.conflict_ids == (conflict.conflict_id,)


def test_partial_sync_retires_successes_and_keeps_failed_delta_for_retry() -> None:
    memory = _memory_with_claim("c1", "May 12")
    adapter = FakeMemoryRepositoryAdapter()
    delta_store = InMemoryLocalProjectionDeltaStore()
    service = ProjectionService(
        memory,
        adapter,
        InMemoryExternalReferenceStore(),
        delta_store,
    )
    service.synchronize("project")
    memory.add_claim(
        namespace="project",
        subject="Project Alpha submission deadline",
        predicate="date",
        value="May 15",
        confidence=0.9,
        evidence_ids=(),
        claim_id="c2",
    )
    adapter.inject_failure("upsert")

    failed = service.synchronize("project")

    assert [item.projection_id for item in failed.failed] == ["statefuse:claim:c1"]
    assert [item.projection_id for item in delta_store.list("fake", "project")] == [
        "statefuse:claim:c1"
    ]

    recovered = service.synchronize("project")

    assert recovered.created == ("statefuse:claim:c1",)
    assert delta_store.list("fake", "project") == ()


def test_pending_delete_suppresses_stale_external_hit() -> None:
    memory = _memory_with_claim("c1", "May 12")
    adapter = FakeMemoryRepositoryAdapter()
    service = ProjectionService(memory, adapter, InMemoryExternalReferenceStore())
    service.synchronize("project")
    memory.retract_claim(target_claim_id="c1", evidence_ids=(), reason="Withdrawn")

    context = service.search(SearchRequest("deadline", "project"))

    assert context.search_hits == ()
    assert context.claims == ()
    assert adapter.record_count == 1


def test_hydration_bounds_are_deterministic_and_explicit() -> None:
    memory = _memory_with_claim("c1", "May 12")
    memory.add_claim(
        namespace="project",
        subject="Project Alpha submission deadline",
        predicate="date",
        value="May 15",
        confidence=0.9,
        evidence_ids=(),
        claim_id="c2",
    )
    adapter = FakeMemoryRepositoryAdapter()
    service = ProjectionService(memory, adapter, InMemoryExternalReferenceStore())
    service.synchronize("project")
    hit = adapter.search(SearchRequest("deadline", "project"))[0]

    context = hydrate_search_hits(memory, (hit,), max_claims=1)

    assert [claim.claim_id for claim in context.claims] == ["c1"]
    assert context.omitted_claim_ids == ("c2",)
    assert context.truncated is True


def test_async_search_uses_the_same_local_freshness_path() -> None:
    async def scenario() -> None:
        memory = _memory_with_claim("c1", "May 12")
        adapter = _AsyncFakeAdapter()
        service = AsyncProjectionService(
            memory,
            adapter,
            InMemoryExternalReferenceStore(),
        )

        context = await service.search(SearchRequest("deadline", "project"))

        assert [claim.claim_id for claim in context.claims] == ["c1"]
        assert context.search_hits[0].metadata["freshness_source"] == "local_pending"

    asyncio.run(scenario())


def test_external_search_failure_still_returns_pending_local_results() -> None:
    memory = _memory_with_claim("c1", "May 12")
    adapter = FakeMemoryRepositoryAdapter()
    adapter.inject_failure("search")
    service = ProjectionService(memory, adapter, InMemoryExternalReferenceStore())

    context = service.search(SearchRequest("deadline", "project"))

    assert [claim.claim_id for claim in context.claims] == ["c1"]
    assert context.search_hits[0].metadata["freshness_source"] == "local_pending"
    assert context.search_failures[0].operation == "search"


def test_async_external_search_failure_still_returns_pending_local_results() -> None:
    async def scenario() -> None:
        memory = _memory_with_claim("c1", "May 12")
        adapter = _AsyncFakeAdapter()
        adapter.inner.inject_failure("search")
        service = AsyncProjectionService(
            memory,
            adapter,
            InMemoryExternalReferenceStore(),
        )

        context = await service.search(SearchRequest("deadline", "project"))

        assert [claim.claim_id for claim in context.claims] == ["c1"]
        assert context.search_hits[0].metadata["freshness_source"] == "local_pending"
        assert context.search_failures[0].operation == "search"

    asyncio.run(scenario())


def test_pending_update_masks_stale_external_hit_that_no_longer_matches_filters() -> None:
    memory = _memory_with_claim("c1", "May 12")
    memory.add_claim(
        namespace="project",
        subject="Project Alpha submission deadline",
        predicate="date",
        value="May 15",
        confidence=0.9,
        evidence_ids=(),
        claim_id="c2",
    )
    adapter = FakeMemoryRepositoryAdapter()
    service = ProjectionService(memory, adapter, InMemoryExternalReferenceStore())
    service.synchronize("project")
    conflict = memory.materialize().conflicts[0]
    memory.add_resolution(
        conflict_ref=conflict.conflict_ref,
        observed_conflict_id=conflict.conflict_id,
        selected_claim_ids=("c2",),
        rejected_claim_ids=("c1",),
        resolution_type="human",
        reason="Confirmed",
        actor_id="reviewer",
        resolution_id="r1",
    )

    context = service.search(
        SearchRequest("conflict", "project", filters={"status": "open"})
    )

    assert context.search_hits == ()


def test_semantic_hit_keeps_its_score_but_hydrates_current_pending_state() -> None:
    memory = _memory_with_claim("c1", "May 12")
    memory.add_claim(
        namespace="project",
        subject="Project Alpha submission deadline",
        predicate="date",
        value="May 15",
        confidence=0.9,
        evidence_ids=(),
        claim_id="c2",
    )
    conflict = memory.materialize().conflicts[0]
    memory.add_resolution(
        conflict_ref=conflict.conflict_ref,
        observed_conflict_id=conflict.conflict_id,
        selected_claim_ids=("c2",),
        rejected_claim_ids=("c1",),
        resolution_type="human",
        reason="Confirmed",
        actor_id="reviewer",
        resolution_id="r1",
    )

    class SemanticAdapter:
        name = "semantic"

        def search(self, _request):  # type: ignore[no-untyped-def]
            return [
                SearchHit(
                    external_id="semantic:c1",
                    projection_id="statefuse:claim:c1",
                    text="stale",
                    score=0.97,
                    claim_ids=("c1",),
                    conflict_ids=(),
                    metadata={},
                )
            ]

    service = ProjectionService(
        memory, SemanticAdapter(), InMemoryExternalReferenceStore()  # type: ignore[arg-type]
    )
    context = service.search(SearchRequest("semantic-only", "project"))

    assert context.search_hits[0].score == 0.97
    assert context.search_hits[0].metadata["freshness_source"] == "local_pending"
    assert {claim.claim_id for claim in context.claims} == {"c1", "c2"}
    assert [item.resolution_id for item in context.resolutions] == ["r1"]


def test_reopened_conflict_does_not_hydrate_a_stale_resolution() -> None:
    memory = _memory_with_claim("c1", "May 12")
    memory.add_claim(
        namespace="project",
        subject="Project Alpha submission deadline",
        predicate="date",
        value="May 15",
        confidence=0.9,
        evidence_ids=(),
        claim_id="c2",
    )
    conflict = memory.materialize().conflicts[0]
    memory.add_resolution(
        conflict_ref=conflict.conflict_ref,
        observed_conflict_id=conflict.conflict_id,
        selected_claim_ids=("c2",),
        rejected_claim_ids=("c1",),
        resolution_type="human",
        reason="Confirmed",
        actor_id="reviewer",
        resolution_id="r1",
    )
    memory.add_claim(
        namespace="project",
        subject="Project Alpha submission deadline",
        predicate="date",
        value="May 18",
        confidence=0.9,
        evidence_ids=(),
        claim_id="c3",
    )
    hit = SearchHit(
        external_id="hit:c1",
        projection_id="statefuse:claim:c1",
        text="deadline",
        score=1.0,
        claim_ids=("c1",),
        conflict_ids=(),
        metadata={},
    )

    context = hydrate_search_hits(memory, (hit,))

    assert context.resolutions == ()


def test_full_rebuild_removes_stale_unsynchronized_pending_upsert() -> None:
    memory = _memory_with_claim("c1", "May 12")
    adapter = FakeMemoryRepositoryAdapter()
    references = InMemoryExternalReferenceStore()
    deltas = InMemoryLocalProjectionDeltaStore()
    first_service = ProjectionService(memory, adapter, references, deltas)
    assert first_service.search(SearchRequest("deadline", "project")).claims
    memory.retract_claim(target_claim_id="c1", evidence_ids=(), reason="Withdrawn")

    restarted_service = ProjectionService(memory, adapter, references, deltas)
    context = restarted_service.search(SearchRequest("deadline", "project"))

    assert context.search_hits == ()
    assert context.claims == ()


class _AsyncFakeAdapter:
    name = "fake"

    def __init__(self) -> None:
        self.inner = FakeMemoryRepositoryAdapter()

    async def aupsert(self, record):  # type: ignore[no-untyped-def]
        return self.inner.upsert(record)

    async def asearch(self, request):  # type: ignore[no-untyped-def]
        return self.inner.search(request)

    async def adelete(self, projection_id, namespace):  # type: ignore[no-untyped-def]
        return self.inner.delete(projection_id, namespace)

    async def ahealthcheck(self) -> bool:
        return self.inner.healthcheck()


def _memory_with_claim(claim_id: str, value: str) -> Memory:
    memory = Memory(replica_id="test")
    memory.add_claim(
        namespace="project",
        subject="Project Alpha submission deadline",
        predicate="date",
        value=value,
        confidence=0.8,
        evidence_ids=(),
        claim_id=claim_id,
    )
    return memory
