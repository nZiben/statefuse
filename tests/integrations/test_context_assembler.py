from __future__ import annotations

from dataclasses import replace

from statefuse import Memory
from statefuse.integrations import ContextAssembler, SearchHit, hydrate_search_hits


def test_unresolved_conflict_is_included_whole_or_reported_whole() -> None:
    memory = _conflicting_memory()
    context = hydrate_search_hits(memory, (_hit("c1"),))
    complete = ContextAssembler(10_000).assemble(context)

    assert complete.included_conflict_ids == (context.conflicts[0].conflict_id,)
    assert "\"claim_id\":\"c1\"" in complete.text
    assert "\"claim_id\":\"c2\"" in complete.text

    small_budget = complete.token_count - 1
    too_small = ContextAssembler(small_budget).assemble(context)

    assert too_small.text == ""
    assert too_small.included_claim_ids == ()
    assert too_small.omitted_conflict_ids == (context.conflicts[0].conflict_id,)
    assert too_small.omitted_claim_ids == ("c1", "c2")
    assert too_small.token_count <= small_budget


def test_resolved_conflict_keeps_resolution_and_provenance() -> None:
    memory = _conflicting_memory()
    conflict = memory.materialize().conflicts[0]
    memory.add_resolution(
        conflict_ref=conflict.conflict_ref,
        observed_conflict_id=conflict.conflict_id,
        selected_claim_ids=("c2",),
        rejected_claim_ids=("c1",),
        resolution_type="human",
        reason="Confirmed",
        actor_id="reviewer",
        evidence_ids=("e2",),
        resolution_id="r1",
    )

    result = ContextAssembler(10_000).assemble(hydrate_search_hits(memory, (_hit("c1"),)))

    assert "\"resolution_id\":\"r1\"" in result.text
    assert "\"source_id\":\"s2\"" in result.text
    assert "\"author\":\"manager\"" in result.text
    assert result.token_count == ContextAssembler.count_tokens(result.text)

    memory.add_claim(
        namespace="project",
        subject="deadline",
        predicate="date",
        value="May 18",
        confidence=0.7,
        evidence_ids=(),
        claim_id="c3",
    )
    reopened = ContextAssembler(10_000).assemble(
        hydrate_search_hits(memory, (_hit("c1"),))
    )

    assert "RESOLUTION_STATE status=reopened" in reopened.text
    assert "\"resolution_id\":\"r1\"" in reopened.text


def test_multiple_neighborhoods_are_packed_deterministically() -> None:
    memory = Memory(replica_id="test")
    for claim_id, predicate in (("a", "alpha"), ("z", "omega")):
        memory.add_claim(
            namespace="project",
            subject="item",
            predicate=predicate,
            value=claim_id,
            confidence=0.8,
            evidence_ids=(),
            claim_id=claim_id,
        )
    context = hydrate_search_hits(memory, (_hit("z"), _hit("a")))
    first_only = replace(
        context,
        claims=(context.claims[0],),
        claim_statuses={"a": "active"},
    )
    budget = ContextAssembler(10_000).assemble(first_only).token_count

    first = ContextAssembler(budget).assemble(context)
    second = ContextAssembler(budget).assemble(context)

    assert first == second
    assert first.included_claim_ids == ("a",)
    assert first.omitted_claim_ids == ("z",)
    assert first.token_count <= budget


def _conflicting_memory() -> Memory:
    memory = Memory(replica_id="test")
    for source_id in ("s1", "s2"):
        memory.add_source(source_type="document", source_id=source_id)
    for evidence_id, source_id in (("e1", "s1"), ("e2", "s2")):
        memory.add_evidence(pointer=evidence_id, source_id=source_id, evidence_id=evidence_id)
    for claim_id, value, evidence_id, author in (
        ("c1", "May 12", "e1", "planner"),
        ("c2", "May 15", "e2", "manager"),
    ):
        memory.add_claim(
            namespace="project",
            subject="deadline",
            predicate="date",
            value=value,
            confidence=0.8,
            evidence_ids=(evidence_id,),
            provenance={"author": author},
            claim_id=claim_id,
        )
    return memory


def _hit(*claim_ids: str) -> SearchHit:
    return SearchHit(
        external_id="-".join(claim_ids),
        projection_id=None,
        text="retrieval text",
        score=None,
        claim_ids=claim_ids,
        conflict_ids=(),
        metadata={},
    )
