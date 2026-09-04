from __future__ import annotations

from statefuse import Memory
from statefuse.integrations import SearchHit, hydrate_search_hits


def test_hydration_loads_canonical_conflicts_provenance_and_derivations() -> None:
    memory = _memory_with_neighborhood()

    context = hydrate_search_hits(memory, (_hit("c1"),))

    assert [claim.claim_id for claim in context.claims] == ["c0", "c1", "c2"]
    assert len(context.conflicts) == 1
    assert {claim.claim_id for claim in context.conflicts[0].candidates} == {"c1", "c2"}
    assert [item.evidence_id for item in context.evidence] == ["e0", "e1", "e2"]
    assert [item.source_id for item in context.sources] == ["s0", "s1", "s2"]
    assert [item.derivation_id for item in context.derivations] == ["d1"]
    assert context.missing_claim_ids == ()
    assert context.missing_conflict_ids == ()


def test_hydration_reports_effective_then_stale_resolution() -> None:
    memory = _memory_with_neighborhood()
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

    resolved = hydrate_search_hits(memory, (_hit("c1"),))

    assert [item.resolution_id for item in resolved.resolutions] == ["r1"]
    assert [item.resolution_id for item in resolved.resolution_history] == ["r1"]
    assert resolved.stale_resolutions == ()
    assert resolved.resolution_statuses == {"r1": "resolved"}

    memory.add_claim(
        namespace="project",
        subject="deadline",
        predicate="date",
        value="May 18",
        confidence=0.7,
        evidence_ids=(),
        claim_id="c3",
    )
    reopened = hydrate_search_hits(memory, (_hit("c1"),))

    assert reopened.resolutions == ()
    assert [item.resolution_id for item in reopened.stale_resolutions] == ["r1"]
    assert reopened.resolution_statuses == {"r1": "reopened"}
    assert set(reopened.conflict_statuses.values()) == {"reopened"}


def test_hydration_depth_status_and_bounds_are_explicit() -> None:
    memory = _memory_with_neighborhood()
    bounded = hydrate_search_hits(memory, (_hit("c1"),), max_evidence=1)
    memory.retract_claim(target_claim_id="c1", evidence_ids=(), reason="Withdrawn")
    hit = SearchHit(
        external_id="external",
        projection_id=None,
        text="stale",
        score=None,
        claim_ids=("c1", "missing"),
        conflict_ids=(),
        metadata={},
    )

    seed_only = hydrate_search_hits(memory, (hit,), max_depth=0)

    assert [claim.claim_id for claim in seed_only.claims] == ["c1"]
    assert seed_only.conflicts == ()
    assert seed_only.claim_statuses == {"c1": "inactive"}
    assert seed_only.missing_claim_ids == ("missing",)
    assert len(bounded.conflicts) == 1
    assert [item.evidence_id for item in bounded.evidence] == ["e0"]
    assert bounded.omitted_evidence_ids == ("e1", "e2")
    assert bounded.truncated is True


def _memory_with_neighborhood() -> Memory:
    memory = Memory(replica_id="test")
    for source_id in ("s0", "s1", "s2"):
        memory.add_source(
            source_type="document",
            uri=f"https://example.test/{source_id}",
            source_id=source_id,
        )
    for evidence_id, source_id in (("e0", "s0"), ("e1", "s1"), ("e2", "s2")):
        memory.add_evidence(
            pointer=f"line:{evidence_id}", source_id=source_id, evidence_id=evidence_id
        )
    memory.add_claim(
        namespace="project",
        subject="brief",
        predicate="status",
        value="approved",
        confidence=0.9,
        evidence_ids=("e0",),
        provenance={"author": "reviewer"},
        claim_id="c0",
    )
    memory.add_claim(
        namespace="project",
        subject="deadline",
        predicate="date",
        value="May 12",
        confidence=0.8,
        evidence_ids=("e1",),
        provenance={"author": "planner"},
        derivation_id="d1",
        claim_id="c1",
    )
    memory.add_claim(
        namespace="project",
        subject="deadline",
        predicate="date",
        value="May 15",
        confidence=0.9,
        evidence_ids=("e2",),
        provenance={"author": "manager"},
        claim_id="c2",
    )
    memory.add_derivation(
        rule_id="deadline-from-brief",
        input_claim_ids=("c0",),
        output_claim_ids=("c1",),
        engine="rules",
        explanation="The approved brief fixes the deadline.",
        derivation_id="d1",
    )
    return memory


def _hit(claim_id: str) -> SearchHit:
    return SearchHit(
        external_id="external",
        projection_id=None,
        text="retrieval text",
        score=0.9,
        claim_ids=(claim_id,),
        conflict_ids=(),
        metadata={},
    )
