from __future__ import annotations

from collections.abc import Iterable

import pytest

from statefuse import (
    Claim,
    ClaimAdded,
    ClaimKey,
    ConflictDetectionContext,
    ConflictSet,
    Derivation,
    DerivationAdded,
    Evidence,
    EvidenceAdded,
    Memory,
    OpLog,
    PredicateRegistry,
    ResolutionAdded,
    ResolutionRecord,
    ValidityInterval,
    ViewConstraints,
    build_view,
    compact_oplog_with_report,
    make_conflict,
    materialize,
)


def _claim(
    claim_id: str,
    subject: str,
    predicate: str,
    value: object,
    index: int,
    *,
    kind: str = "fact",
    context: dict[str, object] | None = None,
    validity: ValidityInterval | None = None,
    derivation_id: str | None = None,
) -> ClaimAdded:
    timestamp = f"2026-03-01T00:00:{index:02d}Z"
    claim = Claim(
        claim_id=claim_id,
        key=ClaimKey("project", subject, predicate),
        value=value,  # type: ignore[arg-type]
        confidence=0.8,
        timestamp=timestamp,
        kind=kind,
        context=dict(context or {}),  # type: ignore[arg-type]
        validity=validity,
        derivation_id=derivation_id,
    )
    return ClaimAdded(
        op_id=f"op-{index:02d}-{claim_id}",
        replica_id="test",
        timestamp=timestamp,
        claim=claim,
    )


def test_direct_conflicts_require_context_and_half_open_time_overlap() -> None:
    overlapping = materialize(
        OpLog(
            [
                _claim(
                    "c1",
                    "shop",
                    "state",
                    "open",
                    1,
                    context={"location": "London"},
                    validity=ValidityInterval(
                        "2026-04-01T10:00:00Z", "2026-04-01T12:00:00Z"
                    ),
                ),
                _claim(
                    "c2",
                    "shop",
                    "state",
                    "closed",
                    2,
                    context={"location": "London"},
                    validity=ValidityInterval(
                        "2026-04-01T11:00:00Z", "2026-04-01T13:00:00Z"
                    ),
                ),
            ]
        )
    )

    assert len(overlapping.conflicts) == 1
    assert overlapping.conflicts[0].witness["incompatible_pairs"] == [
        {
            "claim_ids": ["c1", "c2"],
            "valid_from": "2026-04-01T11:00:00Z",
            "valid_until": "2026-04-01T12:00:00Z",
            "context": {"location": "London"},
        }
    ]

    boundary = materialize(
        OpLog(
            [
                _claim(
                    "c1",
                    "shop",
                    "state",
                    "open",
                    1,
                    validity=ValidityInterval(
                        "2026-04-01T10:00:00Z", "2026-04-01T11:00:00Z"
                    ),
                ),
                _claim(
                    "c2",
                    "shop",
                    "state",
                    "closed",
                    2,
                    validity=ValidityInterval(
                        "2026-04-01T11:00:00Z", "2026-04-01T12:00:00Z"
                    ),
                ),
            ]
        )
    )
    different_context = materialize(
        OpLog(
            [
                _claim("c1", "shop", "state", "open", 1, context={"location": "London"}),
                _claim("c2", "shop", "state", "closed", 2, context={"location": "Paris"}),
            ]
        )
    )

    assert boundary.conflicts == []
    assert different_context.conflicts == []
    key = ClaimKey("project", "shop", "state")
    contextual_view = build_view(different_context, ViewConstraints())
    assert contextual_view.selected_claims == {}
    assert {claim.claim_id for claim in contextual_view.compatible_claims[key]} == {
        "c1",
        "c2",
    }

    empty_interval = materialize(
        OpLog(
            [
                _claim(
                    "c1",
                    "shop",
                    "state",
                    "open",
                    1,
                    validity=ValidityInterval(
                        "2026-04-01T11:00:00Z", "2026-04-01T11:00:00Z"
                    ),
                ),
                _claim("c2", "shop", "state", "closed", 2),
            ]
        )
    )
    assert empty_interval.conflicts == []


def test_valid_at_and_context_filter_claim_applicability() -> None:
    oplog = OpLog(
        [
            _claim(
                "c1",
                "shop",
                "state",
                "open",
                1,
                context={"location": "London"},
                validity=ValidityInterval(valid_until="2026-04-01T11:00:00Z"),
            ),
            _claim(
                "c2",
                "shop",
                "state",
                "closed",
                2,
                context={"location": "London"},
                validity=ValidityInterval(valid_from="2026-04-01T11:00:00Z"),
            ),
            _claim("c3", "shop", "state", "busy", 3, context={"location": "Paris"}),
        ]
    )
    state = materialize(
        oplog,
        valid_at="2026-04-01T11:00:00Z",
        context={"location": "London"},
    )

    key = ClaimKey("project", "shop", "state")
    assert [claim.claim_id for claim in state.active_claims_by_key[key]] == ["c2"]
    assert state.inapplicable_claim_ids == {"c1", "c3"}
    assert state.conflicts == []

    contextual = materialize(
        OpLog(
            [
                _claim("global", "shop", "state", "open", 1),
                _claim(
                    "london",
                    "shop",
                    "state",
                    "closed",
                    2,
                    context={"location": "London"},
                ),
            ]
        )
    )
    assert len(contextual.find_conflicts(context={"location": "London"})) == 1
    assert contextual.find_conflicts(context={"location": "Paris"}) == ()

    aggregate = materialize(
        OpLog(
            [
                _claim("london-open", "shop", "state", "open", 1, context={"city": "London"}),
                _claim(
                    "london-closed", "shop", "state", "closed", 2, context={"city": "London"}
                ),
                _claim("paris-open", "shop", "state", "open", 3, context={"city": "Paris"}),
                _claim(
                    "paris-closed", "shop", "state", "closed", 4, context={"city": "Paris"}
                ),
            ]
        )
    )
    assert aggregate.conflicts[0].annotations["applicability"] == "aggregate"
    assert len(aggregate.find_conflicts(context={"city": "London"})) == 1
    assert len(aggregate.find_conflicts(context={"city": "Paris"})) == 1
    assert aggregate.find_conflicts(context={"city": "Berlin"}) == ()
    assert build_view(aggregate, ViewConstraints()).unresolved_conflicts == aggregate.conflicts

    partial = materialize(
        OpLog(
            [
                _claim("london-open", "shop", "state", "open", 1, context={"city": "London"}),
                _claim(
                    "london-closed", "shop", "state", "closed", 2, context={"city": "London"}
                ),
                _claim("paris-open", "shop", "state", "open", 3, context={"city": "Paris"}),
            ]
        )
    )
    assert [
        claim.claim_id
        for claim in build_view(partial, ViewConstraints()).compatible_claims[
            ClaimKey("project", "shop", "state")
        ]
    ] == ["paris-open"]


def test_claim_semantics_are_optional_but_part_of_non_default_identity() -> None:
    key = ClaimKey("project", "release", "status")
    legacy = Claim("c1", key, "ready", 0.8, "2026-03-01T00:00:00Z")
    instruction = Claim(
        "c2",
        key,
        "ready",
        0.8,
        "2026-03-01T00:00:00Z",
        kind="instruction",
        context={"role": "reviewer"},
    )

    assert "kind" not in legacy.to_dict()
    assert "context" not in legacy.to_dict()
    assert Claim.from_dict(legacy.to_dict()) == legacy
    assert Claim.from_dict(instruction.to_dict()) == instruction
    assert instruction.claim_ref != legacy.claim_ref
    with pytest.raises(ValueError, match="kind must be a non-empty string"):
        Claim.from_dict({**legacy.to_dict(), "kind": None})

    registry = PredicateRegistry()
    registry.register("status", normalize=lambda value: str(value).strip().lower())
    equivalent = materialize(
        OpLog(
            [
                _claim("raw-a", "release", "status", " Ready ", 1),
                _claim("raw-b", "release", "status", "ready", 2),
            ]
        ),
        predicate_registry=registry,
    )
    projection = build_view(equivalent, ViewConstraints())
    assert projection.compatible_claims == {}
    assert next(iter(projection.selected_claims.values())).claim_id == "raw-b"


def _budget_detector(context: ConflictDetectionContext) -> Iterable[ConflictSet]:
    claims = sorted(context.claims_by_id.values(), key=lambda claim: claim.claim_id)
    capacity = next((claim for claim in claims if claim.key.predicate == "capacity"), None)
    costs = [claim for claim in claims if claim.key.predicate == "cost"]
    required = sum(float(claim.value) for claim in costs)
    if capacity is None or required <= float(capacity.value):
        return ()
    candidates = [capacity, *costs]
    return (
        make_conflict(
            candidates=candidates,
            key=capacity.key,
            conflict_type="execution.resource.capacity",
            conflict_class="execution",
            conflict_subclass="resource.capacity",
            detector_id="budget/v1",
            reason="Combined cost exceeds available capacity.",
            witness={"required": required, "available": float(capacity.value)},
        ),
    )


def test_custom_detector_finds_deterministic_multi_key_conflict_and_query_filters() -> None:
    ops = [
        _claim("capacity", "account", "capacity", 100, 1, kind="constraint"),
        _claim("cost-a", "task-a", "cost", 70, 2, kind="resource"),
        _claim("cost-b", "task-b", "cost", 60, 3, kind="resource"),
    ]
    with pytest.raises(ValueError, match="explicit stable key anchor"):
        make_conflict(
            candidates=tuple(op.claim for op in ops),
            conflict_type="execution.resource.capacity",
            conflict_class="execution",
            conflict_subclass="resource.capacity",
            detector_id="budget/v1",
            reason="Missing stable locus.",
        )
    assert materialize(OpLog(ops)).conflicts == []

    left = materialize(OpLog(ops), conflict_detectors=(_budget_detector, _budget_detector))
    right = materialize(OpLog(reversed(ops)), conflict_detectors=(_budget_detector,))
    conflict = left.conflicts[0]

    assert [item.to_dict() for item in left.conflicts] == [
        item.to_dict() for item in right.conflicts
    ]
    assert conflict.conflict_type == "execution.resource.capacity"
    assert len(conflict.keys) == 3
    assert conflict.witness == {"required": 130.0, "available": 100.0}
    assert left.find_conflicts(conflict_class="execution") == (conflict,)
    assert left.find_conflicts(claim_id="cost-b") == (conflict,)
    assert left.find_conflicts(namespace="missing") == ()
    assert left.find_conflicts(status="open") == (conflict,)

    projection = build_view(left, ViewConstraints())
    assert projection.unresolved_conflicts == [conflict]
    assert projection.surfaced_findings == {conflict.conflict_id: conflict}
    assert len(projection.selected_claims) == 3


def test_multi_key_resolution_reopens_when_a_new_candidate_appears() -> None:
    claims = [
        _claim("capacity", "account", "capacity", 100, 1, kind="constraint"),
        _claim("cost-a", "task-a", "cost", 70, 2, kind="resource"),
        _claim("cost-b", "task-b", "cost", 60, 3, kind="resource"),
    ]
    original = materialize(OpLog(claims), conflict_detectors=(_budget_detector,)).conflicts[0]
    resolved = materialize(
        OpLog(
            [
                *claims,
                _resolution(
                    original,
                    outcome="select",
                    selected=("capacity",),
                    rejected=("cost-a", "cost-b"),
                ),
            ]
        ),
        conflict_detectors=(_budget_detector,),
    )

    assert resolved.lifecycle_status_by_conflict_ref_and_scope[
        (original.conflict_ref, None)
    ] == "resolved"
    resolved_view = build_view(resolved, ViewConstraints())
    assert {claim.claim_id for claim in resolved_view.selected_claims.values()} == {"capacity"}

    expanded = materialize(
        OpLog(
            [
                *claims,
                _resolution(
                    original,
                    outcome="select",
                    selected=("capacity",),
                    rejected=("cost-a", "cost-b"),
                ),
                _claim("cost-c", "task-c", "cost", 10, 4, kind="resource"),
            ]
        ),
        conflict_detectors=(_budget_detector,),
    )
    current = expanded.conflicts[0]

    assert current.conflict_ref == original.conflict_ref
    assert current.conflict_id != original.conflict_id
    assert expanded.lifecycle_status_by_conflict_ref_and_scope[
        (original.conflict_ref, None)
    ] == "reopened"


def test_detector_rejects_unknown_candidate_claims() -> None:
    op = _claim("known", "task", "cost", 70, 1)
    unknown = Claim(
        "unknown",
        ClaimKey("project", "other", "cost"),
        60,
        0.8,
        "2026-03-01T00:00:02Z",
    )

    def invalid(context: ConflictDetectionContext) -> Iterable[ConflictSet]:
        return (
            make_conflict(
                candidates=(context.claims_by_id["known"], unknown),
                key=context.claims_by_id["known"].key,
                conflict_type="execution.resource.capacity",
                conflict_class="execution",
                conflict_subclass="resource.capacity",
                detector_id="invalid/v1",
                reason="invalid fixture",
            ),
        )

    with pytest.raises(ValueError, match="inactive, unknown, or modified claims: unknown"):
        materialize(OpLog([op]), conflict_detectors=(invalid,))

    def modified(context: ConflictDetectionContext) -> Iterable[ConflictSet]:
        claim = context.claims_by_id["known"]
        changed = Claim.from_dict({**claim.to_dict(), "value": 999})
        return (
            make_conflict(
                candidates=(changed,),
                conflict_type="execution.resource.capacity",
                conflict_class="execution",
                conflict_subclass="resource.capacity",
                detector_id="invalid/v2",
                reason="invalid fixture",
            ),
        )

    with pytest.raises(ValueError, match="inactive, unknown, or modified claims: known"):
        materialize(OpLog([op]), conflict_detectors=(modified,))

    def noncanonical(context: ConflictDetectionContext) -> Iterable[ConflictSet]:
        claim = context.claims_by_id["known"]
        return (
            ConflictSet(
                conflict_id="random",
                key=claim.key,
                candidates=(claim,),
                distinct_values=(claim.value,),
                reason="invalid fixture",
                conflict_type="execution.resource.capacity",
                conflict_class="execution",
                conflict_subclass="resource.capacity",
                detector_id="invalid/v3",
            ),
        )

    with pytest.raises(ValueError, match="canonical identities"):
        materialize(OpLog([op]), conflict_detectors=(noncanonical,))


def test_normative_and_multi_hop_annotations_use_structured_claims() -> None:
    instructions = materialize(
        OpLog(
            [
                _claim("c1", "report", "delivery", "send", 1, kind="instruction"),
                _claim("c2", "report", "delivery", "wait", 2, kind="instruction"),
            ]
        )
    )
    assert instructions.conflicts[0].conflict_class == "normative"
    assert instructions.conflicts[0].conflict_subclass == "instruction"
    assert build_view(instructions, ViewConstraints()).unresolved_conflicts == [
        instructions.conflicts[0]
    ]

    derivation = DerivationAdded(
        op_id="op-03-derivation",
        replica_id="test",
        timestamp="2026-03-01T00:00:03Z",
        derivation=Derivation(
            derivation_id="d1",
            rule_id="location-rule",
            input_claim_ids=("input-1", "input-2"),
            output_claim_ids=("derived",),
            engine="rules/v1",
            explanation="Combined two location facts.",
            timestamp="2026-03-01T00:00:03Z",
        ),
    )
    multihop = materialize(
        OpLog(
            [
                _claim("input-1", "alice", "location", "Paris", 1),
                _claim("input-2", "laptop", "owner", "Alice", 2),
                derivation,
                _claim("derived", "laptop", "location", "Paris", 4, derivation_id="d1"),
                _claim("observed", "laptop", "location", "Tokyo", 5),
            ]
        )
    )
    conflict = next(item for item in multihop.conflicts if item.key.subject == "laptop")
    assert conflict.annotations["dependency_depth"] == "multi_hop"

    memory = Memory(replica_id="rules")
    derivation_id = memory.add_derivation(
        derivation_id="d-api",
        rule_id="rule/v1",
        input_claim_ids=("c1", "c2"),
        output_claim_ids=("c3",),
        engine="stdlib",
        explanation="Combined two inputs.",
    )
    assert memory.materialize().derivations_by_id[derivation_id].output_claim_ids == ("c3",)


def _resolution(
    conflict: ConflictSet,
    *,
    outcome: str,
    selected: tuple[str, ...] = (),
    rejected: tuple[str, ...] = (),
    retained: tuple[str, ...] = (),
) -> ResolutionAdded:
    record = ResolutionRecord(
        resolution_id=f"resolution-{outcome}",
        conflict_ref=conflict.conflict_ref,
        observed_conflict_id=conflict.conflict_id,
        selected_claim_ids=selected,
        rejected_claim_ids=rejected,
        retained_claim_ids=retained,
        resolution_type="human_review",
        reason="Reviewed all candidates.",
        evidence_ids=(),
        actor_id="reviewer",
        timestamp="2026-03-01T00:01:00Z",
        outcome=outcome,
    )
    return ResolutionAdded(
        op_id=f"op-resolution-{outcome}",
        replica_id="reviewer",
        timestamp=record.timestamp,
        resolution=record,
    )


def test_preserve_and_abstain_are_first_class_resolution_outcomes() -> None:
    claims = [
        _claim("c1", "release", "status", "ready", 1),
        _claim("c2", "release", "status", "blocked", 2),
    ]
    conflict = materialize(OpLog(claims)).conflicts[0]
    preserved = materialize(
        OpLog([*claims, _resolution(conflict, outcome="preserve", retained=("c1", "c2"))])
    )
    preserved_view = build_view(preserved, ViewConstraints())

    assert preserved.lifecycle_status_by_conflict_ref_and_scope[
        (conflict.conflict_ref, None)
    ] == "resolved"
    assert preserved_view.selected_claims == {}
    assert preserved_view.unresolved_conflicts == []
    assert preserved.find_conflicts(status="resolved") == (preserved.conflicts[0],)

    merge_claims = [
        *claims,
        _claim("merged", "release", "status", ["ready", "blocked"], 3),
    ]
    merge_conflict = materialize(OpLog(merge_claims)).conflicts[0]
    merged = materialize(
        OpLog(
            [
                *merge_claims,
                _resolution(
                    merge_conflict,
                    outcome="merge",
                    selected=("merged",),
                    retained=("c1", "c2"),
                ),
            ]
        )
    )
    assert merged.lifecycle_status_by_conflict_ref_and_scope[
        (conflict.conflict_ref, None)
    ] == "resolved"
    key = ClaimKey("project", "release", "status")
    assert build_view(merged, ViewConstraints()).selected_claims[key].claim_id == "merged"

    replaced = materialize(
        OpLog(
            [
                *claims,
                _resolution(
                    conflict,
                    outcome="replace",
                    selected=("c2",),
                    rejected=("c1",),
                ),
            ]
        )
    )
    assert build_view(replaced, ViewConstraints()).selected_claims[key].claim_id == "c2"

    with pytest.raises(ValueError, match="requires one merged claim"):
        _resolution(conflict, outcome="merge", retained=("c1", "c2"))

    reopened = materialize(
        OpLog(
            [
                *claims,
                _resolution(conflict, outcome="preserve", retained=("c1", "c2")),
                _claim("c3", "release", "status", "cancelled", 3),
            ]
        )
    )
    assert reopened.lifecycle_status_by_conflict_ref_and_scope[
        (conflict.conflict_ref, None)
    ] == "reopened"

    abstained = materialize(
        OpLog([*claims, _resolution(conflict, outcome="abstain", retained=("c1", "c2"))])
    )
    abstained_view = build_view(abstained, ViewConstraints())
    assert abstained.lifecycle_status_by_conflict_ref_and_scope[
        (conflict.conflict_ref, None)
    ] == "deferred"
    assert abstained_view.unresolved_conflicts == [abstained.conflicts[0]]

    stale_abstention = materialize(
        OpLog(
            [
                *claims,
                _resolution(conflict, outcome="abstain", retained=("c1", "c2")),
                _claim("c3", "release", "status", "cancelled", 3),
            ]
        )
    )
    assert stale_abstention.lifecycle_status_by_conflict_ref_and_scope[
        (conflict.conflict_ref, None)
    ] == "reopened"


def test_compaction_keeps_evidence_used_by_custom_detectors() -> None:
    claim_op = _claim("known", "task", "status", "ready", 1)
    evidence_op = EvidenceAdded(
        op_id="op-marker-evidence",
        replica_id="test",
        timestamp="2026-03-01T00:00:02Z",
        evidence=Evidence("marker", "memory://marker"),
    )

    def evidence_detector(context: ConflictDetectionContext) -> Iterable[ConflictSet]:
        if "marker" not in context.evidence_by_id:
            return ()
        claim = context.claims_by_id["known"]
        return (
            make_conflict(
                candidates=(claim,),
                key=claim.key,
                conflict_type="epistemic.marker",
                conflict_class="epistemic",
                conflict_subclass="marker",
                detector_id="marker/v1",
                reason="Standalone evidence marker is present.",
            ),
        )

    report = compact_oplog_with_report(
        OpLog([claim_op, evidence_op]), conflict_detectors=(evidence_detector,)
    )

    assert report.conflicts_equivalent
    assert report.projections_equivalent
    assert any(
        isinstance(op, EvidenceAdded) and op.evidence.evidence_id == "marker"
        for op in report.compacted.iter_ops()
    )
