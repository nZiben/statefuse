from __future__ import annotations

from statefuse.conflict import PredicateContractError, PredicateRegistry
from statefuse.materialize import materialize
from statefuse.merge import merge
from statefuse.model import Claim, ClaimKey
from statefuse.oplog import OpLog
from statefuse.ops import ClaimAdded, ClaimRetracted


def _claim(op_id: str, claim_id: str, predicate: str, value: str, replica: str) -> ClaimAdded:
    return ClaimAdded(
        op_id=op_id,
        replica_id=replica,
        timestamp=f"2026-03-01T00:00:0{op_id[-1]}.000000Z",
        claim=Claim(
            claim_id=claim_id,
            key=ClaimKey(namespace="project", subject="item", predicate=predicate),
            value=value,
            confidence=0.7,
            timestamp=f"2026-03-01T00:00:0{op_id[-1]}.000000Z",
            evidence_ids=(),
            provenance={"replica_id": replica},
        ),
    )


def test_functional_predicate_with_distinct_values_creates_conflict() -> None:
    oplog = OpLog(
        [
            _claim("op-1", "c1", "status", "open", "a"),
            _claim("op-2", "c2", "status", "closed", "b"),
        ]
    )
    state = materialize(oplog)
    assert len(state.conflicts) == 1
    conflict = state.conflicts[0]
    assert conflict.key.predicate == "status"
    assert [claim.claim_id for claim in conflict.candidates] == ["c1", "c2"]


def test_multi_valued_predicate_avoids_conflict() -> None:
    registry = PredicateRegistry()
    registry.register("tag", multi_valued=True)
    oplog = OpLog(
        [
            _claim("op-1", "c1", "tag", "alpha", "a"),
            _claim("op-2", "c2", "tag", "beta", "b"),
        ]
    )
    state = materialize(oplog, predicate_registry=registry)
    assert state.conflicts == []


def test_conflicts_are_deterministic_across_merge_order() -> None:
    left = OpLog([_claim("op-1", "c1", "status", "open", "a")])
    right = OpLog([_claim("op-2", "c2", "status", "closed", "b")])

    state_ab = materialize(merge(left, right))
    state_ba = materialize(merge(right, left))

    assert len(state_ab.conflicts) == 1
    assert len(state_ba.conflicts) == 1
    assert state_ab.conflicts[0].conflict_id == state_ba.conflicts[0].conflict_id
    assert [claim.claim_id for claim in state_ab.conflicts[0].candidates] == ["c1", "c2"]
    assert [claim.claim_id for claim in state_ba.conflicts[0].candidates] == ["c1", "c2"]


def test_retraction_before_claim_arrival_keeps_claim_inactive() -> None:
    oplog = OpLog(
        [
            ClaimRetracted(
                op_id="op-r1",
                replica_id="a",
                timestamp="2026-03-01T00:00:00.000000Z",
                target_claim_id="c-late",
                evidence_ids=(),
                reason="invalidate pending claim",
                supersedes_claim_id=None,
            ),
            _claim("op-2", "c-late", "status", "open", "a"),
        ]
    )
    state = materialize(oplog)
    key = ClaimKey(namespace="project", subject="item", predicate="status")
    assert key not in state.active_claims_by_key


def test_claim_ref_retraction_keeps_late_arriving_claim_inactive() -> None:
    late_claim = _claim("op-2", "c-late", "status", "open", "a")
    oplog = OpLog(
        [
            ClaimRetracted(
                op_id="op-r1",
                replica_id="a",
                timestamp="2026-03-01T00:00:00.000000Z",
                target_claim_ref=late_claim.claim.claim_ref,
                evidence_ids=(),
                reason="invalidate by semantic handle",
                supersedes_claim_id=None,
            ),
            late_claim,
        ]
    )
    state = materialize(oplog)
    key = ClaimKey(namespace="project", subject="item", predicate="status")
    assert key not in state.active_claims_by_key
    assert late_claim.claim.claim_id in state.inactive_claim_ids


def test_predicate_contract_normalization_can_merge_equivalent_values() -> None:
    registry = PredicateRegistry()
    registry.register(
        "deadline",
        normalize=lambda value: str(value).strip().lower(),
        normalize_for_claim_ref=True,
    )
    oplog = OpLog(
        [
            _claim("op-1", "c1", "deadline", " 2026-03-25 ", "a"),
            _claim("op-2", "c2", "deadline", "2026-03-25", "b"),
        ]
    )
    state = materialize(oplog, predicate_registry=registry)
    key = ClaimKey(namespace="project", subject="item", predicate="deadline")
    assert state.conflicts == []
    refs = {state.claim_refs_by_id[claim.claim_id] for claim in state.active_claims_by_key[key]}
    assert len(refs) == 1


def test_predicate_contract_validation_rejects_nondeterministic_normalize() -> None:
    registry = PredicateRegistry()
    counter = {"calls": 0}

    def unstable(value: object) -> str:
        counter["calls"] += 1
        return f"{value}:{counter['calls']}"

    registry.register("status", normalize=unstable)
    try:
        registry.validate_contract("status", ["open"])
    except PredicateContractError as exc:
        assert "not deterministic" in str(exc)
    else:
        raise AssertionError("Expected PredicateContractError for unstable normalization.")
