from __future__ import annotations

from statefuse import Claim, ClaimKey, InMemoryStore, Memory, PredicateRegistry, materialize
from statefuse.conflict import build_direct_conflict_index, update_direct_conflict_index
from statefuse.model import ValidityInterval

KEY = ClaimKey("project", "deadline", "value")


def _claim(
    claim_id: str,
    value: str,
    *,
    context: dict[str, str | int | float] | None = None,
    valid_from: str | None = None,
    valid_until: str | None = None,
) -> Claim:
    return Claim(
        claim_id=claim_id,
        key=KEY,
        value=value,
        confidence=0.8,
        timestamp="2026-03-01T00:00:00.000000Z",
        provenance={"replica_id": "test"},
        context=context or {},
        validity=(
            ValidityInterval(valid_from=valid_from, valid_until=valid_until)
            if valid_from is not None or valid_until is not None
            else None
        ),
    )


def test_high_cardinality_add_compares_only_the_new_claim() -> None:
    registry = PredicateRegistry()
    existing = tuple(_claim(f"c-{index:05d}", "same") for index in range(20_000))
    index = build_direct_conflict_index(existing, registry)
    added = _claim("c-new", "different")

    updated, comparisons = update_direct_conflict_index(
        index,
        (*existing, added),
        registry,
    )

    assert comparisons == len(existing)
    assert len(updated.incompatible_pairs) == len(existing)
    assert updated.adjacent_claim_ids["c-new"] == frozenset(
        claim.claim_id for claim in existing
    )


def test_retraction_removes_only_adjacent_edges_without_semantic_comparisons() -> None:
    memory = Memory(store=InMemoryStore(), replica_id="test")
    for claim_id, value in (("c1", "one"), ("c2", "two"), ("c3", "three")):
        memory.add_claim(
            namespace=KEY.namespace,
            subject=KEY.subject,
            predicate=KEY.predicate,
            value=value,
            confidence=0.8,
            evidence_ids=(),
            claim_id=claim_id,
        )
    memory.materialize()
    cursor = memory.materialization_cursor
    assert cursor is not None

    memory.retract_claim(target_claim_id="c2", evidence_ids=(), reason="Withdrawn")
    incremental, delta = memory.materialize_with_delta(cursor)
    full = materialize(memory.load_oplog())
    index = incremental.direct_conflict_indexes[KEY]

    assert incremental == full
    assert delta.conflict_comparisons == 0
    assert set(index.incompatible_pairs) == {("c1", "c3")}
    assert "c2" not in index.claims_by_id
    assert "c2" not in index.adjacent_claim_ids["c1"]
    assert "c2" not in index.adjacent_claim_ids["c3"]


def test_context_validity_and_normalization_match_full_detection() -> None:
    registry = PredicateRegistry()
    registry.register("value", normalize=lambda value: str(value).casefold())
    memory = Memory(
        store=InMemoryStore(),
        replica_id="test",
        predicate_registry=registry,
        materialization_config_token="normalized-v1",
    )
    claims = (
        _claim(
            "c1",
            "ALPHA",
            context={"region": "north"},
            valid_from="2026-01-01T00:00:00.000000Z",
            valid_until="2026-04-01T00:00:00.000000Z",
        ),
        _claim(
            "c2",
            "alpha",
            context={"region": "north"},
            valid_from="2026-01-01T00:00:00.000000Z",
            valid_until="2026-04-01T00:00:00.000000Z",
        ),
        _claim(
            "c3",
            "BETA",
            context={"region": "south"},
            valid_from="2026-01-01T00:00:00.000000Z",
            valid_until="2026-04-01T00:00:00.000000Z",
        ),
        _claim(
            "c4",
            "GAMMA",
            context={"region": "north"},
            valid_from="2026-04-01T00:00:00.000000Z",
            valid_until="2026-05-01T00:00:00.000000Z",
        ),
        _claim(
            "c5",
            "DELTA",
            context={"region": "north"},
            valid_from="2026-02-01T00:00:00.000000Z",
            valid_until="2026-05-01T00:00:00.000000Z",
        ),
    )

    for index, claim in enumerate(claims):
        memory.add_claim(
            namespace=claim.key.namespace,
            subject=claim.key.subject,
            predicate=claim.key.predicate,
            value=claim.value,
            confidence=claim.confidence,
            evidence_ids=(),
            claim_id=claim.claim_id,
            context=claim.context,
            validity=claim.validity,
            op_id=f"op-{index}",
        )
        assert memory.materialize() == materialize(
            memory.load_oplog(),
            predicate_registry=registry,
        )

    conflict = memory.materialize().conflicts[0]
    assert {claim.claim_id for claim in conflict.candidates} == {"c1", "c2", "c4", "c5"}
    assert conflict.annotations["applicability"] == "aggregate"


def test_numeric_context_equality_matches_full_detection() -> None:
    memory = Memory(store=InMemoryStore(), replica_id="test")
    for claim in (
        _claim("c1", "alpha", context={"segment": 1}),
        _claim("c2", "beta", context={"segment": 1.0}),
    ):
        memory.add_claim(
            namespace=claim.key.namespace,
            subject=claim.key.subject,
            predicate=claim.key.predicate,
            value=claim.value,
            confidence=claim.confidence,
            evidence_ids=(),
            claim_id=claim.claim_id,
            context=claim.context,
            op_id=f"op-{claim.claim_id}",
        )
        memory.materialize()

    incremental = memory.materialize()
    full = materialize(memory.load_oplog())

    assert incremental == full
    assert "applicability" not in incremental.conflicts[0].annotations


def test_custom_comparator_uses_canonical_claim_order() -> None:
    registry = PredicateRegistry()
    registry.register("value", equal=lambda left, right: str(left) <= str(right))
    memory = Memory(
        store=InMemoryStore(),
        replica_id="test",
        predicate_registry=registry,
        materialization_config_token="ordered-comparator-v1",
    )
    for claim_id, value in (("c1", "alpha"), ("c2", "beta")):
        memory.add_claim(
            namespace=KEY.namespace,
            subject=KEY.subject,
            predicate=KEY.predicate,
            value=value,
            confidence=0.8,
            evidence_ids=(),
            claim_id=claim_id,
            op_id=f"op-{claim_id}",
        )
        memory.materialize()

    assert memory.materialize() == materialize(
        memory.load_oplog(),
        predicate_registry=registry,
    )
    assert memory.materialize().conflicts == []


def test_custom_comparator_does_not_invoke_an_unrelated_normalizer() -> None:
    registry = PredicateRegistry()
    registry.register(
        "value",
        normalize=lambda _value: (_ for _ in ()).throw(AssertionError("not used")),
        equal=lambda left, right: str(left).casefold() == str(right).casefold(),
    )
    index = build_direct_conflict_index((_claim("c1", "ALPHA"),), registry)

    updated, _ = update_direct_conflict_index(
        index, (_claim("c1", "ALPHA"), _claim("c2", "alpha")), registry
    )

    assert updated.incompatible_pairs == {}


def test_normalizer_shortcut_uses_canonical_claim_order() -> None:
    class DirectionalValue:
        def __init__(self, value: str) -> None:
            self.value = value

        def __eq__(self, other: object) -> bool:
            return isinstance(other, DirectionalValue) and self.value <= other.value

    registry = PredicateRegistry()
    registry.register("value", normalize=lambda value: DirectionalValue(str(value)))
    memory = Memory(
        store=InMemoryStore(),
        replica_id="test",
        predicate_registry=registry,
        materialization_config_token="ordered-normalizer-v1",
    )
    for claim_id, value in (("c1", "beta"), ("c2", "alpha")):
        memory.add_claim(
            namespace=KEY.namespace,
            subject=KEY.subject,
            predicate=KEY.predicate,
            value=value,
            confidence=0.8,
            evidence_ids=(),
            claim_id=claim_id,
            op_id=f"op-{claim_id}",
        )
        memory.materialize()

    assert memory.materialize() == materialize(
        memory.load_oplog(),
        predicate_registry=registry,
    )
    assert len(memory.materialize().conflicts) == 1


def test_checkpoint_restores_conflict_index_without_pair_redetection(tmp_path) -> None:
    from statefuse import SQLiteStore

    path = tmp_path / "ops.sqlite"
    memory = Memory(store=SQLiteStore(path), replica_id="test")
    for claim_id, value in (("c1", "one"), ("c2", "two"), ("c3", "three")):
        memory.add_claim(
            namespace=KEY.namespace,
            subject=KEY.subject,
            predicate=KEY.predicate,
            value=value,
            confidence=0.8,
            evidence_ids=(),
            claim_id=claim_id,
        )
    memory.materialize()

    restored = Memory(store=SQLiteStore(path), replica_id="test")
    restored.materialize()
    cursor = restored.materialization_cursor
    assert cursor is not None
    restored.add_claim(
        namespace=KEY.namespace,
        subject=KEY.subject,
        predicate=KEY.predicate,
        value="four",
        confidence=0.8,
        evidence_ids=(),
        claim_id="c4",
    )

    incremental, delta = restored.materialize_with_delta(cursor)

    assert incremental == materialize(restored.load_oplog())
    assert delta.conflict_comparisons == 3
