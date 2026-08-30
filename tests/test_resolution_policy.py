from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from statefuse import (
    Abstention,
    CausalResolver,
    Claim,
    ClaimKey,
    ConflictSet,
    LatestWriteWinsResolver,
    PreserveResolver,
    ResolutionContext,
    ResolverRegistry,
    SelectedState,
    UnresolvedConflict,
)


def _claim(
    claim_id: str,
    timestamp: str,
    *,
    branch_id: object = "main",
    vector_clock: object | None = None,
) -> Claim:
    provenance = {}
    if branch_id is not None:
        provenance["branch_id"] = branch_id
    if vector_clock is not None:
        provenance["vector_clock"] = vector_clock
    return Claim(
        claim_id=claim_id,
        key=ClaimKey(namespace="project", subject="item", predicate="status"),
        value=claim_id,
        confidence=0.8,
        timestamp=timestamp,
        provenance=provenance,
    )


def _conflict(
    *claims: Claim,
    conflict_type: str = "same_key_distinct_value",
) -> ConflictSet:
    return ConflictSet(
        conflict_id="conflict:test",
        key=claims[0].key,
        candidates=tuple(claims),
        distinct_values=tuple(claim.value for claim in claims),
        reason="test conflict",
        conflict_type=conflict_type,
    )


def _lww_conflict(**second_overrides: object) -> ConflictSet:
    second = {
        "timestamp": "2026-03-01T00:00:01Z",
        "branch_id": "main",
    }
    second.update(second_overrides)
    return _conflict(
        _claim("c1", "2026-03-01T00:00:00Z", branch_id="main"),
        _claim("c2", **second),  # type: ignore[arg-type]
    )


def test_resolution_context_is_immutable() -> None:
    context = ResolutionContext(timestamps_trusted=True, concurrent=False)
    with pytest.raises(FrozenInstanceError):
        context.concurrent = True  # type: ignore[misc]


def test_preserve_and_selected_results_retain_the_original_conflict() -> None:
    conflict = _lww_conflict()
    before = conflict.to_dict()

    preserved = PreserveResolver().resolve(conflict, ResolutionContext())
    selected = LatestWriteWinsResolver().resolve(
        conflict,
        ResolutionContext(
            timestamps_trusted=True,
            concurrent=False,
            replacement_semantics=True,
        ),
    )

    assert isinstance(preserved, UnresolvedConflict)
    assert isinstance(selected, SelectedState)
    assert preserved.conflict_set is conflict
    assert selected.conflict_set is conflict
    assert selected.selected_claim.claim_id == "c2"
    assert selected.audit.candidate_claim_ids == ("c1", "c2")
    assert conflict.to_dict() == before
    assert tuple(claim.claim_id for claim in conflict.candidates) == ("c1", "c2")


def test_selected_state_rejects_a_claim_outside_its_conflict() -> None:
    conflict = _lww_conflict()
    valid = LatestWriteWinsResolver().resolve(
        conflict,
        ResolutionContext(
            timestamps_trusted=True,
            concurrent=False,
            replacement_semantics=True,
        ),
    )
    assert isinstance(valid, SelectedState)

    with pytest.raises(ValueError, match="must belong"):
        SelectedState(
            conflict_set=conflict,
            selected_claim=_claim("foreign", "2026-03-01T00:00:02Z"),
            reason="invalid selection",
            audit=valid.audit,
        )


@pytest.mark.parametrize(
    "context,reason",
    [
        (
            ResolutionContext(
                timestamps_trusted=False,
                concurrent=False,
                replacement_semantics=True,
            ),
            "not explicitly trusted",
        ),
        (
            ResolutionContext(timestamps_trusted=True, replacement_semantics=True),
            "not explicitly established",
        ),
        (
            ResolutionContext(
                timestamps_trusted=True,
                concurrent=True,
                replacement_semantics=True,
            ),
            "Concurrent updates",
        ),
    ],
)
def test_latest_write_wins_requires_trust_and_explicit_non_concurrency(
    context: ResolutionContext,
    reason: str,
) -> None:
    result = LatestWriteWinsResolver().resolve(_lww_conflict(), context)
    assert isinstance(result, Abstention)
    assert reason in result.reason


@pytest.mark.parametrize(
    "conflict,reason",
    [
        (_lww_conflict(timestamp="2026-03-01T00:00:00Z"), "tie"),
        (_lww_conflict(branch_id="other"), "different branches"),
        (_lww_conflict(timestamp="not-a-timestamp"), "parseable timestamp"),
        (_lww_conflict(branch_id=None), "branch_id"),
    ],
)
def test_latest_write_wins_abstains_when_ordering_preconditions_fail(
    conflict: ConflictSet,
    reason: str,
) -> None:
    result = LatestWriteWinsResolver().resolve(
        conflict,
        ResolutionContext(
            timestamps_trusted=True,
            concurrent=False,
            replacement_semantics=True,
        ),
    )
    assert isinstance(result, Abstention)
    assert reason in result.reason


def test_resolvers_abstain_on_unaccepted_conflict_types() -> None:
    conflict = _conflict(
        _claim("c1", "2026-03-01T00:00:00Z", vector_clock={"a": 1}),
        _claim("c2", "2026-03-01T00:00:01Z", vector_clock={"a": 2}),
        conflict_type="semantic_contradiction",
    )
    context = ResolutionContext(timestamps_trusted=True, concurrent=False)

    assert isinstance(LatestWriteWinsResolver().resolve(conflict, context), Abstention)
    assert isinstance(CausalResolver().resolve(conflict, context), Abstention)


def test_resolvers_require_explicit_semantic_and_metadata_assertions() -> None:
    lww_conflict = _lww_conflict()
    lww = LatestWriteWinsResolver().resolve(
        lww_conflict,
        ResolutionContext(timestamps_trusted=True, concurrent=False),
    )
    causal_conflict = _conflict(
        _claim("c1", "2026-03-01T00:00:00Z", vector_clock={"a": 1}),
        _claim("c2", "2026-03-01T00:00:01Z", vector_clock={"a": 2}),
    )
    causal = CausalResolver().resolve(
        causal_conflict,
        ResolutionContext(replacement_semantics=True),
    )

    assert isinstance(lww, Abstention)
    assert "replacement semantics" in lww.reason
    assert isinstance(causal, Abstention)
    assert "not explicitly trusted" in causal.reason


def test_resolvers_abstain_when_a_conflict_has_fewer_than_two_candidates() -> None:
    conflict = _conflict(_claim("c1", "2026-03-01T00:00:00Z", vector_clock={"a": 1}))
    context = ResolutionContext(
        timestamps_trusted=True,
        concurrent=False,
        replacement_semantics=True,
        causal_metadata_trusted=True,
    )

    lww = LatestWriteWinsResolver().resolve(conflict, context)
    causal = CausalResolver().resolve(conflict, context)
    assert isinstance(lww, Abstention)
    assert isinstance(causal, Abstention)
    assert "at least two" in lww.reason
    assert "at least two" in causal.reason


def test_registry_preserves_unknown_types_and_rejects_unknown_explicit_names() -> None:
    registry = ResolverRegistry()
    conflict = _conflict(
        _claim("c1", "2026-03-01T00:00:00Z"),
        _claim("c2", "2026-03-01T00:00:01Z"),
        conflict_type="future_conflict_type",
    )

    assert isinstance(registry.resolve(conflict, ResolutionContext()), UnresolvedConflict)
    with pytest.raises(KeyError, match="Unknown resolver"):
        registry.resolve(conflict, ResolutionContext(), resolver_name="missing")
    with pytest.raises(KeyError, match="Unknown resolver"):
        registry.route("future_conflict_type", "missing")


def test_registry_rejects_a_plugin_result_for_a_different_conflict() -> None:
    requested = _lww_conflict()
    other = _conflict(
        _claim("other-1", "2026-03-01T00:00:00Z"),
        _claim("other-2", "2026-03-01T00:00:01Z"),
    )

    class WrongConflictResolver:
        def resolve(
            self,
            conflict_set: ConflictSet,
            context: ResolutionContext,
        ) -> UnresolvedConflict:
            del conflict_set
            return PreserveResolver().resolve(other, context)

    registry = ResolverRegistry()
    registry.register("wrong", WrongConflictResolver())
    with pytest.raises(ValueError, match="supplied ConflictSet"):
        registry.resolve(requested, ResolutionContext(), resolver_name="wrong")


def test_registry_routes_by_exact_conflict_type() -> None:
    registry = ResolverRegistry()
    conflict = _conflict(
        _claim("c1", "2026-03-01T00:00:00Z", vector_clock={"a": 1}),
        _claim("c2", "2026-03-01T00:00:01Z", vector_clock={"a": 2}),
        conflict_type="causal_update",
    )
    registry.register(
        "causal_update",
        CausalResolver(accepted_conflict_types=frozenset({"causal_update"})),
    )
    registry.route("causal_update", "causal_update")

    result = registry.resolve(
        conflict,
        ResolutionContext(replacement_semantics=True, causal_metadata_trusted=True),
    )
    assert isinstance(result, SelectedState)
    assert result.selected_claim.claim_id == "c2"


def test_causal_resolver_selects_only_a_unique_dominant_candidate() -> None:
    conflict = _conflict(
        _claim("c1", "2026-03-01T00:00:00Z", vector_clock={"a": 1, "b": 1}),
        _claim("c2", "2026-03-01T00:00:01Z", vector_clock={"a": 2, "b": 1}),
    )

    result = CausalResolver().resolve(
        conflict,
        ResolutionContext(replacement_semantics=True, causal_metadata_trusted=True),
    )
    assert isinstance(result, SelectedState)
    assert result.selected_claim.claim_id == "c2"
    assert result.conflict_set is conflict


def test_causal_resolver_preserves_concurrent_alternatives() -> None:
    conflict = _conflict(
        _claim("c1", "2026-03-01T00:00:00Z", vector_clock={"a": 1, "b": 0}),
        _claim("c2", "2026-03-01T00:00:01Z", vector_clock={"a": 0, "b": 1}),
    )

    result = CausalResolver().resolve(
        conflict,
        ResolutionContext(replacement_semantics=True, causal_metadata_trusted=True),
    )
    assert isinstance(result, UnresolvedConflict)
    assert result.conflict_set is conflict
    assert ("maximal_claim_ids", "c1,c2") in result.audit.details


@pytest.mark.parametrize("clock", [None, {}, {"a": -1}, {"a": True}, {1: 1}])
def test_causal_resolver_abstains_on_missing_or_invalid_clocks(clock: object) -> None:
    conflict = _conflict(
        _claim("c1", "2026-03-01T00:00:00Z", vector_clock=clock),
        _claim("c2", "2026-03-01T00:00:01Z", vector_clock={"a": 2}),
    )
    context = ResolutionContext(replacement_semantics=True, causal_metadata_trusted=True)
    assert isinstance(CausalResolver().resolve(conflict, context), Abstention)
