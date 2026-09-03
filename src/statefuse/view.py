from __future__ import annotations

from dataclasses import dataclass, field

from .conflict import DIRECT_CONFLICT_TYPE, ConflictSet
from .materialize import MemoryState
from .model import Claim, ClaimKey
from .resolver import HeuristicResolver, Resolver, ViewConstraints
from .utils import parse_utc_iso


@dataclass
class Projection:
    selected_claims: dict[ClaimKey, Claim] = field(default_factory=dict)
    unresolved_conflicts: list[ConflictSet] = field(default_factory=list)
    # Public contract: surfaced conflicts remain visible even when a resolver picks a
    # projection-time winner.
    surfaced_conflicts: dict[ClaimKey, ConflictSet] = field(default_factory=dict)
    explanations: dict[str, str] = field(default_factory=dict)
    surfaced_findings: dict[str, ConflictSet] = field(default_factory=dict)
    compatible_claims: dict[ClaimKey, tuple[Claim, ...]] = field(default_factory=dict)


def _key_label(key: ClaimKey) -> str:
    return f"{key.namespace}:{key.subject}:{key.predicate}"


def _deterministic_claim_choice(candidates: list[Claim]) -> Claim:
    best = sorted(
        candidates,
        key=lambda claim: (
            claim.confidence,
            parse_utc_iso(claim.timestamp).timestamp(),
            len(claim.evidence_ids),
            claim.claim_id,
        ),
        reverse=True,
    )
    return best[0]


def _resolution_lane(
    state: MemoryState,
    conflict: ConflictSet,
    scope: str | None,
) -> tuple[str, str | None]:
    global_lane = (conflict.conflict_ref, None)
    if scope is None:
        return global_lane
    exact_lane = (conflict.conflict_ref, scope)
    if (
        exact_lane in state.resolutions_by_conflict_ref_and_scope
        or exact_lane in state.lifecycle_history_by_conflict_ref_and_scope
    ):
        return exact_lane
    return global_lane


def _committed_choice(
    state: MemoryState,
    conflict: ConflictSet,
    scope: str | None,
) -> tuple[Claim | None, str | None, bool]:
    lane = _resolution_lane(state, conflict, scope)
    resolution = state.effective_resolutions_by_conflict_ref_and_scope.get(lane)
    is_effective = resolution is not None
    if resolution is None:
        resolution = state.active_resolutions_by_conflict_ref_and_scope.get(lane)
        lane_status = state.lifecycle_status_by_conflict_ref_and_scope.get(lane)
        if lane_status == "deferred":
            detail = (
                f"Committed abstention {resolution.resolution_id}"
                if resolution is not None and resolution.outcome == "abstain"
                else "Conflict resolution is deferred"
            )
            return None, detail, False
        if resolution is None or lane_status != "reopened":
            return None, None, False

    candidates = {claim.claim_id: claim for claim in conflict.candidates}
    classified = (
        set(resolution.selected_claim_ids)
        | set(resolution.rejected_claim_ids)
        | set(resolution.retained_claim_ids)
    )
    uncovered = sorted(set(candidates) - classified)
    selected = sorted(set(candidates) & set(resolution.selected_claim_ids))
    if is_effective and resolution.outcome == "preserve":
        return (
            None,
            f"Applied committed {resolution.outcome} resolution {resolution.resolution_id}",
            True,
        )
    if is_effective and len(selected) == 1:
        return (
            candidates[selected[0]],
            f"Applied committed {resolution.outcome} resolution {resolution.resolution_id}",
            True,
        )
    if not is_effective or uncovered or len(selected) != 1:
        detail = f"Committed resolution {resolution.resolution_id} is stale/reopened"
        if uncovered:
            detail += f"; uncovered candidates: {', '.join(uncovered)}"
        if len(selected) != 1:
            detail += f"; current selected candidate count: {len(selected)}"
        if not is_effective and not uncovered and len(selected) == 1:
            detail += "; observed snapshot does not match classified claims"
        return None, detail, False
    return None, None, False


def _is_selectable_conflict(conflict: ConflictSet) -> bool:
    return (
        conflict.conflict_type == DIRECT_CONFLICT_TYPE
        and conflict.keys == (conflict.key,)
        and all(claim.key == conflict.key for claim in conflict.candidates)
    )


def build_view(
    state: MemoryState,
    constraints: ViewConstraints,
    resolver: Resolver | None = None,
) -> Projection:
    active_resolver = resolver or HeuristicResolver()
    conflicts_by_key = {
        key: tuple(conflict for conflict in conflicts if _is_selectable_conflict(conflict))
        for key, conflicts in state.conflicts_by_key.items()
    }

    selected_claims: dict[ClaimKey, Claim] = {}
    unresolved_conflicts: list[ConflictSet] = []
    surfaced_conflicts: dict[ClaimKey, ConflictSet] = {}
    surfaced_findings = {conflict.conflict_id: conflict for conflict in state.conflicts}
    compatible_claims: dict[ClaimKey, tuple[Claim, ...]] = {}
    explanations: dict[str, str] = {}
    handled_conflict_ids: set[str] = set()

    for key in sorted(state.active_claims_by_key):
        claims = state.active_claims_by_key[key]
        label = _key_label(key)
        key_conflicts = conflicts_by_key.get(key, ())
        if not key_conflicts:
            if len(claims) > 1 and any(
                not state.predicate_registry.values_equal(
                    key.predicate, claim.value, claims[0].value
                )
                for claim in claims[1:]
            ):
                compatible_claims[key] = tuple(claims)
                explanations[label] = (
                    "Compatible context, time, or multi-valued alternatives; "
                    "no single claim selected."
                )
                continue
            chosen = _deterministic_claim_choice(claims)
            selected_claims[key] = chosen
            explanations[label] = f"No conflict. Selected {chosen.claim_id} deterministically."
            continue

        conflict = key_conflicts[0]
        handled_conflict_ids.add(conflict.conflict_id)
        surfaced_conflicts[key] = conflict
        candidate_ids = {claim.claim_id for claim in conflict.candidates}
        compatible = tuple(claim for claim in claims if claim.claim_id not in candidate_ids)
        if compatible:
            compatible_claims[key] = compatible
        committed, committed_detail, committed_applied = _committed_choice(
            state,
            conflict,
            constraints.scope,
        )
        if committed_applied:
            if committed is not None:
                selected_claims[key] = committed
                explanations[label] = (
                    f"Resolved {conflict.conflict_id} -> {committed.claim_id}: "
                    f"{committed_detail}."
                )
            else:
                explanations[label] = f"Resolved {conflict.conflict_id}: {committed_detail}."
            continue
        if committed_detail is not None:
            unresolved_conflicts.append(conflict)
            explanations[label] = f"Unresolved {conflict.conflict_id}: {committed_detail}."
            continue

        resolution = active_resolver.resolve(conflict, constraints, state)
        chosen: Claim | None = None
        if resolution.chosen_claim_id:
            for candidate in conflict.candidates:
                if candidate.claim_id == resolution.chosen_claim_id:
                    chosen = candidate
                    break

        raw_response = resolution.metadata.get("raw_response")
        if chosen is None:
            unresolved_conflicts.append(conflict)
            detail = f"Unresolved {conflict.conflict_id}: {resolution.reason}"
            if raw_response is not None:
                detail += f" | raw_response={raw_response}"
            explanations[label] = detail
            continue

        selected_claims[key] = chosen
        detail = f"Resolved {conflict.conflict_id} -> {chosen.claim_id}: {resolution.reason}"
        if raw_response is not None:
            detail += f" | raw_response={raw_response}"
        explanations[label] = detail

    for conflict in state.conflicts:
        if conflict.conflict_id in handled_conflict_ids:
            continue
        committed, committed_detail, committed_applied = _committed_choice(
            state, conflict, constraints.scope
        )
        label = f"conflict:{conflict.conflict_id}"
        if committed_applied:
            if committed is not None:
                lane = _resolution_lane(state, conflict, constraints.scope)
                resolution = state.effective_resolutions_by_conflict_ref_and_scope[lane]
                for claim in conflict.candidates:
                    if claim.claim_id not in resolution.rejected_claim_ids:
                        continue
                    if selected_claims.get(claim.key) == claim:
                        selected_claims.pop(claim.key)
                        explanations[_key_label(claim.key)] = (
                            f"Excluded {claim.claim_id} by committed "
                            f"{resolution.outcome} resolution {resolution.resolution_id}."
                        )
                    alternatives = compatible_claims.get(claim.key)
                    if alternatives and claim in alternatives:
                        remaining = tuple(item for item in alternatives if item != claim)
                        if len(remaining) == 1:
                            compatible_claims.pop(claim.key)
                            selected_claims[claim.key] = remaining[0]
                        elif remaining:
                            compatible_claims[claim.key] = remaining
                        else:
                            compatible_claims.pop(claim.key)
            explanations[label] = f"Resolved {conflict.conflict_id}: {committed_detail}."
            continue
        unresolved_conflicts.append(conflict)
        explanations[label] = (
            f"Unresolved {conflict.conflict_id}: "
            f"{committed_detail or 'no committed resolution for this conflict type'}."
        )

    unresolved_conflicts = sorted(
        {conflict.conflict_id: conflict for conflict in unresolved_conflicts}.values(),
        key=lambda conflict: (conflict.key, conflict.conflict_id),
    )
    return Projection(
        selected_claims=selected_claims,
        unresolved_conflicts=unresolved_conflicts,
        surfaced_conflicts=surfaced_conflicts,
        surfaced_findings=surfaced_findings,
        explanations=explanations,
        compatible_claims=compatible_claims,
    )
