from __future__ import annotations

from dataclasses import dataclass, field

from .conflict import ConflictSet
from .materialize import MemoryState
from .model import Claim, ClaimKey
from .resolver import HeuristicResolver, Resolver, ViewConstraints
from .utils import parse_utc_iso


@dataclass
class Projection:
    selected_claims: dict[ClaimKey, Claim] = field(default_factory=dict)
    unresolved_conflicts: list[ConflictSet] = field(default_factory=list)
    # Public contract: surfaced conflicts remain visible even when a resolver picks a projection-time winner.
    surfaced_conflicts: dict[ClaimKey, ConflictSet] = field(default_factory=dict)
    explanations: dict[str, str] = field(default_factory=dict)


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


def build_view(
    state: MemoryState,
    constraints: ViewConstraints,
    resolver: Resolver | None = None,
) -> Projection:
    active_resolver = resolver or HeuristicResolver()
    conflicts_by_key = {conflict.key: conflict for conflict in state.conflicts}

    selected_claims: dict[ClaimKey, Claim] = {}
    unresolved_conflicts: list[ConflictSet] = []
    surfaced_conflicts: dict[ClaimKey, ConflictSet] = {}
    explanations: dict[str, str] = {}

    for key in sorted(state.active_claims_by_key):
        claims = state.active_claims_by_key[key]
        label = _key_label(key)
        conflict = conflicts_by_key.get(key)
        if conflict is None:
            chosen = _deterministic_claim_choice(claims)
            selected_claims[key] = chosen
            explanations[label] = f"No conflict. Selected {chosen.claim_id} deterministically."
            continue

        surfaced_conflicts[key] = conflict
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

    unresolved_conflicts.sort(key=lambda conflict: (conflict.key, conflict.conflict_id))
    return Projection(
        selected_claims=selected_claims,
        unresolved_conflicts=unresolved_conflicts,
        surfaced_conflicts=surfaced_conflicts,
        explanations=explanations,
    )
