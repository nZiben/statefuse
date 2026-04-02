from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .conflict import ConflictSet
from .materialize import MemoryState
from .model import Claim
from .utils import parse_utc_iso


@dataclass(frozen=True)
class ViewConstraints:
    scope: str | None = None
    preferred_replica_ids: tuple[str, ...] = field(default_factory=tuple)
    preferred_branch_ids: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "preferred_replica_ids", tuple(self.preferred_replica_ids))
        object.__setattr__(self, "preferred_branch_ids", tuple(self.preferred_branch_ids))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass
class Resolution:
    chosen_claim_id: str | None
    reason: str
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Resolver(Protocol):
    def resolve(
        self,
        conflict: ConflictSet,
        constraints: ViewConstraints,
        state: MemoryState,
    ) -> Resolution:
        ...


class HeuristicResolver:
    """Deterministic conflict resolver for functional predicates."""

    _QUALITY_RANK = {
        "verified": 6,
        "primary": 5,
        "trusted": 5,
        "authoritative": 5,
        "secondary": 2,
        "plausible": 2,
        "unverified": 1,
        "stale": 0,
    }
    _FRESHNESS_RANK = {
        "current": 3,
        "recent": 2,
        "stale_cache": 0,
        "old": 0,
    }
    _SOURCE_RANK = {
        "verified_correction": 6,
        "official_record": 5,
        "verification_tool": 4,
        "trusted_api": 4,
        "peer_report": 2,
        "browser_summary": 1,
        "model_guess": 0,
        "draft_note": 0,
    }
    _TRUST_RANK = {
        "trusted": 2,
        "unknown": 1,
        "revoked": -2,
    }

    def resolve(
        self,
        conflict: ConflictSet,
        constraints: ViewConstraints,
        state: MemoryState,
    ) -> Resolution:
        if not conflict.candidates:
            return Resolution(chosen_claim_id=None, reason="No candidates available for conflict.")

        scored: list[tuple[Claim, tuple[int, float, float, int, int, int]]] = []
        for claim in conflict.candidates:
            scored.append((claim, self._score_claim(claim, constraints, state)))

        best_primary = max(score for _, score in scored)
        tied_claims = [claim for claim, score in scored if score == best_primary]
        chosen = min(tied_claims, key=lambda claim: claim.claim_id)

        score_trace = {claim.claim_id: score for claim, score in scored}
        reason = (
            f"Selected {chosen.claim_id} by deterministic heuristic "
            "("
            "supersedes recency, evidence quality, source type, freshness, trust, "
            "confidence, timestamp, evidence count, preferred provenance, claim_id"
            ")."
        )
        return Resolution(
            chosen_claim_id=chosen.claim_id,
            reason=reason,
            confidence=chosen.confidence,
            metadata={"scores": score_trace},
        )

    def _score_claim(
        self,
        claim: Claim,
        constraints: ViewConstraints,
        state: MemoryState,
    ) -> tuple[int, float, int, int, int, int, float, float, int, int]:
        supersedes_epoch = self._latest_supersede_epoch(claim.claim_id, state)
        has_supersede = 1 if supersedes_epoch is not None else 0
        supersedes_value = supersedes_epoch if supersedes_epoch is not None else -1.0
        evidence_quality, source_rank, freshness_rank, trust_rank = self._evidence_rankings(claim, state)
        claim_epoch = parse_utc_iso(claim.timestamp).timestamp()
        evidence_count = len(claim.evidence_ids)
        replica_score = self._preference_score(
            str(claim.provenance.get("replica_id", "")),
            constraints.preferred_replica_ids,
        )
        branch_score = self._preference_score(
            str(claim.provenance.get("branch_id", "")),
            constraints.preferred_branch_ids,
        )
        return (
            has_supersede,
            supersedes_value,
            evidence_quality,
            source_rank,
            freshness_rank,
            trust_rank,
            claim.confidence,
            claim_epoch,
            evidence_count,
            replica_score + branch_score,
        )

    def _evidence_rankings(self, claim: Claim, state: MemoryState) -> tuple[int, int, int, int]:
        best_quality = 0
        best_source = 0
        best_freshness = 0
        best_trust = 0
        for evidence_id in claim.evidence_ids:
            evidence = state.evidence_by_id.get(evidence_id)
            if evidence is None:
                continue
            metadata = evidence.metadata
            source_quality = str(metadata.get("source_quality", "")).strip().lower()
            source_type = str(metadata.get("source_type", "")).strip().lower()
            freshness = str(metadata.get("freshness", "")).strip().lower()
            trust = str(metadata.get("trust_status", "unknown")).strip().lower()
            best_quality = max(best_quality, self._QUALITY_RANK.get(source_quality, 0))
            best_source = max(best_source, self._SOURCE_RANK.get(source_type, 0))
            best_freshness = max(best_freshness, self._FRESHNESS_RANK.get(freshness, 0))
            best_trust = max(best_trust, self._TRUST_RANK.get(trust, 1))
        return best_quality, best_source, best_freshness, best_trust

    def _latest_supersede_epoch(self, claim_id: str, state: MemoryState) -> float | None:
        retractions = state.retractions_by_superseder.get(claim_id, [])
        if not retractions:
            return None
        return max(parse_utc_iso(retraction.timestamp).timestamp() for retraction in retractions)

    @staticmethod
    def _preference_score(value: str, preferred_values: tuple[str, ...]) -> int:
        if not value:
            return 0
        for index, preferred in enumerate(preferred_values):
            if value == preferred:
                return len(preferred_values) - index
        return 0


class ConservativeHeuristicResolver(HeuristicResolver):
    """
    Deterministic variant that abstains on close or symmetric multi-value conflicts.

    This is intended for benchmark surfaces where we want to measure the trade-off
    between aggressive automatic selection and fail-closed abstention.
    """

    def __init__(
        self,
        *,
        confidence_margin: float = 0.05,
        time_margin_seconds: float = 300.0,
        evidence_count_margin: int = 1,
    ) -> None:
        self.confidence_margin = confidence_margin
        self.time_margin_seconds = time_margin_seconds
        self.evidence_count_margin = evidence_count_margin

    def resolve(
        self,
        conflict: ConflictSet,
        constraints: ViewConstraints,
        state: MemoryState,
    ) -> Resolution:
        if not conflict.candidates:
            return Resolution(chosen_claim_id=None, reason="No candidates available for conflict.")

        scored: list[tuple[Claim, tuple[int, float, int, int, int, int, float, float, int, int]]] = []
        for claim in conflict.candidates:
            scored.append((claim, self._score_claim(claim, constraints, state)))

        best_score = max(score for _, score in scored)
        top_claims = [claim for claim, score in scored if score == best_score]
        if len({str(claim.value) for claim in top_claims}) > 1:
            return Resolution(
                chosen_claim_id=None,
                reason="Conservative heuristic abstained on an exact top-score tie across values.",
                metadata={"scores": {claim.claim_id: score for claim, score in scored}},
            )

        ordered = sorted(scored, key=lambda item: (item[1], item[0].claim_id), reverse=True)
        best_claim, best = ordered[0]
        runner_up: tuple[Claim, tuple[int, float, int, int, int, int, float, float, int, int]] | None = None
        for candidate, score in ordered[1:]:
            if str(candidate.value) != str(best_claim.value):
                runner_up = (candidate, score)
                break

        if runner_up is not None and self._is_close_competitor(best, runner_up[1]):
            return Resolution(
                chosen_claim_id=None,
                reason="Conservative heuristic abstained on a close competing value.",
                metadata={"scores": {claim.claim_id: score for claim, score in scored}},
            )

        return Resolution(
            chosen_claim_id=best_claim.claim_id,
            reason="Conservative heuristic selected the highest-ranked claim without a close competitor.",
            confidence=best_claim.confidence,
            metadata={"scores": {claim.claim_id: score for claim, score in scored}},
        )

    def _is_close_competitor(
        self,
        best: tuple[int, float, int, int, int, int, float, float, int, int],
        runner: tuple[int, float, int, int, int, int, float, float, int, int],
    ) -> bool:
        if best[:6] != runner[:6]:
            return False
        if abs(best[6] - runner[6]) > self.confidence_margin:
            return False
        if abs(best[7] - runner[7]) > self.time_margin_seconds:
            return False
        if abs(best[8] - runner[8]) > self.evidence_count_margin:
            return False
        return best[9] == runner[9]
