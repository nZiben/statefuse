from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .conflict import ConflictSet, PredicateRegistry, detect_conflicts
from .model import Claim, ClaimKey, Decision, Evidence
from .oplog import OpLog
from .ops import ClaimAdded, ClaimRetracted, DecisionAdded, EvidenceAdded
from .utils import parse_utc_iso


@dataclass
class MemoryState:
    evidence_by_id: dict[str, Evidence]
    active_claims_by_key: dict[ClaimKey, list[Claim]]
    active_decisions: list[Decision]
    conflicts: list[ConflictSet]
    claims_by_id: dict[str, Claim] = field(default_factory=dict, repr=False)
    claim_refs_by_id: dict[str, str] = field(default_factory=dict, repr=False)
    claim_ids_by_ref: dict[str, tuple[str, ...]] = field(default_factory=dict, repr=False)
    retractions_by_target: dict[str, list[ClaimRetracted]] = field(default_factory=dict, repr=False)
    retractions_by_target_ref: dict[str, list[ClaimRetracted]] = field(default_factory=dict, repr=False)
    retractions_by_superseder: dict[str, list[ClaimRetracted]] = field(default_factory=dict, repr=False)
    retractions_by_superseder_ref: dict[str, list[ClaimRetracted]] = field(default_factory=dict, repr=False)
    inactive_claim_ids: set[str] = field(default_factory=set, repr=False)


def _sort_decisions(decisions: list[Decision]) -> list[Decision]:
    return sorted(
        decisions,
        key=lambda decision: (
            parse_utc_iso(decision.timestamp).timestamp() if decision.timestamp else float("-inf"),
            decision.decision_id,
        ),
    )


def materialize(oplog: OpLog, predicate_registry: PredicateRegistry | None = None) -> MemoryState:
    registry = predicate_registry or PredicateRegistry()

    evidence_by_id: dict[str, Evidence] = {}
    claims_by_id: dict[str, Claim] = {}
    claim_refs_by_id: dict[str, str] = {}
    claim_ids_by_ref_raw: defaultdict[str, list[str]] = defaultdict(list)
    decisions_by_id: dict[str, Decision] = {}
    retractions_by_target: defaultdict[str, list[ClaimRetracted]] = defaultdict(list)
    retractions_by_target_ref: defaultdict[str, list[ClaimRetracted]] = defaultdict(list)
    retractions_by_superseder: defaultdict[str, list[ClaimRetracted]] = defaultdict(list)
    retractions_by_superseder_ref: defaultdict[str, list[ClaimRetracted]] = defaultdict(list)

    for op in oplog.iter_ops():
        if isinstance(op, EvidenceAdded):
            existing = evidence_by_id.get(op.evidence.evidence_id)
            if existing is not None and existing != op.evidence:
                raise ValueError(f"evidence_id collision with different payload: {op.evidence.evidence_id}")
            evidence_by_id[op.evidence.evidence_id] = op.evidence
            continue

        if isinstance(op, ClaimAdded):
            existing = claims_by_id.get(op.claim.claim_id)
            if existing is not None and existing != op.claim:
                raise ValueError(f"claim_id collision with different payload: {op.claim.claim_id}")
            claims_by_id[op.claim.claim_id] = op.claim
            claim_ref = registry.claim_ref_for_claim(op.claim)
            claim_refs_by_id[op.claim.claim_id] = claim_ref
            claim_ids_by_ref_raw[claim_ref].append(op.claim.claim_id)
            continue

        if isinstance(op, ClaimRetracted):
            if op.target_claim_id:
                retractions_by_target[op.target_claim_id].append(op)
            if op.target_claim_ref:
                retractions_by_target_ref[op.target_claim_ref].append(op)
            if op.supersedes_claim_id:
                retractions_by_superseder[op.supersedes_claim_id].append(op)
            if op.supersedes_claim_ref:
                retractions_by_superseder_ref[op.supersedes_claim_ref].append(op)
            continue

        if isinstance(op, DecisionAdded):
            existing = decisions_by_id.get(op.decision.decision_id)
            if existing is not None and existing != op.decision:
                raise ValueError(
                    f"decision_id collision with different payload: {op.decision.decision_id}"
                )
            decisions_by_id[op.decision.decision_id] = op.decision

    inactive_claim_ids = set(retractions_by_target)
    for claim_ref in retractions_by_target_ref:
        inactive_claim_ids.update(claim_ids_by_ref_raw.get(claim_ref, []))
    active_claims_by_key_raw: defaultdict[ClaimKey, list[Claim]] = defaultdict(list)
    for claim_id, claim in claims_by_id.items():
        if claim_id in inactive_claim_ids:
            continue
        active_claims_by_key_raw[claim.key].append(claim)

    active_claims_by_key: dict[ClaimKey, list[Claim]] = {}
    for key in sorted(active_claims_by_key_raw):
        active_claims_by_key[key] = sorted(
            active_claims_by_key_raw[key], key=lambda item: item.claim_id
        )

    for claim_id in sorted(retractions_by_target):
        retractions_by_target[claim_id].sort(key=lambda item: (item.timestamp, item.op_id))
    for claim_ref in sorted(retractions_by_target_ref):
        retractions_by_target_ref[claim_ref].sort(key=lambda item: (item.timestamp, item.op_id))
    for claim_id in sorted(retractions_by_superseder):
        retractions_by_superseder[claim_id].sort(key=lambda item: (item.timestamp, item.op_id))
    for claim_ref in sorted(retractions_by_superseder_ref):
        retractions_by_superseder_ref[claim_ref].sort(key=lambda item: (item.timestamp, item.op_id))

    conflicts = detect_conflicts(active_claims_by_key, registry)
    active_decisions = _sort_decisions(list(decisions_by_id.values()))
    claim_ids_by_ref = {
        claim_ref: tuple(sorted(claim_ids))
        for claim_ref, claim_ids in claim_ids_by_ref_raw.items()
    }

    return MemoryState(
        evidence_by_id=evidence_by_id,
        active_claims_by_key=active_claims_by_key,
        active_decisions=active_decisions,
        conflicts=conflicts,
        claims_by_id=claims_by_id,
        claim_refs_by_id=claim_refs_by_id,
        claim_ids_by_ref=claim_ids_by_ref,
        retractions_by_target=dict(retractions_by_target),
        retractions_by_target_ref=dict(retractions_by_target_ref),
        retractions_by_superseder=dict(retractions_by_superseder),
        retractions_by_superseder_ref=dict(retractions_by_superseder_ref),
        inactive_claim_ids=inactive_claim_ids,
    )
