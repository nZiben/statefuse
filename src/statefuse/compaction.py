from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .conflict import ConflictDetector, PredicateRegistry
from .materialize import materialize
from .oplog import OpLog
from .ops import ClaimAdded, ClaimRetracted, DecisionAdded, EvidenceAdded
from .resolver import HeuristicResolver, Resolver, ViewConstraints
from .view import build_view


@dataclass(frozen=True)
class CompactionReport:
    compacted: OpLog
    original_ops: int
    compacted_ops: int
    active_claims_equivalent: bool = True
    conflicts_equivalent: bool = True
    projections_equivalent: bool = True

    @property
    def dropped_ops(self) -> int:
        return self.original_ops - self.compacted_ops


def compact_oplog(
    oplog: OpLog,
    *,
    predicate_registry: PredicateRegistry | None = None,
    conflict_detectors: Sequence[ConflictDetector] = (),
) -> OpLog:
    return compact_oplog_with_report(
        oplog,
        predicate_registry=predicate_registry,
        conflict_detectors=conflict_detectors,
    ).compacted


def compact_oplog_with_report(
    oplog: OpLog,
    *,
    predicate_registry: PredicateRegistry | None = None,
    conflict_detectors: Sequence[ConflictDetector] = (),
) -> CompactionReport:
    return compact_projection_equivalent_with_report(
        oplog,
        predicate_registry=predicate_registry,
        conflict_detectors=conflict_detectors,
    )


def _projection_signature(
    oplog: OpLog,
    *,
    predicate_registry: PredicateRegistry,
    constraints: ViewConstraints,
    resolver: Resolver,
    conflict_detectors: Sequence[ConflictDetector],
) -> tuple[
    tuple[tuple[str, str], ...],
    tuple[tuple[str, tuple[str, ...]], ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[tuple[str, str], ...],
]:
    state = materialize(
        oplog,
        predicate_registry=predicate_registry,
        conflict_detectors=conflict_detectors,
        valid_at=constraints.valid_at,
        context=constraints.context,
    )
    projection = build_view(
        state=state,
        constraints=constraints,
        resolver=resolver,
    )
    selected = tuple(
        (f"{key.namespace}:{key.subject}:{key.predicate}", claim.claim_id)
        for key, claim in sorted(projection.selected_claims.items())
    )
    compatible = tuple(
        (
            f"{key.namespace}:{key.subject}:{key.predicate}",
            tuple(claim.claim_id for claim in claims),
        )
        for key, claims in sorted(projection.compatible_claims.items())
    )
    unresolved = tuple(conflict.conflict_id for conflict in projection.unresolved_conflicts)
    surfaced = tuple(
        conflict_id for conflict_id in sorted(projection.surfaced_findings)
    )
    explanations = tuple(sorted(projection.explanations.items()))
    return selected, compatible, unresolved, surfaced, explanations


def compact_projection_equivalent(
    oplog: OpLog,
    *,
    predicate_registry: PredicateRegistry | None = None,
    conflict_detectors: Sequence[ConflictDetector] = (),
) -> OpLog:
    return compact_projection_equivalent_with_report(
        oplog,
        predicate_registry=predicate_registry,
        conflict_detectors=conflict_detectors,
    ).compacted


def compact_projection_equivalent_with_report(
    oplog: OpLog,
    *,
    predicate_registry: PredicateRegistry | None = None,
    constraints: ViewConstraints | None = None,
    resolver: Resolver | None = None,
    conflict_detectors: Sequence[ConflictDetector] = (),
) -> CompactionReport:
    registry = predicate_registry or PredicateRegistry()
    active_resolver = resolver or HeuristicResolver()
    active_constraints = constraints or ViewConstraints(scope="projection-equivalent-compaction")

    state = materialize(
        oplog,
        predicate_registry=registry,
        conflict_detectors=conflict_detectors,
    )
    active_claim_ids = {
        claim.claim_id for claims in state.active_claims_by_key.values() for claim in claims
    }
    active_decision_ids = {decision.decision_id for decision in state.active_decisions}
    needed_evidence_ids = set()
    retraction_op_ids = set()

    for claims in state.active_claims_by_key.values():
        for claim in claims:
            needed_evidence_ids.update(claim.evidence_ids)
    for retractions in state.retractions_by_target.values():
        for retraction in retractions:
            retraction_op_ids.add(retraction.op_id)
            needed_evidence_ids.update(retraction.evidence_ids)
    for retractions in state.retractions_by_target_ref.values():
        for retraction in retractions:
            retraction_op_ids.add(retraction.op_id)
            needed_evidence_ids.update(retraction.evidence_ids)
    for resolution in state.resolutions_by_id.values():
        needed_evidence_ids.update(resolution.evidence_ids)

    kept_ops = []
    for op in oplog.iter_ops():
        if isinstance(op, EvidenceAdded):
            # ponytail: detectors are opaque; add declared dependencies before pruning evidence.
            if conflict_detectors or op.evidence.evidence_id in needed_evidence_ids:
                kept_ops.append(op)
        elif isinstance(op, ClaimAdded):
            if op.claim.claim_id in active_claim_ids:
                kept_ops.append(op)
        elif isinstance(op, ClaimRetracted):
            if op.op_id in retraction_op_ids:
                kept_ops.append(op)
        elif isinstance(op, DecisionAdded):
            if op.decision.decision_id in active_decision_ids:
                kept_ops.append(op)
        else:
            kept_ops.append(op)

    compacted = OpLog(kept_ops)
    original_state = materialize(
        oplog, predicate_registry=registry, conflict_detectors=conflict_detectors
    )
    compacted_state = materialize(
        compacted, predicate_registry=registry, conflict_detectors=conflict_detectors
    )
    original_projection = _projection_signature(
        oplog,
        predicate_registry=registry,
        constraints=active_constraints,
        resolver=active_resolver,
        conflict_detectors=conflict_detectors,
    )
    compacted_projection = _projection_signature(
        compacted,
        predicate_registry=registry,
        constraints=active_constraints,
        resolver=active_resolver,
        conflict_detectors=conflict_detectors,
    )
    return CompactionReport(
        compacted=compacted,
        original_ops=len(oplog),
        compacted_ops=len(compacted),
        active_claims_equivalent=(
            original_state.active_claims_by_key == compacted_state.active_claims_by_key
        ),
        conflicts_equivalent=original_state.conflicts == compacted_state.conflicts,
        projections_equivalent=original_projection == compacted_projection,
    )
