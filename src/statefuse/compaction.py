from __future__ import annotations

from dataclasses import dataclass

from .conflict import PredicateRegistry
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


def compact_oplog(oplog: OpLog) -> OpLog:
    return compact_oplog_with_report(oplog).compacted


def compact_oplog_with_report(oplog: OpLog) -> CompactionReport:
    return compact_projection_equivalent_with_report(oplog)


def _projection_signature(oplog: OpLog, *, predicate_registry: PredicateRegistry, constraints: ViewConstraints, resolver: Resolver) -> tuple[tuple[str, str], tuple[str, ...], tuple[str, ...], tuple[tuple[str, str], ...]]:
    state = materialize(oplog, predicate_registry=predicate_registry)
    projection = build_view(
        state=state,
        constraints=constraints,
        resolver=resolver,
    )
    selected = tuple(
        (f"{key.namespace}:{key.subject}:{key.predicate}", claim.claim_id)
        for key, claim in sorted(projection.selected_claims.items())
    )
    unresolved = tuple(conflict.conflict_id for conflict in projection.unresolved_conflicts)
    surfaced = tuple(
        conflict.conflict_id
        for _, conflict in sorted(
            projection.surfaced_conflicts.items(),
            key=lambda item: (item[0].namespace, item[0].subject, item[0].predicate),
        )
    )
    explanations = tuple(sorted(projection.explanations.items()))
    return selected, unresolved, surfaced, explanations


def compact_projection_equivalent(oplog: OpLog) -> OpLog:
    return compact_projection_equivalent_with_report(oplog).compacted


def compact_projection_equivalent_with_report(
    oplog: OpLog,
    *,
    predicate_registry: PredicateRegistry | None = None,
    constraints: ViewConstraints | None = None,
    resolver: Resolver | None = None,
) -> CompactionReport:
    registry = predicate_registry or PredicateRegistry()
    active_resolver = resolver or HeuristicResolver()
    active_constraints = constraints or ViewConstraints(scope="projection-equivalent-compaction")

    state = materialize(oplog, predicate_registry=registry)
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

    kept_ops = []
    for op in oplog.iter_ops():
        if isinstance(op, EvidenceAdded) and op.evidence.evidence_id in needed_evidence_ids:
            kept_ops.append(op)
            continue
        if isinstance(op, ClaimAdded) and op.claim.claim_id in active_claim_ids:
            kept_ops.append(op)
            continue
        if isinstance(op, ClaimRetracted) and op.op_id in retraction_op_ids:
            kept_ops.append(op)
            continue
        if isinstance(op, DecisionAdded) and op.decision.decision_id in active_decision_ids:
            kept_ops.append(op)

    compacted = OpLog(kept_ops)
    original_state = materialize(oplog, predicate_registry=registry)
    compacted_state = materialize(compacted, predicate_registry=registry)
    original_projection = _projection_signature(
        oplog,
        predicate_registry=registry,
        constraints=active_constraints,
        resolver=active_resolver,
    )
    compacted_projection = _projection_signature(
        compacted,
        predicate_registry=registry,
        constraints=active_constraints,
        resolver=active_resolver,
    )
    return CompactionReport(
        compacted=compacted,
        original_ops=len(oplog),
        compacted_ops=len(compacted),
        active_claims_equivalent=original_state.active_claims_by_key == compacted_state.active_claims_by_key,
        conflicts_equivalent=original_state.conflicts == compacted_state.conflicts,
        projections_equivalent=original_projection == compacted_projection,
    )
