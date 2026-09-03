from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .conflict import (
    DIRECT_CONFLICT_TYPE,
    ConflictDetectionContext,
    ConflictDetector,
    ConflictSet,
    DirectConflictIndex,
    PredicateRegistry,
    build_direct_conflict_index,
    claim_applies,
    conflict_from_direct_index,
    derive_conflict_id,
    run_conflict_detectors,
    update_direct_conflict_index,
)
from .model import (
    Claim,
    ClaimKey,
    ConflictLifecycleEvent,
    Decision,
    Derivation,
    Evidence,
    JSONValue,
    ResolutionRecord,
    Source,
)
from .oplog import OpLog
from .ops import (
    ClaimAdded,
    ClaimRetracted,
    ConflictLifecycleEventAdded,
    DecisionAdded,
    DerivationAdded,
    EvidenceAdded,
    ResolutionAdded,
    SourceAdded,
)
from .utils import canonical_json_dumps, canonical_json_loads, parse_utc_iso

MATERIALIZATION_CHECKPOINT_VERSION = "4"


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
    retractions_by_target_ref: dict[str, list[ClaimRetracted]] = field(
        default_factory=dict, repr=False
    )
    retractions_by_superseder: dict[str, list[ClaimRetracted]] = field(
        default_factory=dict, repr=False
    )
    retractions_by_superseder_ref: dict[str, list[ClaimRetracted]] = field(
        default_factory=dict, repr=False
    )
    inactive_claim_ids: set[str] = field(default_factory=set, repr=False)
    sources_by_id: dict[str, Source] = field(default_factory=dict)
    derivations_by_id: dict[str, Derivation] = field(default_factory=dict)
    resolutions_by_id: dict[str, ResolutionRecord] = field(default_factory=dict, repr=False)
    lifecycle_events_by_id: dict[str, ConflictLifecycleEvent] = field(
        default_factory=dict, repr=False
    )
    resolutions_by_conflict_ref: dict[str, tuple[ResolutionRecord, ...]] = field(
        default_factory=dict
    )
    resolutions_by_conflict_ref_and_scope: dict[
        tuple[str, str | None], tuple[ResolutionRecord, ...]
    ] = field(default_factory=dict, repr=False)
    lifecycle_history_by_conflict_ref: dict[str, tuple[ConflictLifecycleEvent, ...]] = field(
        default_factory=dict
    )
    lifecycle_history_by_conflict_ref_and_scope: dict[
        tuple[str, str | None], tuple[ConflictLifecycleEvent, ...]
    ] = field(default_factory=dict, repr=False)
    effective_resolutions_by_conflict_ref_and_scope: dict[
        tuple[str, str | None], ResolutionRecord
    ] = field(default_factory=dict, repr=False)
    active_resolutions_by_conflict_ref_and_scope: dict[tuple[str, str | None], ResolutionRecord] = (
        field(default_factory=dict, repr=False)
    )
    lifecycle_status_by_conflict_ref_and_scope: dict[tuple[str, str | None], str] = field(
        default_factory=dict
    )
    conflicts_by_id: dict[str, ConflictSet] = field(default_factory=dict, repr=False)
    conflicts_by_ref: dict[str, ConflictSet] = field(default_factory=dict, repr=False)
    conflicts_by_key: dict[ClaimKey, tuple[ConflictSet, ...]] = field(default_factory=dict)
    inapplicable_claim_ids: set[str] = field(default_factory=set, repr=False)
    claim_ids_by_evidence_id: dict[str, tuple[str, ...]] = field(
        default_factory=dict, repr=False
    )
    claim_ids_by_derivation_id: dict[str, tuple[str, ...]] = field(
        default_factory=dict, repr=False
    )
    evidence_ids_by_source_id: dict[str, tuple[str, ...]] = field(
        default_factory=dict, repr=False
    )
    resolution_ids_by_claim_id: dict[str, tuple[str, ...]] = field(
        default_factory=dict, repr=False
    )
    direct_conflict_indexes: dict[ClaimKey, DirectConflictIndex] = field(
        default_factory=dict, repr=False, compare=False
    )
    predicate_registry: PredicateRegistry = field(
        default_factory=PredicateRegistry, repr=False, compare=False
    )

    def find_conflicts(
        self,
        *,
        conflict_id: str | None = None,
        conflict_ref: str | None = None,
        conflict_type: str | None = None,
        conflict_class: str | None = None,
        conflict_subclass: str | None = None,
        detector_id: str | None = None,
        claim_id: str | None = None,
        source_id: str | None = None,
        source_type: str | None = None,
        key: ClaimKey | None = None,
        namespace: str | None = None,
        context: Mapping[str, JSONValue] | None = None,
        status: str | None = None,
        scope: str | None = None,
    ) -> tuple[ConflictSet, ...]:
        matches: list[ConflictSet] = []
        for conflict in self.conflicts:
            if conflict_id is not None and conflict.conflict_id != conflict_id:
                continue
            if conflict_ref is not None and conflict.conflict_ref != conflict_ref:
                continue
            if conflict_type is not None and conflict.conflict_type != conflict_type:
                continue
            if conflict_class is not None and conflict.conflict_class != conflict_class:
                continue
            if conflict_subclass is not None and conflict.conflict_subclass != conflict_subclass:
                continue
            if detector_id is not None and conflict.detector_id != detector_id:
                continue
            if claim_id is not None and all(
                claim.claim_id != claim_id for claim in conflict.candidates
            ):
                continue
            if key is not None and key not in conflict.keys:
                continue
            if namespace is not None and all(
                related_key.namespace != namespace for related_key in conflict.keys
            ):
                continue
            candidate_source_ids = {
                evidence.source_id
                for claim in conflict.candidates
                for evidence_id in claim.evidence_ids
                if (evidence := self.evidence_by_id.get(evidence_id)) is not None
                and evidence.source_id is not None
            }
            if source_id is not None and source_id not in candidate_source_ids:
                continue
            if source_type is not None and all(
                self.sources_by_id.get(candidate_source_id) is None
                or self.sources_by_id[candidate_source_id].source_type != source_type
                for candidate_source_id in candidate_source_ids
            ):
                continue
            if context and not _conflict_matches_context(conflict, context):
                continue
            lane = (conflict.conflict_ref, scope)
            if scope is not None and lane not in self.lifecycle_status_by_conflict_ref_and_scope:
                lane = (conflict.conflict_ref, None)
            current_status = self.lifecycle_status_by_conflict_ref_and_scope.get(lane, "open")
            if status is not None and current_status != status:
                continue
            matches.append(conflict)
        return tuple(sorted(matches, key=lambda item: item.conflict_id))


@dataclass(frozen=True)
class MaterializationDelta:
    processed_op_ids: tuple[str, ...] = ()
    affected_keys: tuple[ClaimKey, ...] = ()
    affected_namespaces: tuple[str, ...] = ()
    affected_claim_ids: tuple[str, ...] = ()
    previous_conflict_ids: tuple[str, ...] = ()
    current_conflict_ids: tuple[str, ...] = ()
    affected_conflict_refs: tuple[str, ...] = ()
    affected_resolution_ids: tuple[str, ...] = ()
    cursor_before: int = 0
    cursor_after: int = 0
    full_rebuild: bool = False
    conflict_comparisons: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.processed_op_ids and not self.full_rebuild


def _conflict_matches_context(conflict: ConflictSet, context: Mapping[str, JSONValue]) -> bool:
    pairs = conflict.witness.get("incompatible_pairs")
    if conflict.conflict_type == DIRECT_CONFLICT_TYPE and isinstance(pairs, list):
        return any(
            isinstance(pair, dict)
            and isinstance(pair.get("context"), dict)
            and all(pair["context"].get(name, value) == value for name, value in context.items())
            for pair in pairs
        )
    return all(
        all(claim.context.get(name, value) == value for name, value in context.items())
        for claim in conflict.candidates
    )


def _sort_decisions(decisions: list[Decision]) -> list[Decision]:
    return sorted(
        decisions,
        key=lambda decision: (
            parse_utc_iso(decision.timestamp).timestamp() if decision.timestamp else float("-inf"),
            decision.decision_id,
        ),
    )


def _timestamp(value: str) -> datetime:
    return parse_utc_iso(value)


def _lane_sort_key(lane: tuple[str, str | None]) -> tuple[str, int, str]:
    conflict_ref, scope = lane
    return conflict_ref, 0 if scope is None else 1, scope or ""


def _group_resolution_history(
    resolutions_by_id: dict[str, ResolutionRecord],
) -> tuple[
    dict[str, tuple[ResolutionRecord, ...]],
    dict[tuple[str, str | None], tuple[ResolutionRecord, ...]],
]:
    by_ref: defaultdict[str, list[ResolutionRecord]] = defaultdict(list)
    by_lane: defaultdict[tuple[str, str | None], list[ResolutionRecord]] = defaultdict(list)
    for resolution in resolutions_by_id.values():
        by_ref[resolution.conflict_ref].append(resolution)
        by_lane[(resolution.conflict_ref, resolution.scope)].append(resolution)
    return (
        {
            key: tuple(
                sorted(
                    by_ref[key],
                    key=lambda item: (_timestamp(item.timestamp), item.resolution_id),
                )
            )
            for key in sorted(by_ref)
        },
        {
            key: tuple(
                sorted(
                    by_lane[key],
                    key=lambda item: (_timestamp(item.timestamp), item.resolution_id),
                )
            )
            for key in sorted(by_lane, key=_lane_sort_key)
        },
    )


def _group_lifecycle_history(
    events_by_id: dict[str, ConflictLifecycleEvent],
) -> tuple[
    dict[str, tuple[ConflictLifecycleEvent, ...]],
    dict[tuple[str, str | None], tuple[ConflictLifecycleEvent, ...]],
]:
    by_ref: defaultdict[str, list[ConflictLifecycleEvent]] = defaultdict(list)
    by_lane: defaultdict[tuple[str, str | None], list[ConflictLifecycleEvent]] = defaultdict(list)
    for event in events_by_id.values():
        by_ref[event.conflict_ref].append(event)
        by_lane[(event.conflict_ref, event.scope)].append(event)
    return (
        {
            key: tuple(
                sorted(
                    by_ref[key],
                    key=lambda item: (_timestamp(item.timestamp), item.event_id),
                )
            )
            for key in sorted(by_ref)
        },
        {
            key: tuple(
                sorted(
                    by_lane[key],
                    key=lambda item: (_timestamp(item.timestamp), item.event_id),
                )
            )
            for key in sorted(by_lane, key=_lane_sort_key)
        },
    )


def _fold_lifecycle(
    *,
    resolutions_by_id: dict[str, ResolutionRecord],
    resolutions_by_lane: dict[tuple[str, str | None], tuple[ResolutionRecord, ...]],
    events_by_lane: dict[tuple[str, str | None], tuple[ConflictLifecycleEvent, ...]],
    conflicts_by_ref: dict[str, ConflictSet],
    valid_at: str | None = None,
) -> tuple[
    dict[tuple[str, str | None], ResolutionRecord],
    dict[tuple[str, str | None], ResolutionRecord],
    dict[tuple[str, str | None], str],
]:
    active: dict[tuple[str, str | None], ResolutionRecord] = {}
    statuses: dict[tuple[str, str | None], str] = {}
    lanes = set(resolutions_by_lane) | set(events_by_lane)

    for lane in sorted(lanes, key=_lane_sort_key):
        timeline: list[tuple[datetime, int, str, ResolutionRecord | ConflictLifecycleEvent]] = []
        timeline.extend(
            (_timestamp(item.timestamp), 0, item.resolution_id, item)
            for item in resolutions_by_lane.get(lane, ())
            if _resolution_applies(item, valid_at)
        )
        timeline.extend(
            (_timestamp(item.timestamp), 1, item.event_id, item)
            for item in events_by_lane.get(lane, ())
        )
        for _, kind, _, item in sorted(timeline, key=lambda entry: entry[:3]):
            if kind == 0:
                resolution = item
                assert isinstance(resolution, ResolutionRecord)
                active[lane] = resolution
                statuses[lane] = "deferred" if resolution.outcome == "abstain" else "resolved"
                continue

            event = item
            assert isinstance(event, ConflictLifecycleEvent)
            if event.status != "resolved":
                active.pop(lane, None)
                statuses[lane] = event.status
                continue
            resolution = resolutions_by_id.get(event.resolution_id or "")
            if (
                resolution is None
                or resolution.conflict_ref != event.conflict_ref
                or resolution.scope != event.scope
                or resolution.observed_conflict_id != event.observed_conflict_id
                or not _resolution_applies(resolution, valid_at)
            ):
                active.pop(lane, None)
                statuses[lane] = "open"
                continue
            if resolution.outcome == "abstain":
                active[lane] = resolution
                statuses[lane] = "deferred"
                continue
            active[lane] = resolution
            statuses[lane] = "resolved"

    effective: dict[tuple[str, str | None], ResolutionRecord] = {}
    for conflict_ref, conflict in sorted(conflicts_by_ref.items()):
        global_lane = (conflict_ref, None)
        statuses.setdefault(global_lane, "open")
        current_ids = {claim.claim_id for claim in conflict.candidates}
        conflict_lanes = {global_lane} | {lane for lane in lanes if lane[0] == conflict_ref}
        for lane in sorted(conflict_lanes, key=_lane_sort_key):
            resolution = active.get(lane)
            if resolution is None:
                continue
            classified = (
                set(resolution.selected_claim_ids)
                | set(resolution.rejected_claim_ids)
                | set(resolution.retained_claim_ids)
            )
            selected = current_ids & set(resolution.selected_claim_ids)
            observed_matches = resolution.observed_conflict_id == derive_conflict_id(
                conflict.key,
                classified,
                conflict_type=conflict.conflict_type,
                keys=conflict.keys,
                detector_id=conflict.detector_id,
            )
            if resolution.outcome == "abstain":
                statuses[lane] = (
                    "deferred" if observed_matches and current_ids <= classified else "reopened"
                )
                continue
            preserves = (
                resolution.outcome == "preserve"
                and not resolution.selected_claim_ids
                and not resolution.rejected_claim_ids
                and current_ids <= set(resolution.retained_claim_ids)
            )
            selects = resolution.outcome in {"select", "replace", "merge"} and len(selected) == 1
            if observed_matches and current_ids <= classified and (preserves or selects):
                effective[lane] = resolution
                statuses[lane] = "resolved"
            else:
                statuses[lane] = "reopened"

    return active, effective, statuses


def _resolution_applies(resolution: ResolutionRecord, valid_at: str | None) -> bool:
    if valid_at is None:
        return True
    instant = parse_utc_iso(valid_at)
    valid_from = parse_utc_iso(resolution.valid_from) if resolution.valid_from else None
    valid_until = parse_utc_iso(resolution.valid_until) if resolution.valid_until else None
    return (valid_from is None or valid_from <= instant) and (
        valid_until is None or instant < valid_until
    )


def materialize(
    oplog: OpLog,
    predicate_registry: PredicateRegistry | None = None,
    *,
    conflict_detectors: Sequence[ConflictDetector] = (),
    valid_at: str | None = None,
    context: Mapping[str, JSONValue] | None = None,
) -> MemoryState:
    registry = predicate_registry or PredicateRegistry()
    if valid_at is not None:
        parse_utc_iso(valid_at)

    sources_by_id: dict[str, Source] = {}
    evidence_by_id: dict[str, Evidence] = {}
    claims_by_id: dict[str, Claim] = {}
    claim_refs_by_id: dict[str, str] = {}
    claim_ids_by_ref_raw: defaultdict[str, set[str]] = defaultdict(set)
    decisions_by_id: dict[str, Decision] = {}
    derivations_by_id: dict[str, Derivation] = {}
    resolutions_by_id: dict[str, ResolutionRecord] = {}
    lifecycle_events_by_id: dict[str, ConflictLifecycleEvent] = {}
    retractions_by_target: defaultdict[str, list[ClaimRetracted]] = defaultdict(list)
    retractions_by_target_ref: defaultdict[str, list[ClaimRetracted]] = defaultdict(list)
    retractions_by_superseder: defaultdict[str, list[ClaimRetracted]] = defaultdict(list)
    retractions_by_superseder_ref: defaultdict[str, list[ClaimRetracted]] = defaultdict(list)

    for op in oplog.iter_ops():
        if isinstance(op, SourceAdded):
            existing = sources_by_id.get(op.source.source_id)
            if existing is not None and existing != op.source:
                raise ValueError(
                    f"source_id collision with different payload: {op.source.source_id}"
                )
            sources_by_id[op.source.source_id] = op.source
            continue

        if isinstance(op, EvidenceAdded):
            existing = evidence_by_id.get(op.evidence.evidence_id)
            if existing is not None and existing != op.evidence:
                raise ValueError(
                    f"evidence_id collision with different payload: {op.evidence.evidence_id}"
                )
            evidence_by_id[op.evidence.evidence_id] = op.evidence
            continue

        if isinstance(op, ClaimAdded):
            existing = claims_by_id.get(op.claim.claim_id)
            if existing is not None and existing != op.claim:
                raise ValueError(f"claim_id collision with different payload: {op.claim.claim_id}")
            claims_by_id[op.claim.claim_id] = op.claim
            claim_ref = registry.claim_ref_for_claim(op.claim)
            claim_refs_by_id[op.claim.claim_id] = claim_ref
            claim_ids_by_ref_raw[claim_ref].add(op.claim.claim_id)
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
            continue

        if isinstance(op, DerivationAdded):
            existing = derivations_by_id.get(op.derivation.derivation_id)
            if existing is not None and existing != op.derivation:
                raise ValueError(
                    f"derivation_id collision with different payload: {op.derivation.derivation_id}"
                )
            derivations_by_id[op.derivation.derivation_id] = op.derivation
            continue

        if isinstance(op, ResolutionAdded):
            existing = resolutions_by_id.get(op.resolution.resolution_id)
            if existing is not None and existing != op.resolution:
                raise ValueError(
                    f"resolution_id collision with different payload: {op.resolution.resolution_id}"
                )
            resolutions_by_id[op.resolution.resolution_id] = op.resolution
            continue

        if isinstance(op, ConflictLifecycleEventAdded):
            existing = lifecycle_events_by_id.get(op.event.event_id)
            if existing is not None and existing != op.event:
                raise ValueError(f"event_id collision with different payload: {op.event.event_id}")
            lifecycle_events_by_id[op.event.event_id] = op.event

    inactive_claim_ids = set(retractions_by_target)
    for claim_ref in retractions_by_target_ref:
        inactive_claim_ids.update(claim_ids_by_ref_raw.get(claim_ref, []))
    active_claims_by_key_raw: defaultdict[ClaimKey, list[Claim]] = defaultdict(list)
    inapplicable_claim_ids: set[str] = set()
    for claim_id, claim in claims_by_id.items():
        if claim_id in inactive_claim_ids:
            continue
        if not claim_applies(claim, valid_at=valid_at, context=context):
            inapplicable_claim_ids.add(claim_id)
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

    active_decisions = _sort_decisions(list(decisions_by_id.values()))
    detection_context = ConflictDetectionContext(
        active_claims_by_key=active_claims_by_key,
        claims_by_id={
            claim.claim_id: claim for claims in active_claims_by_key.values() for claim in claims
        },
        evidence_by_id=evidence_by_id,
        sources_by_id=sources_by_id,
        derivations_by_id=derivations_by_id,
        active_decisions=tuple(active_decisions),
        predicate_registry=registry,
    )
    conflicts = run_conflict_detectors(detection_context, conflict_detectors)
    conflicts_by_id = {conflict.conflict_id: conflict for conflict in conflicts}
    conflicts_by_ref = {conflict.conflict_ref: conflict for conflict in conflicts}
    conflicts_by_key_raw: defaultdict[ClaimKey, list[ConflictSet]] = defaultdict(list)
    for conflict in conflicts:
        for related_key in conflict.keys:
            conflicts_by_key_raw[related_key].append(conflict)
    conflicts_by_key = {
        key: tuple(sorted(items, key=lambda conflict: conflict.conflict_id))
        for key, items in sorted(conflicts_by_key_raw.items())
    }
    direct_conflicts_by_key = {
        conflict.key: conflict
        for conflict in conflicts
        if conflict.conflict_type == DIRECT_CONFLICT_TYPE
        and conflict.detector_id == "direct"
        and conflict.keys == (conflict.key,)
    }
    direct_conflict_indexes = {
        key: build_direct_conflict_index(
            claims,
            registry,
            conflict=direct_conflicts_by_key.get(key),
        )
        for key, claims in active_claims_by_key.items()
    }
    claim_ids_by_ref = {
        claim_ref: tuple(sorted(claim_ids)) for claim_ref, claim_ids in claim_ids_by_ref_raw.items()
    }
    claim_ids_by_evidence_id = _claim_ids_by_evidence_id(claims_by_id.values())
    claim_ids_by_derivation_id = _claim_ids_by_derivation_id(claims_by_id.values())
    evidence_ids_by_source_id = _evidence_ids_by_source_id(evidence_by_id.values())
    resolution_ids_by_claim_id = _resolution_ids_by_claim_id(resolutions_by_id.values())
    resolutions_by_conflict_ref, resolutions_by_lane = _group_resolution_history(resolutions_by_id)
    lifecycle_history_by_conflict_ref, lifecycle_history_by_lane = _group_lifecycle_history(
        lifecycle_events_by_id
    )
    active_resolutions, effective_resolutions, lifecycle_statuses = _fold_lifecycle(
        resolutions_by_id=resolutions_by_id,
        resolutions_by_lane=resolutions_by_lane,
        events_by_lane=lifecycle_history_by_lane,
        conflicts_by_ref=conflicts_by_ref,
        valid_at=valid_at,
    )

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
        inapplicable_claim_ids=inapplicable_claim_ids,
        sources_by_id={key: sources_by_id[key] for key in sorted(sources_by_id)},
        derivations_by_id={key: derivations_by_id[key] for key in sorted(derivations_by_id)},
        resolutions_by_id={key: resolutions_by_id[key] for key in sorted(resolutions_by_id)},
        lifecycle_events_by_id={
            key: lifecycle_events_by_id[key] for key in sorted(lifecycle_events_by_id)
        },
        resolutions_by_conflict_ref=resolutions_by_conflict_ref,
        resolutions_by_conflict_ref_and_scope=resolutions_by_lane,
        lifecycle_history_by_conflict_ref=lifecycle_history_by_conflict_ref,
        lifecycle_history_by_conflict_ref_and_scope=lifecycle_history_by_lane,
        effective_resolutions_by_conflict_ref_and_scope=effective_resolutions,
        active_resolutions_by_conflict_ref_and_scope=active_resolutions,
        lifecycle_status_by_conflict_ref_and_scope=lifecycle_statuses,
        conflicts_by_id=conflicts_by_id,
        conflicts_by_ref=conflicts_by_ref,
        conflicts_by_key=conflicts_by_key,
        claim_ids_by_evidence_id=claim_ids_by_evidence_id,
        claim_ids_by_derivation_id=claim_ids_by_derivation_id,
        evidence_ids_by_source_id=evidence_ids_by_source_id,
        resolution_ids_by_claim_id=resolution_ids_by_claim_id,
        direct_conflict_indexes=direct_conflict_indexes,
        predicate_registry=registry,
    )


def apply_operations_incrementally(
    state: MemoryState,
    ops: Sequence[Any],
    *,
    conflict_detectors: Sequence[ConflictDetector] = (),
) -> tuple[MemoryState, MaterializationDelta]:
    """Apply unseen operations to an unscoped canonical state.

    Custom detectors are opaque and may depend on any record, so callers must use full
    materialization when they are configured.
    """

    if conflict_detectors:
        raise ValueError("Incremental materialization does not support custom detectors.")
    if not ops:
        return state, MaterializationDelta()

    registry = state.predicate_registry
    sources_by_id = dict(state.sources_by_id)
    evidence_by_id = dict(state.evidence_by_id)
    claims_by_id = dict(state.claims_by_id)
    claim_refs_by_id = dict(state.claim_refs_by_id)
    claim_ids_by_ref = {key: set(value) for key, value in state.claim_ids_by_ref.items()}
    decisions_by_id = {item.decision_id: item for item in state.active_decisions}
    derivations_by_id = dict(state.derivations_by_id)
    resolutions_by_id = dict(state.resolutions_by_id)
    lifecycle_events_by_id = dict(state.lifecycle_events_by_id)
    retractions_by_target = {key: list(value) for key, value in state.retractions_by_target.items()}
    retractions_by_target_ref = {
        key: list(value) for key, value in state.retractions_by_target_ref.items()
    }
    retractions_by_superseder = {
        key: list(value) for key, value in state.retractions_by_superseder.items()
    }
    retractions_by_superseder_ref = {
        key: list(value) for key, value in state.retractions_by_superseder_ref.items()
    }
    inactive_claim_ids = set(state.inactive_claim_ids)
    inapplicable_claim_ids = set(state.inapplicable_claim_ids)
    active_claims_by_key = {key: list(value) for key, value in state.active_claims_by_key.items()}
    resolutions_by_ref = dict(state.resolutions_by_conflict_ref)
    resolutions_by_lane = dict(state.resolutions_by_conflict_ref_and_scope)
    lifecycle_by_ref = dict(state.lifecycle_history_by_conflict_ref)
    lifecycle_by_lane = dict(state.lifecycle_history_by_conflict_ref_and_scope)
    claim_ids_by_evidence_id = {
        key: set(value) for key, value in state.claim_ids_by_evidence_id.items()
    }
    claim_ids_by_derivation_id = {
        key: set(value) for key, value in state.claim_ids_by_derivation_id.items()
    }
    evidence_ids_by_source_id = {
        key: set(value) for key, value in state.evidence_ids_by_source_id.items()
    }
    resolution_ids_by_claim_id = {
        key: set(value) for key, value in state.resolution_ids_by_claim_id.items()
    }
    direct_conflict_indexes = dict(state.direct_conflict_indexes)

    affected_keys: set[ClaimKey] = set()
    affected_claim_ids: set[str] = set()
    affected_conflict_refs: set[str] = set()
    affected_resolution_ids: set[str] = set()

    for op in ops:
        if isinstance(op, SourceAdded):
            inserted = _insert_immutable(
                sources_by_id, op.source.source_id, op.source, "source_id"
            )
            if inserted:
                for evidence_id in evidence_ids_by_source_id.get(op.source.source_id, set()):
                    _mark_claim_keys(
                        claim_ids_by_evidence_id.get(evidence_id, set()),
                        claims_by_id,
                        affected_keys,
                    )
            continue
        if isinstance(op, EvidenceAdded):
            inserted = _insert_immutable(
                evidence_by_id, op.evidence.evidence_id, op.evidence, "evidence_id"
            )
            if inserted:
                if op.evidence.source_id is not None:
                    evidence_ids_by_source_id.setdefault(op.evidence.source_id, set()).add(
                        op.evidence.evidence_id
                    )
                _mark_claim_keys(
                    claim_ids_by_evidence_id.get(op.evidence.evidence_id, set()),
                    claims_by_id,
                    affected_keys,
                )
            continue
        if isinstance(op, ClaimAdded):
            claim = op.claim
            inserted = _insert_immutable(claims_by_id, claim.claim_id, claim, "claim_id")
            if not inserted:
                continue
            claim_ref = registry.claim_ref_for_claim(claim)
            claim_refs_by_id[claim.claim_id] = claim_ref
            claim_ids_by_ref.setdefault(claim_ref, set()).add(claim.claim_id)
            for evidence_id in claim.evidence_ids:
                claim_ids_by_evidence_id.setdefault(evidence_id, set()).add(claim.claim_id)
            if claim.derivation_id is not None:
                claim_ids_by_derivation_id.setdefault(claim.derivation_id, set()).add(
                    claim.claim_id
                )
            affected_keys.add(claim.key)
            affected_claim_ids.add(claim.claim_id)
            affected_resolution_ids.update(
                resolution_ids_by_claim_id.get(claim.claim_id, set())
            )
            if claim.claim_id in retractions_by_target or claim_ref in retractions_by_target_ref:
                inactive_claim_ids.add(claim.claim_id)
            else:
                active_claims_by_key.setdefault(claim.key, []).append(claim)
            continue
        if isinstance(op, ClaimRetracted):
            if op.target_claim_id:
                _append_sorted_retraction(retractions_by_target, op.target_claim_id, op)
                inactive_claim_ids.add(op.target_claim_id)
                affected_claim_ids.add(op.target_claim_id)
            if op.target_claim_ref:
                _append_sorted_retraction(retractions_by_target_ref, op.target_claim_ref, op)
            if op.supersedes_claim_id:
                _append_sorted_retraction(retractions_by_superseder, op.supersedes_claim_id, op)
            if op.supersedes_claim_ref:
                _append_sorted_retraction(
                    retractions_by_superseder_ref, op.supersedes_claim_ref, op
                )
            targets: set[str] = set()
            if op.target_claim_id:
                targets.add(op.target_claim_id)
            if op.target_claim_ref:
                targets.update(claim_ids_by_ref.get(op.target_claim_ref, set()))
            for claim_id in targets:
                claim = claims_by_id.get(claim_id)
                if claim is None:
                    continue
                inactive_claim_ids.add(claim_id)
                inapplicable_claim_ids.discard(claim_id)
                affected_keys.add(claim.key)
                affected_claim_ids.add(claim_id)
                active_claims_by_key[claim.key] = [
                    item
                    for item in active_claims_by_key.get(claim.key, [])
                    if item.claim_id != claim_id
                ]
            continue
        if isinstance(op, DecisionAdded):
            _insert_immutable(decisions_by_id, op.decision.decision_id, op.decision, "decision_id")
            continue
        if isinstance(op, DerivationAdded):
            inserted = _insert_immutable(
                derivations_by_id,
                op.derivation.derivation_id,
                op.derivation,
                "derivation_id",
            )
            if inserted:
                _mark_claim_keys(
                    claim_ids_by_derivation_id.get(op.derivation.derivation_id, set()),
                    claims_by_id,
                    affected_keys,
                )
            continue
        if isinstance(op, ResolutionAdded):
            resolution = op.resolution
            inserted = _insert_immutable(
                resolutions_by_id,
                resolution.resolution_id,
                resolution,
                "resolution_id",
            )
            if inserted:
                affected_conflict_refs.add(resolution.conflict_ref)
                affected_resolution_ids.add(resolution.resolution_id)
                for claim_id in (
                    *resolution.selected_claim_ids,
                    *resolution.rejected_claim_ids,
                    *resolution.retained_claim_ids,
                ):
                    resolution_ids_by_claim_id.setdefault(claim_id, set()).add(
                        resolution.resolution_id
                    )
                _append_resolution_history(resolutions_by_ref, resolutions_by_lane, resolution)
            continue
        if isinstance(op, ConflictLifecycleEventAdded):
            event = op.event
            inserted = _insert_immutable(lifecycle_events_by_id, event.event_id, event, "event_id")
            if inserted:
                affected_conflict_refs.add(event.conflict_ref)
                _append_lifecycle_history(lifecycle_by_ref, lifecycle_by_lane, event)

    for key in tuple(active_claims_by_key):
        claims = sorted(active_claims_by_key[key], key=lambda item: item.claim_id)
        if claims:
            active_claims_by_key[key] = claims
        else:
            active_claims_by_key.pop(key)

    conflicts_by_id = dict(state.conflicts_by_id)
    conflicts_by_ref = dict(state.conflicts_by_ref)
    conflicts_by_key = dict(state.conflicts_by_key)
    previous_conflict_ids: set[str] = set()
    current_conflict_ids: set[str] = set()
    active_claims_by_id = {
        claim.claim_id: claim
        for claims in active_claims_by_key.values()
        for claim in claims
    }
    active_decisions = tuple(_sort_decisions(list(decisions_by_id.values())))
    conflict_comparisons = 0

    for key in sorted(affected_keys):
        previous = tuple(
            conflict
            for conflict in conflicts_by_key.get(key, ())
            if conflict.detector_id == "direct" and conflict.keys == (key,)
        )
        previous_conflict_ids.update(item.conflict_id for item in previous)
        affected_claim_ids.update(
            claim.claim_id for conflict in previous for claim in conflict.candidates
        )
        for conflict in previous:
            conflicts_by_id.pop(conflict.conflict_id, None)
            conflicts_by_ref.pop(conflict.conflict_ref, None)
            affected_conflict_refs.add(conflict.conflict_ref)

        active_for_key = {key: active_claims_by_key.get(key, [])}
        detection_context = ConflictDetectionContext(
            active_claims_by_key=active_for_key,
            claims_by_id=active_claims_by_id,
            evidence_by_id=evidence_by_id,
            sources_by_id=sources_by_id,
            derivations_by_id=derivations_by_id,
            active_decisions=active_decisions,
            predicate_registry=registry,
        )
        previous_index = direct_conflict_indexes.get(key)
        if previous_index is None:
            previous_index = build_direct_conflict_index(
                state.active_claims_by_key.get(key, ()),
                registry,
                conflict=previous[0] if previous else None,
            )
        current_index, comparisons = update_direct_conflict_index(
            previous_index,
            active_for_key[key],
            registry,
        )
        conflict_comparisons += comparisons
        current_conflict = conflict_from_direct_index(current_index, detection_context, key)
        current = (current_conflict,) if current_conflict is not None else ()
        if current_index.claims_by_id:
            direct_conflict_indexes[key] = current_index
        else:
            direct_conflict_indexes.pop(key, None)
        current_conflict_ids.update(item.conflict_id for item in current)
        affected_claim_ids.update(
            claim.claim_id for conflict in current for claim in conflict.candidates
        )
        for conflict in current:
            conflicts_by_id[conflict.conflict_id] = conflict
            conflicts_by_ref[conflict.conflict_ref] = conflict
            affected_conflict_refs.add(conflict.conflict_ref)
        if current:
            conflicts_by_key[key] = current
        else:
            conflicts_by_key.pop(key, None)

    conflicts = sorted(conflicts_by_id.values(), key=lambda item: item.conflict_id)
    active_resolutions = dict(state.active_resolutions_by_conflict_ref_and_scope)
    effective_resolutions = dict(state.effective_resolutions_by_conflict_ref_and_scope)
    lifecycle_statuses = dict(state.lifecycle_status_by_conflict_ref_and_scope)
    for conflict_ref in sorted(affected_conflict_refs):
        _replace_lifecycle_fold_for_ref(
            conflict_ref=conflict_ref,
            resolutions_by_id=resolutions_by_id,
            resolutions_by_ref=resolutions_by_ref,
            lifecycle_by_ref=lifecycle_by_ref,
            conflicts_by_ref=conflicts_by_ref,
            active=active_resolutions,
            effective=effective_resolutions,
            statuses=lifecycle_statuses,
        )
        affected_resolution_ids.update(
            item.resolution_id for item in resolutions_by_ref.get(conflict_ref, ())
        )

    result = MemoryState(
        evidence_by_id=evidence_by_id,
        active_claims_by_key={
            key: active_claims_by_key[key] for key in sorted(active_claims_by_key)
        },
        active_decisions=list(active_decisions),
        conflicts=conflicts,
        claims_by_id=claims_by_id,
        claim_refs_by_id=claim_refs_by_id,
        claim_ids_by_ref={
            key: tuple(sorted(value)) for key, value in sorted(claim_ids_by_ref.items())
        },
        retractions_by_target=retractions_by_target,
        retractions_by_target_ref=retractions_by_target_ref,
        retractions_by_superseder=retractions_by_superseder,
        retractions_by_superseder_ref=retractions_by_superseder_ref,
        inactive_claim_ids=inactive_claim_ids,
        inapplicable_claim_ids=inapplicable_claim_ids,
        sources_by_id={key: sources_by_id[key] for key in sorted(sources_by_id)},
        derivations_by_id={key: derivations_by_id[key] for key in sorted(derivations_by_id)},
        resolutions_by_id={key: resolutions_by_id[key] for key in sorted(resolutions_by_id)},
        lifecycle_events_by_id={
            key: lifecycle_events_by_id[key] for key in sorted(lifecycle_events_by_id)
        },
        resolutions_by_conflict_ref=resolutions_by_ref,
        resolutions_by_conflict_ref_and_scope=resolutions_by_lane,
        lifecycle_history_by_conflict_ref=lifecycle_by_ref,
        lifecycle_history_by_conflict_ref_and_scope=lifecycle_by_lane,
        effective_resolutions_by_conflict_ref_and_scope=effective_resolutions,
        active_resolutions_by_conflict_ref_and_scope=active_resolutions,
        lifecycle_status_by_conflict_ref_and_scope=lifecycle_statuses,
        conflicts_by_id=conflicts_by_id,
        conflicts_by_ref=conflicts_by_ref,
        conflicts_by_key=conflicts_by_key,
        claim_ids_by_evidence_id={
            key: tuple(sorted(value)) for key, value in sorted(claim_ids_by_evidence_id.items())
        },
        claim_ids_by_derivation_id={
            key: tuple(sorted(value))
            for key, value in sorted(claim_ids_by_derivation_id.items())
        },
        evidence_ids_by_source_id={
            key: tuple(sorted(value)) for key, value in sorted(evidence_ids_by_source_id.items())
        },
        resolution_ids_by_claim_id={
            key: tuple(sorted(value))
            for key, value in sorted(resolution_ids_by_claim_id.items())
        },
        direct_conflict_indexes=direct_conflict_indexes,
        predicate_registry=registry,
    )
    namespaces = {key.namespace for key in affected_keys}
    return result, MaterializationDelta(
        processed_op_ids=tuple(op.op_id for op in ops),
        affected_keys=tuple(sorted(affected_keys)),
        affected_namespaces=tuple(sorted(namespaces)),
        affected_claim_ids=tuple(sorted(affected_claim_ids)),
        previous_conflict_ids=tuple(sorted(previous_conflict_ids)),
        current_conflict_ids=tuple(sorted(current_conflict_ids)),
        affected_conflict_refs=tuple(sorted(affected_conflict_refs)),
        affected_resolution_ids=tuple(sorted(affected_resolution_ids)),
        conflict_comparisons=conflict_comparisons,
    )


def memory_state_to_checkpoint(state: MemoryState) -> str:
    retractions = {
        item.op_id: item
        for groups in (
            state.retractions_by_target.values(),
            state.retractions_by_target_ref.values(),
            state.retractions_by_superseder.values(),
            state.retractions_by_superseder_ref.values(),
        )
        for group in groups
        for item in group
    }
    return canonical_json_dumps(
        {
            "sources": [item.to_dict() for item in state.sources_by_id.values()],
            "evidence": [item.to_dict() for item in state.evidence_by_id.values()],
            "claims": [item.to_dict() for item in state.claims_by_id.values()],
            "retractions": [item.to_dict() for item in retractions.values()],
            "decisions": [item.to_dict() for item in state.active_decisions],
            "derivations": [item.to_dict() for item in state.derivations_by_id.values()],
            "resolutions": [item.to_dict() for item in state.resolutions_by_id.values()],
            "lifecycle_events": [item.to_dict() for item in state.lifecycle_events_by_id.values()],
            "conflicts": [item.to_dict() for item in state.conflicts],
            "inactive_claim_ids": sorted(state.inactive_claim_ids),
            "inapplicable_claim_ids": sorted(state.inapplicable_claim_ids),
        }
    )


def memory_state_from_checkpoint(
    payload: str,
    *,
    predicate_registry: PredicateRegistry,
) -> MemoryState:
    value = canonical_json_loads(payload)
    if not isinstance(value, dict):
        raise ValueError("Materialization checkpoint must contain an object.")
    sources = [Source.from_dict(item) for item in _checkpoint_list(value, "sources")]
    evidence = [Evidence.from_dict(item) for item in _checkpoint_list(value, "evidence")]
    claims = [Claim.from_dict(item) for item in _checkpoint_list(value, "claims")]
    decisions = [Decision.from_dict(item) for item in _checkpoint_list(value, "decisions")]
    derivations = [Derivation.from_dict(item) for item in _checkpoint_list(value, "derivations")]
    resolutions = [
        ResolutionRecord.from_dict(item) for item in _checkpoint_list(value, "resolutions")
    ]
    lifecycle_events = [
        ConflictLifecycleEvent.from_dict(item)
        for item in _checkpoint_list(value, "lifecycle_events")
    ]
    retractions = []
    for item in _checkpoint_list(value, "retractions"):
        op = ClaimRetracted._from_dict(item)
        retractions.append(op)

    claims_by_id = {item.claim_id: item for item in claims}
    claim_refs_by_id = {
        item.claim_id: predicate_registry.claim_ref_for_claim(item) for item in claims
    }
    claim_ids_by_ref_raw: defaultdict[str, set[str]] = defaultdict(set)
    for claim_id, claim_ref in claim_refs_by_id.items():
        claim_ids_by_ref_raw[claim_ref].add(claim_id)
    inactive_claim_ids = _checkpoint_string_set(value, "inactive_claim_ids")
    inapplicable_claim_ids = _checkpoint_string_set(value, "inapplicable_claim_ids")
    active_claims_raw: defaultdict[ClaimKey, list[Claim]] = defaultdict(list)
    for claim in claims:
        if claim.claim_id not in inactive_claim_ids | inapplicable_claim_ids:
            active_claims_raw[claim.key].append(claim)
    active_claims_by_key = {
        key: sorted(items, key=lambda item: item.claim_id)
        for key, items in sorted(active_claims_raw.items())
    }

    conflicts: list[ConflictSet] = []
    for item in _checkpoint_list(value, "conflicts"):
        candidate_ids = item.get("candidate_claim_ids")
        if not isinstance(candidate_ids, list):
            raise ValueError("Checkpoint conflict candidates must be a list.")
        conflicts.append(
            ConflictSet(
                conflict_id=str(item["conflict_id"]),
                conflict_ref=str(item["conflict_ref"]),
                conflict_type=str(item["conflict_type"]),
                conflict_class=str(item["conflict_class"]),
                conflict_subclass=str(item["conflict_subclass"]),
                detector_id=str(item["detector_id"]),
                key=ClaimKey.from_dict(item["key"]),
                keys=tuple(ClaimKey.from_dict(key) for key in item.get("keys", [])),
                candidates=tuple(claims_by_id[str(claim_id)] for claim_id in candidate_ids),
                distinct_values=tuple(item.get("distinct_values", [])),
                reason=str(item["reason"]),
                annotations=dict(item.get("annotations", {})),
                witness=dict(item.get("witness", {})),
            )
        )
    conflicts.sort(key=lambda item: item.conflict_id)
    conflicts_by_id = {item.conflict_id: item for item in conflicts}
    conflicts_by_ref = {item.conflict_ref: item for item in conflicts}
    conflicts_by_key_raw: defaultdict[ClaimKey, list[ConflictSet]] = defaultdict(list)
    for conflict in conflicts:
        for key in conflict.keys:
            conflicts_by_key_raw[key].append(conflict)
    direct_conflicts_by_key = {
        conflict.key: conflict
        for conflict in conflicts
        if conflict.conflict_type == DIRECT_CONFLICT_TYPE
        and conflict.detector_id == "direct"
        and conflict.keys == (conflict.key,)
    }
    direct_conflict_indexes = {
        key: build_direct_conflict_index(
            claims_for_key,
            predicate_registry,
            conflict=direct_conflicts_by_key.get(key),
        )
        for key, claims_for_key in active_claims_by_key.items()
    }

    retractions_by_target: defaultdict[str, list[ClaimRetracted]] = defaultdict(list)
    retractions_by_target_ref: defaultdict[str, list[ClaimRetracted]] = defaultdict(list)
    retractions_by_superseder: defaultdict[str, list[ClaimRetracted]] = defaultdict(list)
    retractions_by_superseder_ref: defaultdict[str, list[ClaimRetracted]] = defaultdict(list)
    for item in retractions:
        if item.target_claim_id:
            retractions_by_target[item.target_claim_id].append(item)
        if item.target_claim_ref:
            retractions_by_target_ref[item.target_claim_ref].append(item)
        if item.supersedes_claim_id:
            retractions_by_superseder[item.supersedes_claim_id].append(item)
        if item.supersedes_claim_ref:
            retractions_by_superseder_ref[item.supersedes_claim_ref].append(item)
    for mapping in (
        retractions_by_target,
        retractions_by_target_ref,
        retractions_by_superseder,
        retractions_by_superseder_ref,
    ):
        for items in mapping.values():
            items.sort(key=lambda item: (item.timestamp, item.op_id))

    resolutions_by_id = {item.resolution_id: item for item in resolutions}
    lifecycle_events_by_id = {item.event_id: item for item in lifecycle_events}
    claim_ids_by_evidence_id = _claim_ids_by_evidence_id(claims)
    claim_ids_by_derivation_id = _claim_ids_by_derivation_id(claims)
    evidence_ids_by_source_id = _evidence_ids_by_source_id(evidence)
    resolution_ids_by_claim_id = _resolution_ids_by_claim_id(resolutions)
    resolutions_by_ref, resolutions_by_lane = _group_resolution_history(resolutions_by_id)
    lifecycle_by_ref, lifecycle_by_lane = _group_lifecycle_history(lifecycle_events_by_id)
    active, effective, statuses = _fold_lifecycle(
        resolutions_by_id=resolutions_by_id,
        resolutions_by_lane=resolutions_by_lane,
        events_by_lane=lifecycle_by_lane,
        conflicts_by_ref=conflicts_by_ref,
    )
    return MemoryState(
        evidence_by_id={item.evidence_id: item for item in evidence},
        active_claims_by_key=active_claims_by_key,
        active_decisions=_sort_decisions(decisions),
        conflicts=conflicts,
        claims_by_id=claims_by_id,
        claim_refs_by_id=claim_refs_by_id,
        claim_ids_by_ref={
            key: tuple(sorted(items)) for key, items in sorted(claim_ids_by_ref_raw.items())
        },
        retractions_by_target=dict(retractions_by_target),
        retractions_by_target_ref=dict(retractions_by_target_ref),
        retractions_by_superseder=dict(retractions_by_superseder),
        retractions_by_superseder_ref=dict(retractions_by_superseder_ref),
        inactive_claim_ids=inactive_claim_ids,
        inapplicable_claim_ids=inapplicable_claim_ids,
        sources_by_id={item.source_id: item for item in sources},
        derivations_by_id={item.derivation_id: item for item in derivations},
        resolutions_by_id=resolutions_by_id,
        lifecycle_events_by_id=lifecycle_events_by_id,
        resolutions_by_conflict_ref=resolutions_by_ref,
        resolutions_by_conflict_ref_and_scope=resolutions_by_lane,
        lifecycle_history_by_conflict_ref=lifecycle_by_ref,
        lifecycle_history_by_conflict_ref_and_scope=lifecycle_by_lane,
        effective_resolutions_by_conflict_ref_and_scope=effective,
        active_resolutions_by_conflict_ref_and_scope=active,
        lifecycle_status_by_conflict_ref_and_scope=statuses,
        conflicts_by_id=conflicts_by_id,
        conflicts_by_ref=conflicts_by_ref,
        conflicts_by_key={
            key: tuple(sorted(items, key=lambda item: item.conflict_id))
            for key, items in sorted(conflicts_by_key_raw.items())
        },
        claim_ids_by_evidence_id=claim_ids_by_evidence_id,
        claim_ids_by_derivation_id=claim_ids_by_derivation_id,
        evidence_ids_by_source_id=evidence_ids_by_source_id,
        resolution_ids_by_claim_id=resolution_ids_by_claim_id,
        direct_conflict_indexes=direct_conflict_indexes,
        predicate_registry=predicate_registry,
    )


def _claim_ids_by_evidence_id(claims: Iterable[Claim]) -> dict[str, tuple[str, ...]]:
    result: defaultdict[str, set[str]] = defaultdict(set)
    for claim in claims:
        for evidence_id in claim.evidence_ids:
            result[evidence_id].add(claim.claim_id)
    return {key: tuple(sorted(value)) for key, value in sorted(result.items())}


def _claim_ids_by_derivation_id(claims: Iterable[Claim]) -> dict[str, tuple[str, ...]]:
    result: defaultdict[str, set[str]] = defaultdict(set)
    for claim in claims:
        if claim.derivation_id is not None:
            result[claim.derivation_id].add(claim.claim_id)
    return {key: tuple(sorted(value)) for key, value in sorted(result.items())}


def _evidence_ids_by_source_id(evidence: Iterable[Evidence]) -> dict[str, tuple[str, ...]]:
    result: defaultdict[str, set[str]] = defaultdict(set)
    for item in evidence:
        if item.source_id is not None:
            result[item.source_id].add(item.evidence_id)
    return {key: tuple(sorted(value)) for key, value in sorted(result.items())}


def _resolution_ids_by_claim_id(
    resolutions: Iterable[ResolutionRecord],
) -> dict[str, tuple[str, ...]]:
    result: defaultdict[str, set[str]] = defaultdict(set)
    for resolution in resolutions:
        for claim_id in (
            *resolution.selected_claim_ids,
            *resolution.rejected_claim_ids,
            *resolution.retained_claim_ids,
        ):
            result[claim_id].add(resolution.resolution_id)
    return {key: tuple(sorted(value)) for key, value in sorted(result.items())}


def _mark_claim_keys(
    claim_ids: Iterable[str],
    claims_by_id: Mapping[str, Claim],
    affected_keys: set[ClaimKey],
) -> None:
    for claim_id in claim_ids:
        claim = claims_by_id.get(claim_id)
        if claim is not None:
            affected_keys.add(claim.key)


def _insert_immutable(mapping: dict[str, Any], key: str, value: Any, identifier_name: str) -> bool:
    existing = mapping.get(key)
    if existing is None:
        mapping[key] = value
        return True
    if existing != value:
        raise ValueError(f"{identifier_name} collision with different payload: {key}")
    return False


def _append_sorted_retraction(
    mapping: dict[str, list[ClaimRetracted]], key: str, op: ClaimRetracted
) -> None:
    items = mapping.setdefault(key, [])
    if any(item.op_id == op.op_id for item in items):
        return
    items.append(op)
    items.sort(key=lambda item: (item.timestamp, item.op_id))


def _append_resolution_history(
    by_ref: dict[str, tuple[ResolutionRecord, ...]],
    by_lane: dict[tuple[str, str | None], tuple[ResolutionRecord, ...]],
    resolution: ResolutionRecord,
) -> None:
    by_ref[resolution.conflict_ref] = tuple(
        sorted(
            (*by_ref.get(resolution.conflict_ref, ()), resolution),
            key=lambda item: (_timestamp(item.timestamp), item.resolution_id),
        )
    )
    lane = (resolution.conflict_ref, resolution.scope)
    by_lane[lane] = tuple(
        sorted(
            (*by_lane.get(lane, ()), resolution),
            key=lambda item: (_timestamp(item.timestamp), item.resolution_id),
        )
    )


def _append_lifecycle_history(
    by_ref: dict[str, tuple[ConflictLifecycleEvent, ...]],
    by_lane: dict[tuple[str, str | None], tuple[ConflictLifecycleEvent, ...]],
    event: ConflictLifecycleEvent,
) -> None:
    by_ref[event.conflict_ref] = tuple(
        sorted(
            (*by_ref.get(event.conflict_ref, ()), event),
            key=lambda item: (_timestamp(item.timestamp), item.event_id),
        )
    )
    lane = (event.conflict_ref, event.scope)
    by_lane[lane] = tuple(
        sorted(
            (*by_lane.get(lane, ()), event),
            key=lambda item: (_timestamp(item.timestamp), item.event_id),
        )
    )


def _replace_lifecycle_fold_for_ref(
    *,
    conflict_ref: str,
    resolutions_by_id: dict[str, ResolutionRecord],
    resolutions_by_ref: dict[str, tuple[ResolutionRecord, ...]],
    lifecycle_by_ref: dict[str, tuple[ConflictLifecycleEvent, ...]],
    conflicts_by_ref: dict[str, ConflictSet],
    active: dict[tuple[str, str | None], ResolutionRecord],
    effective: dict[tuple[str, str | None], ResolutionRecord],
    statuses: dict[tuple[str, str | None], str],
) -> None:
    for mapping in (active, effective, statuses):
        for lane in [item for item in mapping if item[0] == conflict_ref]:
            mapping.pop(lane, None)
    resolutions_by_lane: defaultdict[tuple[str, str | None], list[ResolutionRecord]] = defaultdict(
        list
    )
    for item in resolutions_by_ref.get(conflict_ref, ()):
        resolutions_by_lane[(conflict_ref, item.scope)].append(item)
    lifecycle_by_lane: defaultdict[tuple[str, str | None], list[ConflictLifecycleEvent]] = (
        defaultdict(list)
    )
    for item in lifecycle_by_ref.get(conflict_ref, ()):
        lifecycle_by_lane[(conflict_ref, item.scope)].append(item)
    folded_active, folded_effective, folded_statuses = _fold_lifecycle(
        resolutions_by_id=resolutions_by_id,
        resolutions_by_lane={key: tuple(value) for key, value in resolutions_by_lane.items()},
        events_by_lane={key: tuple(value) for key, value in lifecycle_by_lane.items()},
        conflicts_by_ref={conflict_ref: conflicts_by_ref[conflict_ref]}
        if conflict_ref in conflicts_by_ref
        else {},
    )
    active.update(folded_active)
    effective.update(folded_effective)
    statuses.update(folded_statuses)


def _checkpoint_list(value: dict[str, Any], key: str) -> list[dict[str, Any]]:
    items = value.get(key)
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        raise ValueError(f"Checkpoint field {key!r} must be a list of objects.")
    return items


def _checkpoint_string_set(value: dict[str, Any], key: str) -> set[str]:
    items = value.get(key)
    if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
        raise ValueError(f"Checkpoint field {key!r} must be a list of strings.")
    return set(items)
