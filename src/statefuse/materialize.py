from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from .conflict import ConflictSet, PredicateRegistry, derive_conflict_id, detect_conflicts
from .model import (
    Claim,
    ClaimKey,
    ConflictLifecycleEvent,
    Decision,
    Derivation,
    Evidence,
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
    active_resolutions_by_conflict_ref_and_scope: dict[
        tuple[str, str | None], ResolutionRecord
    ] = field(default_factory=dict, repr=False)
    lifecycle_status_by_conflict_ref_and_scope: dict[tuple[str, str | None], str] = field(
        default_factory=dict
    )
    conflicts_by_id: dict[str, ConflictSet] = field(default_factory=dict, repr=False)
    conflicts_by_ref: dict[str, ConflictSet] = field(default_factory=dict, repr=False)


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
                statuses[lane] = "resolved"
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
            ):
                active.pop(lane, None)
                statuses[lane] = "open"
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
                conflict.key, classified
            )
            if observed_matches and current_ids <= classified and len(selected) == 1:
                effective[lane] = resolution
                statuses[lane] = "resolved"
            else:
                statuses[lane] = "reopened"

    return active, effective, statuses


def materialize(oplog: OpLog, predicate_registry: PredicateRegistry | None = None) -> MemoryState:
    registry = predicate_registry or PredicateRegistry()

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
                    "evidence_id collision with different payload: "
                    f"{op.evidence.evidence_id}"
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
                    "derivation_id collision with different payload: "
                    f"{op.derivation.derivation_id}"
                )
            derivations_by_id[op.derivation.derivation_id] = op.derivation
            continue

        if isinstance(op, ResolutionAdded):
            existing = resolutions_by_id.get(op.resolution.resolution_id)
            if existing is not None and existing != op.resolution:
                raise ValueError(
                    "resolution_id collision with different payload: "
                    f"{op.resolution.resolution_id}"
                )
            resolutions_by_id[op.resolution.resolution_id] = op.resolution
            continue

        if isinstance(op, ConflictLifecycleEventAdded):
            existing = lifecycle_events_by_id.get(op.event.event_id)
            if existing is not None and existing != op.event:
                raise ValueError(
                    f"event_id collision with different payload: {op.event.event_id}"
                )
            lifecycle_events_by_id[op.event.event_id] = op.event

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
    conflicts_by_id = {conflict.conflict_id: conflict for conflict in conflicts}
    conflicts_by_ref = {conflict.conflict_ref: conflict for conflict in conflicts}
    active_decisions = _sort_decisions(list(decisions_by_id.values()))
    claim_ids_by_ref = {
        claim_ref: tuple(sorted(claim_ids))
        for claim_ref, claim_ids in claim_ids_by_ref_raw.items()
    }
    resolutions_by_conflict_ref, resolutions_by_lane = _group_resolution_history(
        resolutions_by_id
    )
    lifecycle_history_by_conflict_ref, lifecycle_history_by_lane = _group_lifecycle_history(
        lifecycle_events_by_id
    )
    active_resolutions, effective_resolutions, lifecycle_statuses = _fold_lifecycle(
        resolutions_by_id=resolutions_by_id,
        resolutions_by_lane=resolutions_by_lane,
        events_by_lane=lifecycle_history_by_lane,
        conflicts_by_ref=conflicts_by_ref,
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
    )
