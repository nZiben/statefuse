from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Protocol

from .model import (
    Claim,
    ClaimKey,
    Decision,
    Derivation,
    Evidence,
    JSONValue,
    Source,
    ValidityInterval,
    claim_ref_from_payload,
    claim_ref_payload,
)
from .utils import digest_json_value, parse_utc_iso

ValueComparator = Callable[[Any, Any], bool]
ValueNormalizer = Callable[[Any], Any]
DIRECT_CONFLICT_TYPE = "same_key_distinct_value"


@dataclass(frozen=True)
class PredicateRule:
    multi_valued: bool = False
    normalize: ValueNormalizer | None = None
    equal: ValueComparator | None = None
    normalize_for_claim_ref: bool = False


class PredicateContractError(ValueError):
    pass


class PredicateRegistry:
    """Predicate behavior registry for deterministic, replica-invariant conflict rules."""

    def __init__(self) -> None:
        self._rules: dict[str, PredicateRule] = {}

    def register(
        self,
        predicate: str,
        *,
        multi_valued: bool = False,
        normalize: ValueNormalizer | None = None,
        equal: ValueComparator | None = None,
        normalize_for_claim_ref: bool = False,
    ) -> None:
        self._rules[predicate] = PredicateRule(
            multi_valued=multi_valued,
            normalize=normalize,
            equal=equal,
            normalize_for_claim_ref=normalize_for_claim_ref,
        )

    def rule_for(self, predicate: str) -> PredicateRule:
        return self._rules.get(predicate, PredicateRule(multi_valued=False))

    def is_multi_valued(self, predicate: str) -> bool:
        return self.rule_for(predicate).multi_valued

    def normalize_value(self, predicate: str, value: Any) -> Any:
        rule = self.rule_for(predicate)
        if rule.normalize is None:
            return value
        return rule.normalize(value)

    def values_equal(self, predicate: str, left: Any, right: Any) -> bool:
        rule = self.rule_for(predicate)
        if rule.equal is not None:
            return bool(rule.equal(left, right))
        return self.normalize_value(predicate, left) == self.normalize_value(predicate, right)

    def claim_ref_value(self, predicate: str, value: Any) -> Any:
        rule = self.rule_for(predicate)
        if rule.normalize is None or not rule.normalize_for_claim_ref:
            return value
        return rule.normalize(value)

    def claim_ref_for_claim(self, claim: Claim) -> str:
        return claim_ref_from_payload(
            claim_ref_payload(
                key=claim.key,
                value=self.claim_ref_value(claim.key.predicate, claim.value),
                confidence=claim.confidence,
                timestamp=claim.timestamp,
                evidence_ids=claim.evidence_ids,
                provenance=claim.provenance,
                kind=claim.kind,
                context=claim.context,
            )
        )

    def validate_contract(
        self,
        predicate: str,
        sample_values: Sequence[Any],
        *,
        repeats: int = 3,
    ) -> None:
        rule = self.rule_for(predicate)
        if rule.normalize is not None:
            for value in sample_values:
                normalized = [rule.normalize(value) for _ in range(max(2, repeats))]
                if any(item != normalized[0] for item in normalized[1:]):
                    raise PredicateContractError(
                        f"Predicate {predicate!r} normalize() is not deterministic "
                        f"for sample {value!r}."
                    )
        if rule.equal is not None:
            for left in sample_values:
                for right in sample_values:
                    outcomes = [bool(rule.equal(left, right)) for _ in range(max(2, repeats))]
                    if any(item != outcomes[0] for item in outcomes[1:]):
                        raise PredicateContractError(
                            f"Predicate {predicate!r} equal() is not deterministic "
                            f"for samples {left!r}, {right!r}."
                        )
        if rule.normalize_for_claim_ref and rule.normalize is None:
            raise PredicateContractError(
                f"Predicate {predicate!r} cannot normalize claim refs "
                "without a normalize() function."
            )

    def validate_contracts(
        self,
        samples_by_predicate: Mapping[str, Sequence[Any]] | None = None,
        *,
        repeats: int = 3,
    ) -> None:
        for predicate, rule in self._rules.items():
            if rule.normalize_for_claim_ref and rule.normalize is None:
                raise PredicateContractError(
                    f"Predicate {predicate!r} cannot normalize claim refs "
                    "without a normalize() function."
                )
            samples = tuple((samples_by_predicate or {}).get(predicate, ()))
            if not samples and rule.normalize is None and rule.equal is None:
                continue
            self.validate_contract(predicate, samples, repeats=repeats)


@dataclass(frozen=True)
class ConflictSet:
    conflict_id: str
    key: ClaimKey
    candidates: tuple[Claim, ...]
    distinct_values: tuple[Any, ...]
    reason: str
    conflict_ref: str = ""
    conflict_type: str = DIRECT_CONFLICT_TYPE
    keys: tuple[ClaimKey, ...] = field(default_factory=tuple)
    conflict_class: str = "epistemic"
    conflict_subclass: str = "factual.value"
    detector_id: str = "direct"
    annotations: dict[str, JSONValue] = field(default_factory=dict)
    witness: dict[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        candidates = tuple(sorted(self.candidates, key=lambda claim: claim.claim_id))
        if len(candidates) != len({claim.claim_id for claim in candidates}):
            raise ValueError("Conflict candidates must have unique claim IDs.")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "distinct_values", tuple(self.distinct_values))
        keys = tuple(
            sorted({self.key, *self.keys, *(claim.key for claim in candidates)})
        )
        object.__setattr__(self, "keys", keys)
        object.__setattr__(self, "annotations", dict(self.annotations))
        object.__setattr__(self, "witness", dict(self.witness))
        required = (
            self.conflict_id,
            self.conflict_type,
            self.conflict_class,
            self.conflict_subclass,
            self.detector_id,
        )
        if any(not isinstance(item, str) or not item for item in required):
            raise ValueError("Conflict identifiers, type, class, and detector_id are required.")
        if not self.conflict_ref:
            object.__setattr__(
                self,
                "conflict_ref",
                derive_conflict_ref(
                    self.key,
                    conflict_type=self.conflict_type,
                    keys=keys,
                    detector_id=self.detector_id,
                ),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "conflict_id": self.conflict_id,
            "conflict_ref": self.conflict_ref,
            "conflict_type": self.conflict_type,
            "conflict_class": self.conflict_class,
            "conflict_subclass": self.conflict_subclass,
            "detector_id": self.detector_id,
            "key": self.key.to_dict(),
            "keys": [key.to_dict() for key in self.keys],
            "reason": self.reason,
            "candidate_claim_ids": [claim.claim_id for claim in self.candidates],
            "distinct_values": list(self.distinct_values),
            "annotations": dict(self.annotations),
            "witness": dict(self.witness),
        }


@dataclass(frozen=True)
class ConflictDetectionContext:
    active_claims_by_key: Mapping[ClaimKey, list[Claim]]
    claims_by_id: Mapping[str, Claim]
    evidence_by_id: Mapping[str, Evidence]
    sources_by_id: Mapping[str, Source]
    derivations_by_id: Mapping[str, Derivation]
    active_decisions: tuple[Decision, ...]
    predicate_registry: PredicateRegistry


class ConflictDetector(Protocol):
    def __call__(self, context: ConflictDetectionContext) -> Iterable[ConflictSet]:
        ...


def derive_conflict_ref(
    key: ClaimKey,
    *,
    conflict_type: str = DIRECT_CONFLICT_TYPE,
    keys: Sequence[ClaimKey] = (),
    detector_id: str = "direct",
) -> str:
    is_legacy = (
        conflict_type == DIRECT_CONFLICT_TYPE
        and detector_id == "direct"
        and set(keys).issubset({key})
    )
    payload: dict[str, Any] = {"conflict_type": conflict_type, "key": key.to_dict()}
    if not is_legacy:
        payload["detector_id"] = detector_id
    return f"conflict-ref:{digest_json_value(payload)}"


def derive_conflict_id(
    key: ClaimKey,
    claim_ids: Sequence[str],
    *,
    conflict_type: str = DIRECT_CONFLICT_TYPE,
    keys: Sequence[ClaimKey] = (),
    detector_id: str = "direct",
) -> str:
    is_legacy = (
        conflict_type == DIRECT_CONFLICT_TYPE
        and detector_id == "direct"
        and set(keys).issubset({key})
    )
    payload: dict[str, Any] = {"key": key.to_dict(), "claim_ids": sorted(claim_ids)}
    if not is_legacy:
        payload = {
            "conflict_ref": derive_conflict_ref(
                key,
                conflict_type=conflict_type,
                keys=keys,
                detector_id=detector_id,
            ),
            "claim_ids": sorted(claim_ids),
        }
    return f"conflict:{digest_json_value(payload)}"


def make_conflict(
    *,
    candidates: Sequence[Claim],
    conflict_type: str,
    reason: str,
    detector_id: str,
    conflict_class: str,
    conflict_subclass: str,
    key: ClaimKey | None = None,
    keys: Sequence[ClaimKey] = (),
    distinct_values: Sequence[Any] | None = None,
    annotations: Mapping[str, JSONValue] | None = None,
    witness: Mapping[str, JSONValue] | None = None,
) -> ConflictSet:
    ordered = tuple(sorted(candidates, key=lambda claim: claim.claim_id))
    if not ordered:
        raise ValueError("A conflict requires at least one candidate claim.")
    related_keys = tuple(sorted({*keys, *(claim.key for claim in ordered)}))
    if key is None and len(related_keys) > 1:
        raise ValueError("Multi-key conflicts require an explicit stable key anchor.")
    anchor = key or related_keys[0]
    values = list(distinct_values or ())
    if distinct_values is None:
        for claim in ordered:
            if claim.value not in values:
                values.append(claim.value)
    return ConflictSet(
        conflict_id=derive_conflict_id(
            anchor,
            [claim.claim_id for claim in ordered],
            conflict_type=conflict_type,
            keys=related_keys,
            detector_id=detector_id,
        ),
        conflict_ref=derive_conflict_ref(
            anchor,
            conflict_type=conflict_type,
            keys=related_keys,
            detector_id=detector_id,
        ),
        conflict_type=conflict_type,
        key=anchor,
        keys=related_keys,
        candidates=ordered,
        distinct_values=tuple(values),
        reason=reason,
        conflict_class=conflict_class,
        conflict_subclass=conflict_subclass,
        detector_id=detector_id,
        annotations=dict(annotations or {}),
        witness=dict(witness or {}),
    )


def _distinct_values_for_claims(
    predicate: str,
    claims: list[Claim],
    registry: PredicateRegistry,
) -> list[Any]:
    distinct: list[Any] = []
    for claim in claims:
        if not any(registry.values_equal(predicate, claim.value, value) for value in distinct):
            distinct.append(claim.value)
    return distinct


def contexts_overlap(left: Mapping[str, JSONValue], right: Mapping[str, JSONValue]) -> bool:
    return all(left[key] == right[key] for key in left.keys() & right.keys())


def validity_overlaps(left: ValidityInterval | None, right: ValidityInterval | None) -> bool:
    left_from, left_until = _validity_bounds(left)
    right_from, right_until = _validity_bounds(right)
    if left_from is not None and left_until is not None and left_from == left_until:
        return False
    if right_from is not None and right_until is not None and right_from == right_until:
        return False
    if left_until is not None and right_from is not None and left_until <= right_from:
        return False
    if right_until is not None and left_from is not None and right_until <= left_from:
        return False
    return True


def claim_applies(
    claim: Claim,
    *,
    valid_at: str | None = None,
    context: Mapping[str, JSONValue] | None = None,
) -> bool:
    if context and any(claim.context.get(key, value) != value for key, value in context.items()):
        return False
    if valid_at is None:
        return True
    instant = parse_utc_iso(valid_at)
    valid_from, valid_until = _validity_bounds(claim.validity)
    return (valid_from is None or valid_from <= instant) and (
        valid_until is None or instant < valid_until
    )


def _validity_bounds(interval: ValidityInterval | None) -> tuple[Any | None, Any | None]:
    if interval is None:
        return None, None
    return (
        parse_utc_iso(interval.valid_from) if interval.valid_from is not None else None,
        parse_utc_iso(interval.valid_until) if interval.valid_until is not None else None,
    )


def _overlap_witness(left: Claim, right: Claim) -> dict[str, JSONValue]:
    starts = [
        value
        for value in (
            left.validity.valid_from if left.validity else None,
            right.validity.valid_from if right.validity else None,
        )
        if value is not None
    ]
    ends = [
        value
        for value in (
            left.validity.valid_until if left.validity else None,
            right.validity.valid_until if right.validity else None,
        )
        if value is not None
    ]
    shared_context = dict(left.context)
    shared_context.update(right.context)
    return {
        "claim_ids": [left.claim_id, right.claim_id],
        "valid_from": max(starts, key=parse_utc_iso) if starts else None,
        "valid_until": min(ends, key=parse_utc_iso) if ends else None,
        "context": shared_context,
    }


def _taxonomy_for(claims: Sequence[Claim]) -> tuple[str, str]:
    kinds = {claim.kind for claim in claims}
    mapping = {
        "belief": ("epistemic", "source.belief"),
        "instruction": ("normative", "instruction"),
        "preference": ("normative", "preference"),
        "policy": ("normative", "policy.rule"),
        "goal": ("normative", "goal"),
    }
    classified = {mapping[kind] for kind in kinds if kind in mapping}
    classes = {item[0] for item in classified}
    if len(classified) == 1:
        return next(iter(classified))
    if len(classes) == 1:
        return next(iter(classes)), "mixed"
    return "epistemic", "factual.value"


def _annotations_for(
    claims: Sequence[Claim], context: ConflictDetectionContext
) -> dict[str, JSONValue]:
    source_types = sorted(
        {
            source.source_type
            for claim in claims
            for evidence_id in claim.evidence_ids
            if (evidence := context.evidence_by_id.get(evidence_id)) is not None
            and evidence.source_id is not None
            and (source := context.sources_by_id.get(evidence.source_id)) is not None
        }
    )
    values = [claim.value for claim in claims]
    representation = (
        "numeric"
        if values
        and all(
            isinstance(value, (int, float)) and not isinstance(value, bool) for value in values
        )
        else "categorical"
        if values and all(isinstance(value, (str, bool)) for value in values)
        else "structured"
    )
    annotations: dict[str, JSONValue] = {
        "representation": representation,
        "dependency_depth": (
            "multi_hop"
            if any(
                claim.derivation_id is not None
                and (derivation := context.derivations_by_id.get(claim.derivation_id))
                is not None
                and len(derivation.input_claim_ids) > 1
                for claim in claims
            )
            else "direct"
        ),
    }
    if source_types:
        annotations["provenance"] = source_types
    return annotations


def _direct_conflicts(context: ConflictDetectionContext) -> list[ConflictSet]:
    registry = context.predicate_registry
    conflicts: list[ConflictSet] = []
    for key in sorted(context.active_claims_by_key):
        claims = sorted(context.active_claims_by_key[key], key=lambda item: item.claim_id)
        if len(claims) <= 1 or registry.is_multi_valued(key.predicate):
            continue
        # ponytail: pair scan; index intervals/context if profiling shows large per-key sets.
        incompatible_pairs = [
            (left, right)
            for left, right in combinations(claims, 2)
            if not registry.values_equal(key.predicate, left.value, right.value)
            and contexts_overlap(left.context, right.context)
            and validity_overlaps(left.validity, right.validity)
        ]
        if not incompatible_pairs:
            continue
        participant_ids = {
            claim.claim_id for pair in incompatible_pairs for claim in pair
        }
        # ponytail: one finding per key; split overlap components if they need separate review.
        candidates = [claim for claim in claims if claim.claim_id in participant_ids]
        conflict_class, conflict_subclass = _taxonomy_for(candidates)
        annotations = _annotations_for(candidates, context)
        if any(
            not contexts_overlap(left.context, right.context)
            or not validity_overlaps(left.validity, right.validity)
            for left, right in combinations(candidates, 2)
        ):
            annotations["applicability"] = "aggregate"
        conflicts.append(
            make_conflict(
                candidates=candidates,
                distinct_values=_distinct_values_for_claims(key.predicate, candidates, registry),
                conflict_type=DIRECT_CONFLICT_TYPE,
                reason=(
                    "functional predicate has incompatible active values in overlapping "
                    "context and validity"
                ),
                detector_id="direct",
                conflict_class=conflict_class,
                conflict_subclass=conflict_subclass,
                key=key,
                keys=(key,),
                annotations=annotations,
                witness={
                    "incompatible_pairs": [
                        _overlap_witness(left, right) for left, right in incompatible_pairs
                    ]
                },
            )
        )
    return conflicts


def detect_conflicts(
    active_claims_by_key: Mapping[ClaimKey, list[Claim]],
    registry: PredicateRegistry,
) -> list[ConflictSet]:
    context = ConflictDetectionContext(
        active_claims_by_key=active_claims_by_key,
        claims_by_id={
            claim.claim_id: claim
            for claims in active_claims_by_key.values()
            for claim in claims
        },
        evidence_by_id={},
        sources_by_id={},
        derivations_by_id={},
        active_decisions=(),
        predicate_registry=registry,
    )
    return _direct_conflicts(context)


def run_conflict_detectors(
    context: ConflictDetectionContext,
    detectors: Sequence[ConflictDetector] = (),
) -> list[ConflictSet]:
    active_ids = set(context.claims_by_id)
    conflicts = [*_direct_conflicts(context)]
    for detector in detectors:
        conflicts.extend(detector(context))

    by_id: dict[str, ConflictSet] = {}
    refs: dict[str, str] = {}
    for conflict in conflicts:
        if not isinstance(conflict, ConflictSet):
            raise TypeError("Conflict detectors must return ConflictSet objects.")
        missing = sorted(
            claim.claim_id
            for claim in conflict.candidates
            if claim.claim_id not in active_ids
            or context.claims_by_id[claim.claim_id] != claim
        )
        if missing:
            raise ValueError(
                "Conflict detector referenced inactive, unknown, or modified claims: "
                + ", ".join(missing)
            )
        expected_ref = derive_conflict_ref(
            conflict.key,
            conflict_type=conflict.conflict_type,
            keys=conflict.keys,
            detector_id=conflict.detector_id,
        )
        expected_id = derive_conflict_id(
            conflict.key,
            [claim.claim_id for claim in conflict.candidates],
            conflict_type=conflict.conflict_type,
            keys=conflict.keys,
            detector_id=conflict.detector_id,
        )
        if conflict.conflict_ref != expected_ref or conflict.conflict_id != expected_id:
            raise ValueError(
                "Conflict detectors must use canonical identities; call make_conflict()."
            )
        existing = by_id.get(conflict.conflict_id)
        if existing is not None:
            if existing != conflict:
                raise ValueError(f"conflict_id collision: {conflict.conflict_id}")
            continue
        prior_id = refs.get(conflict.conflict_ref)
        if prior_id is not None and prior_id != conflict.conflict_id:
            raise ValueError(f"conflict_ref collision: {conflict.conflict_ref}")
        by_id[conflict.conflict_id] = conflict
        refs[conflict.conflict_ref] = conflict.conflict_id
    return [by_id[conflict_id] for conflict_id in sorted(by_id)]
