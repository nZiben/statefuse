from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
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
        self._revision = 0

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
        self._revision += 1

    @property
    def revision(self) -> int:
        return self._revision

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


@dataclass(frozen=True)
class DirectConflictIndex:
    """Per-key claims and incompatible edges used by incremental materialization."""

    claims_by_id: dict[str, Claim] = field(default_factory=dict)
    normalized_values_by_claim_id: dict[str, Any] = field(default_factory=dict)
    contexts_by_claim_id: dict[str, dict[str, JSONValue]] = field(default_factory=dict)
    validity_by_claim_id: dict[str, tuple[datetime | None, datetime | None]] = field(
        default_factory=dict
    )
    incompatible_pairs: dict[tuple[str, str], dict[str, JSONValue]] = field(
        default_factory=dict
    )
    adjacent_claim_ids: dict[str, frozenset[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "claims_by_id", dict(self.claims_by_id))
        object.__setattr__(
            self, "normalized_values_by_claim_id", dict(self.normalized_values_by_claim_id)
        )
        object.__setattr__(
            self,
            "contexts_by_claim_id",
            {key: dict(value) for key, value in self.contexts_by_claim_id.items()},
        )
        object.__setattr__(self, "validity_by_claim_id", dict(self.validity_by_claim_id))
        object.__setattr__(
            self,
            "incompatible_pairs",
            {key: dict(value) for key, value in self.incompatible_pairs.items()},
        )
        object.__setattr__(
            self,
            "adjacent_claim_ids",
            {key: frozenset(value) for key, value in self.adjacent_claim_ids.items()},
        )


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
    return _validity_bounds_overlap(_validity_bounds(left), _validity_bounds(right))


def _validity_bounds_overlap(
    left: tuple[datetime | None, datetime | None],
    right: tuple[datetime | None, datetime | None],
) -> bool:
    left_from, left_until = left
    right_from, right_until = right
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


def _validity_bounds(
    interval: ValidityInterval | None,
) -> tuple[datetime | None, datetime | None]:
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


def _has_aggregate_applicability(claims: Sequence[Claim]) -> bool:
    context_values: dict[str, set[Any]] = {}
    for claim in claims:
        for name, value in claim.context.items():
            values = context_values.setdefault(name, set())
            values.add(_context_equality_key(value))
            if len(values) > 1:
                return True

    starts = [
        parse_utc_iso(claim.validity.valid_from)
        for claim in claims
        if claim.validity is not None and claim.validity.valid_from is not None
    ]
    ends = [
        parse_utc_iso(claim.validity.valid_until)
        for claim in claims
        if claim.validity is not None and claim.validity.valid_until is not None
    ]
    return bool(starts and ends and max(starts) >= min(ends))


def _context_equality_key(value: JSONValue) -> Any:
    """Hash JSON values with the same equality semantics as contexts_overlap()."""
    if value is None:
        return ("null",)
    if isinstance(value, (bool, int, float)):
        # Python intentionally considers True == 1 == 1.0; preserve that behavior.
        return ("number", value)
    if isinstance(value, str):
        return ("string", value)
    if isinstance(value, list):
        return ("list", tuple(_context_equality_key(item) for item in value))
    return (
        "object",
        tuple(sorted((key, _context_equality_key(item)) for key, item in value.items())),
    )


def build_direct_conflict_index(
    claims: Iterable[Claim],
    registry: PredicateRegistry,
    *,
    conflict: ConflictSet | None = None,
) -> DirectConflictIndex:
    """Build an index without repeating semantic pair detection.

    Full materialization supplies its already-detected conflict. Checkpoint recovery can
    therefore reconstruct the index from the stored conflict witness in linear space.
    """

    claims_by_id = {claim.claim_id: claim for claim in claims}
    normalized = {
        claim_id: (
            registry.normalize_value(claim.key.predicate, claim.value)
            if registry.rule_for(claim.key.predicate).equal is None
            else claim.value
        )
        for claim_id, claim in claims_by_id.items()
    }
    contexts = {claim_id: dict(claim.context) for claim_id, claim in claims_by_id.items()}
    validity = {
        claim_id: _validity_bounds(claim.validity)
        for claim_id, claim in claims_by_id.items()
    }
    pairs: dict[tuple[str, str], dict[str, JSONValue]] = {}
    if conflict is not None:
        raw_pairs = conflict.witness.get("incompatible_pairs", [])
        if isinstance(raw_pairs, list):
            for raw_pair in raw_pairs:
                if not isinstance(raw_pair, dict):
                    continue
                claim_ids = raw_pair.get("claim_ids")
                if (
                    not isinstance(claim_ids, list)
                    or len(claim_ids) != 2
                    or not all(isinstance(item, str) for item in claim_ids)
                ):
                    continue
                left_id, right_id = sorted(claim_ids)
                left = claims_by_id.get(left_id)
                right = claims_by_id.get(right_id)
                if left is not None and right is not None:
                    pairs[(left_id, right_id)] = _overlap_witness(left, right)
    return _direct_index(
        claims_by_id=claims_by_id,
        normalized=normalized,
        contexts=contexts,
        validity=validity,
        pairs=pairs,
    )


def update_direct_conflict_index(
    index: DirectConflictIndex,
    claims: Iterable[Claim],
    registry: PredicateRegistry,
) -> tuple[DirectConflictIndex, int]:
    """Update one key's index and return the semantic comparison count."""

    desired = {claim.claim_id: claim for claim in claims}
    claims_by_id = dict(index.claims_by_id)
    normalized = dict(index.normalized_values_by_claim_id)
    contexts = {key: dict(value) for key, value in index.contexts_by_claim_id.items()}
    validity = dict(index.validity_by_claim_id)
    pairs = {key: dict(value) for key, value in index.incompatible_pairs.items()}
    adjacent = {key: set(value) for key, value in index.adjacent_claim_ids.items()}

    for claim_id in sorted(set(claims_by_id) - set(desired)):
        for other_id in tuple(adjacent.get(claim_id, ())):
            pairs.pop(_pair_key(claim_id, other_id), None)
            adjacent.get(other_id, set()).discard(claim_id)
        adjacent.pop(claim_id, None)
        claims_by_id.pop(claim_id, None)
        normalized.pop(claim_id, None)
        contexts.pop(claim_id, None)
        validity.pop(claim_id, None)

    comparisons = 0
    for claim_id in sorted(set(desired) - set(claims_by_id)):
        claim = desired[claim_id]
        rule = registry.rule_for(claim.key.predicate)
        normalized_value = (
            registry.normalize_value(claim.key.predicate, claim.value)
            if rule.equal is None
            else claim.value
        )
        claim_context = dict(claim.context)
        claim_validity = _validity_bounds(claim.validity)
        if not rule.multi_valued:
            for other_id in sorted(claims_by_id):
                other = claims_by_id[other_id]
                if not contexts_overlap(claim_context, contexts[other_id]):
                    continue
                if not _validity_bounds_overlap(claim_validity, validity[other_id]):
                    continue
                left, right = (
                    (claim, other) if claim.claim_id < other.claim_id else (other, claim)
                )
                if rule.equal is None:
                    left_normalized, right_normalized = (
                        (normalized_value, normalized[other_id])
                        if left is claim
                        else (normalized[other_id], normalized_value)
                    )
                    if left_normalized == right_normalized:
                        continue
                comparisons += 1
                if registry.values_equal(
                    claim.key.predicate,
                    left.value,
                    right.value,
                ):
                    continue
                pair = _pair_key(claim_id, other_id)
                pairs[pair] = _overlap_witness(
                    claim if pair[0] == claim_id else other,
                    other if pair[1] == other_id else claim,
                )
                adjacent.setdefault(claim_id, set()).add(other_id)
                adjacent.setdefault(other_id, set()).add(claim_id)
        claims_by_id[claim_id] = claim
        normalized[claim_id] = normalized_value
        contexts[claim_id] = claim_context
        validity[claim_id] = claim_validity
        adjacent.setdefault(claim_id, set())

    return (
        _direct_index(
            claims_by_id=claims_by_id,
            normalized=normalized,
            contexts=contexts,
            validity=validity,
            pairs=pairs,
            adjacent=adjacent,
        ),
        comparisons,
    )


def conflict_from_direct_index(
    index: DirectConflictIndex,
    context: ConflictDetectionContext,
    key: ClaimKey,
) -> ConflictSet | None:
    if not index.incompatible_pairs:
        return None
    participant_ids = {
        claim_id for pair in index.incompatible_pairs for claim_id in pair
    }
    candidates = [index.claims_by_id[claim_id] for claim_id in sorted(participant_ids)]
    conflict_class, conflict_subclass = _taxonomy_for(candidates)
    annotations = _annotations_for(candidates, context)
    if _has_aggregate_applicability(candidates):
        annotations["applicability"] = "aggregate"
    return make_conflict(
        candidates=candidates,
        distinct_values=_distinct_values_for_claims(
            key.predicate, candidates, context.predicate_registry
        ),
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
                index.incompatible_pairs[pair] for pair in sorted(index.incompatible_pairs)
            ]
        },
    )


def _pair_key(left_id: str, right_id: str) -> tuple[str, str]:
    return (left_id, right_id) if left_id <= right_id else (right_id, left_id)


def _direct_index(
    *,
    claims_by_id: dict[str, Claim],
    normalized: dict[str, Any],
    contexts: dict[str, dict[str, JSONValue]],
    validity: dict[str, tuple[datetime | None, datetime | None]],
    pairs: dict[tuple[str, str], dict[str, JSONValue]],
    adjacent: dict[str, set[str]] | None = None,
) -> DirectConflictIndex:
    if adjacent is None:
        adjacent = {claim_id: set() for claim_id in claims_by_id}
        for left_id, right_id in pairs:
            adjacent.setdefault(left_id, set()).add(right_id)
            adjacent.setdefault(right_id, set()).add(left_id)
    return DirectConflictIndex(
        claims_by_id={key: claims_by_id[key] for key in sorted(claims_by_id)},
        normalized_values_by_claim_id={key: normalized[key] for key in sorted(normalized)},
        contexts_by_claim_id={key: contexts[key] for key in sorted(contexts)},
        validity_by_claim_id={key: validity[key] for key in sorted(validity)},
        incompatible_pairs={key: pairs[key] for key in sorted(pairs)},
        adjacent_claim_ids={
            key: frozenset(adjacent.get(key, set())) for key in sorted(claims_by_id)
        },
    )


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
        if _has_aggregate_applicability(candidates):
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
