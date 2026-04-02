from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .model import Claim, ClaimKey, claim_ref_from_payload, claim_ref_payload
from .utils import digest_json_value

ValueComparator = Callable[[Any, Any], bool]
ValueNormalizer = Callable[[Any], Any]


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
                        f"Predicate {predicate!r} normalize() is not deterministic for sample {value!r}."
                    )
        if rule.equal is not None:
            for left in sample_values:
                for right in sample_values:
                    outcomes = [bool(rule.equal(left, right)) for _ in range(max(2, repeats))]
                    if any(item != outcomes[0] for item in outcomes[1:]):
                        raise PredicateContractError(
                            f"Predicate {predicate!r} equal() is not deterministic for samples {left!r}, {right!r}."
                        )
        if rule.normalize_for_claim_ref and rule.normalize is None:
            raise PredicateContractError(
                f"Predicate {predicate!r} cannot normalize claim refs without a normalize() function."
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
                    f"Predicate {predicate!r} cannot normalize claim refs without a normalize() function."
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "conflict_id": self.conflict_id,
            "key": self.key.to_dict(),
            "reason": self.reason,
            "candidate_claim_ids": [claim.claim_id for claim in self.candidates],
            "distinct_values": list(self.distinct_values),
        }


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


def detect_conflicts(
    active_claims_by_key: Mapping[ClaimKey, list[Claim]],
    registry: PredicateRegistry,
) -> list[ConflictSet]:
    conflicts: list[ConflictSet] = []
    for key in sorted(active_claims_by_key):
        claims = sorted(active_claims_by_key[key], key=lambda item: item.claim_id)
        if len(claims) <= 1:
            continue
        if registry.is_multi_valued(key.predicate):
            continue
        distinct_values = _distinct_values_for_claims(key.predicate, claims, registry)
        if len(distinct_values) <= 1:
            continue
        payload = {
            "key": key.to_dict(),
            "claim_ids": [claim.claim_id for claim in claims],
        }
        conflict_id = f"conflict:{digest_json_value(payload)}"
        conflicts.append(
            ConflictSet(
                conflict_id=conflict_id,
                key=key,
                candidates=tuple(claims),
                distinct_values=tuple(distinct_values),
                reason="functional predicate has multiple distinct active values",
            )
        )
    return conflicts
