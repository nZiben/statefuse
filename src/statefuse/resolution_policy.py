from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import ClassVar, Protocol, TypeAlias

from .conflict import DIRECT_CONFLICT_TYPE, ConflictSet
from .model import Claim
from .utils import parse_utc_iso


@dataclass(frozen=True)
class ResolutionContext:
    """Caller assertions required by ordering-based policies."""

    timestamps_trusted: bool = False
    concurrent: bool | None = None
    replacement_semantics: bool = False
    causal_metadata_trusted: bool = False

    def __post_init__(self) -> None:
        if type(self.timestamps_trusted) is not bool:
            raise TypeError("ResolutionContext.timestamps_trusted must be a bool.")
        if self.concurrent is not None and type(self.concurrent) is not bool:
            raise TypeError("ResolutionContext.concurrent must be a bool or None.")
        if type(self.replacement_semantics) is not bool:
            raise TypeError("ResolutionContext.replacement_semantics must be a bool.")
        if type(self.causal_metadata_trusted) is not bool:
            raise TypeError("ResolutionContext.causal_metadata_trusted must be a bool.")


@dataclass(frozen=True)
class ResolutionAudit:
    resolver_name: str
    conflict_id: str
    conflict_ref: str
    conflict_type: str
    candidate_claim_ids: tuple[str, ...]
    outcome: str
    details: tuple[tuple[str, str], ...] = ()
    deterministic: bool = True


@dataclass(frozen=True)
class SelectedState:
    conflict_set: ConflictSet
    selected_claim: Claim
    reason: str
    audit: ResolutionAudit

    def __post_init__(self) -> None:
        if self.selected_claim not in self.conflict_set.candidates:
            raise ValueError("SelectedState.selected_claim must belong to its conflict_set.")


@dataclass(frozen=True)
class UnresolvedConflict:
    conflict_set: ConflictSet
    reason: str
    audit: ResolutionAudit


@dataclass(frozen=True)
class Abstention:
    conflict_set: ConflictSet
    reason: str
    audit: ResolutionAudit


ResolutionResult: TypeAlias = SelectedState | UnresolvedConflict | Abstention


class ConflictResolver(Protocol):
    def resolve(
        self,
        conflict_set: ConflictSet,
        context: ResolutionContext,
    ) -> ResolutionResult:
        ...


def _audit(
    resolver_name: str,
    conflict_set: ConflictSet,
    outcome: str,
    *details: tuple[str, str],
) -> ResolutionAudit:
    return ResolutionAudit(
        resolver_name=resolver_name,
        conflict_id=conflict_set.conflict_id,
        conflict_ref=conflict_set.conflict_ref,
        conflict_type=conflict_set.conflict_type,
        candidate_claim_ids=tuple(claim.claim_id for claim in conflict_set.candidates),
        outcome=outcome,
        details=tuple(details),
    )


def _accepted_types(values: Iterable[str]) -> frozenset[str]:
    result = frozenset(values)
    if not result or any(not isinstance(value, str) or not value for value in result):
        raise ValueError("accepted_conflict_types must contain non-empty strings.")
    return result


class PreserveResolver:
    resolver_name = "preserve"

    def resolve(
        self,
        conflict_set: ConflictSet,
        context: ResolutionContext,
    ) -> UnresolvedConflict:
        del context
        reason = "Conflict preserved by policy."
        return UnresolvedConflict(
            conflict_set=conflict_set,
            reason=reason,
            audit=_audit(self.resolver_name, conflict_set, "unresolved"),
        )


@dataclass(frozen=True)
class LatestWriteWinsResolver:
    accepted_conflict_types: frozenset[str] = field(
        default_factory=lambda: frozenset({DIRECT_CONFLICT_TYPE})
    )
    resolver_name: ClassVar[str] = "latest_write_wins"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "accepted_conflict_types",
            _accepted_types(self.accepted_conflict_types),
        )

    def resolve(
        self,
        conflict_set: ConflictSet,
        context: ResolutionContext,
    ) -> ResolutionResult:
        if conflict_set.conflict_type not in self.accepted_conflict_types:
            return self._abstain(conflict_set, "Conflict type is not accepted by this resolver.")
        if not context.replacement_semantics:
            return self._abstain(
                conflict_set,
                "The caller did not establish replacement semantics for this predicate.",
            )
        if not context.timestamps_trusted:
            return self._abstain(conflict_set, "Candidate timestamps are not explicitly trusted.")
        if context.concurrent is not False:
            reason = (
                "Concurrent updates cannot be ordered by latest-write-wins."
                if context.concurrent is True
                else "Non-concurrency was not explicitly established."
            )
            return self._abstain(conflict_set, reason)
        if len(conflict_set.candidates) < 2:
            return self._abstain(conflict_set, "Conflict requires at least two candidates.")

        branch_ids: list[str] = []
        for claim in conflict_set.candidates:
            branch_id = claim.provenance.get("branch_id")
            if not isinstance(branch_id, str) or not branch_id.strip():
                return self._abstain(conflict_set, "Every candidate requires a branch_id.")
            branch_ids.append(branch_id)
        if len(set(branch_ids)) != 1:
            return self._abstain(conflict_set, "Candidates belong to different branches.")

        ordered: list[tuple[Claim, object]] = []
        try:
            ordered = [(claim, parse_utc_iso(claim.timestamp)) for claim in conflict_set.candidates]
        except (AttributeError, OverflowError, TypeError, ValueError):
            return self._abstain(conflict_set, "Every candidate requires a parseable timestamp.")
        if len({timestamp for _, timestamp in ordered}) != len(ordered):
            return self._abstain(conflict_set, "Candidate timestamps contain a tie.")

        selected, _ = max(ordered, key=lambda item: item[1])
        reason = "Selected the unique latest trusted write on one non-concurrent branch."
        return SelectedState(
            conflict_set=conflict_set,
            selected_claim=selected,
            reason=reason,
            audit=_audit(
                self.resolver_name,
                conflict_set,
                "selected",
                ("branch_id", branch_ids[0]),
                ("ordering", "trusted_timestamp"),
                ("selected_claim_id", selected.claim_id),
            ),
        )

    def _abstain(self, conflict_set: ConflictSet, reason: str) -> Abstention:
        return Abstention(
            conflict_set=conflict_set,
            reason=reason,
            audit=_audit(self.resolver_name, conflict_set, "abstained"),
        )


@dataclass(frozen=True)
class CausalResolver:
    accepted_conflict_types: frozenset[str] = field(
        default_factory=lambda: frozenset({DIRECT_CONFLICT_TYPE})
    )
    resolver_name: ClassVar[str] = "causal"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "accepted_conflict_types",
            _accepted_types(self.accepted_conflict_types),
        )

    def resolve(
        self,
        conflict_set: ConflictSet,
        context: ResolutionContext,
    ) -> ResolutionResult:
        if conflict_set.conflict_type not in self.accepted_conflict_types:
            return self._abstain(conflict_set, "Conflict type is not accepted by this resolver.")
        if not context.replacement_semantics:
            return self._abstain(
                conflict_set,
                "The caller did not establish replacement semantics for this predicate.",
            )
        if not context.causal_metadata_trusted:
            return self._abstain(
                conflict_set,
                "Candidate causal metadata is not explicitly trusted.",
            )
        if len(conflict_set.candidates) < 2:
            return self._abstain(conflict_set, "Conflict requires at least two candidates.")

        clocks: list[tuple[Claim, dict[str, int]]] = []
        for claim in conflict_set.candidates:
            clock = self._vector_clock(claim.provenance.get("vector_clock"))
            if clock is None:
                return self._abstain(
                    conflict_set,
                    "Every candidate requires a valid non-empty vector_clock.",
                )
            clocks.append((claim, clock))

        maximal = [
            (claim, clock)
            for index, (claim, clock) in enumerate(clocks)
            if not any(
                other_index != index and self._dominates(other_clock, clock)
                for other_index, (_, other_clock) in enumerate(clocks)
            )
        ]
        if len(maximal) != 1:
            claim_ids = ",".join(sorted(claim.claim_id for claim, _ in maximal))
            reason = "Concurrent or equivalent causal alternatives remain unresolved."
            return UnresolvedConflict(
                conflict_set=conflict_set,
                reason=reason,
                audit=_audit(
                    self.resolver_name,
                    conflict_set,
                    "unresolved",
                    ("maximal_claim_ids", claim_ids),
                ),
            )

        selected, _ = maximal[0]
        reason = "Selected the unique causally dominant candidate."
        return SelectedState(
            conflict_set=conflict_set,
            selected_claim=selected,
            reason=reason,
            audit=_audit(
                self.resolver_name,
                conflict_set,
                "selected",
                ("ordering", "vector_clock"),
                ("selected_claim_id", selected.claim_id),
            ),
        )

    @staticmethod
    def _vector_clock(value: object) -> dict[str, int] | None:
        if not isinstance(value, Mapping) or not value:
            return None
        if any(
            not isinstance(replica, str)
            or not replica
            or type(counter) is not int
            or counter < 0
            for replica, counter in value.items()
        ):
            return None
        return dict(value)

    @staticmethod
    def _dominates(left: Mapping[str, int], right: Mapping[str, int]) -> bool:
        replicas = left.keys() | right.keys()
        return all(left.get(replica, 0) >= right.get(replica, 0) for replica in replicas) and any(
            left.get(replica, 0) > right.get(replica, 0) for replica in replicas
        )

    def _abstain(self, conflict_set: ConflictSet, reason: str) -> Abstention:
        return Abstention(
            conflict_set=conflict_set,
            reason=reason,
            audit=_audit(self.resolver_name, conflict_set, "abstained"),
        )


class ResolverRegistry:
    """Explicit resolver lookup with preserve-on-unknown-type routing."""

    def __init__(self) -> None:
        self._resolvers: dict[str, ConflictResolver] = {
            PreserveResolver.resolver_name: PreserveResolver(),
            LatestWriteWinsResolver.resolver_name: LatestWriteWinsResolver(),
            CausalResolver.resolver_name: CausalResolver(),
        }
        self._routes: dict[str, str] = {}

    def register(self, name: str, resolver: ConflictResolver) -> None:
        if not name or name in self._resolvers:
            raise ValueError(f"Resolver name must be non-empty and unique: {name!r}.")
        if not callable(getattr(resolver, "resolve", None)):
            raise TypeError("resolver must implement resolve().")
        self._resolvers[name] = resolver

    def route(self, conflict_type: str, resolver_name: str) -> None:
        if resolver_name not in self._resolvers:
            raise KeyError(f"Unknown resolver: {resolver_name!r}.")
        if not conflict_type:
            raise ValueError("conflict_type must be non-empty.")
        self._routes[conflict_type] = resolver_name

    def resolve(
        self,
        conflict_set: ConflictSet,
        context: ResolutionContext,
        *,
        resolver_name: str | None = None,
    ) -> ResolutionResult:
        if resolver_name is not None:
            if resolver_name not in self._resolvers:
                raise KeyError(f"Unknown resolver: {resolver_name!r}.")
            resolver = self._resolvers[resolver_name]
        else:
            resolver = self._resolvers[
                self._routes.get(conflict_set.conflict_type, PreserveResolver.resolver_name)
            ]
        result = resolver.resolve(conflict_set, context)
        if not isinstance(result, (SelectedState, UnresolvedConflict, Abstention)):
            raise TypeError("Resolver returned an unsupported result type.")
        if result.conflict_set is not conflict_set:
            raise ValueError("Resolver result must retain the supplied ConflictSet object.")
        return result
