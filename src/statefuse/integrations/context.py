from __future__ import annotations

import re
from dataclasses import dataclass

from ..model import Claim
from ..utils import canonical_json_dumps
from .models import HydratedContext

_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")


@dataclass(frozen=True)
class AssembledContext:
    text: str
    token_count: int
    included_claim_ids: tuple[str, ...]
    included_conflict_ids: tuple[str, ...]
    omitted_claim_ids: tuple[str, ...]
    omitted_conflict_ids: tuple[str, ...]


class ContextAssembler:
    """Pack canonical context using deterministic word-or-punctuation token accounting."""

    def __init__(self, max_tokens: int) -> None:
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer.")
        self.max_tokens = max_tokens

    @staticmethod
    def count_tokens(text: str) -> int:
        return len(_TOKEN_PATTERN.findall(text))

    def assemble(self, context: HydratedContext) -> AssembledContext:
        blocks: list[tuple[str, str, tuple[str, ...], str]] = []
        omitted_claim_ids = set(context.omitted_claim_ids)
        omitted_conflict_ids = set(context.omitted_conflict_ids)
        participant_ids = {
            claim.claim_id for conflict in context.conflicts for claim in conflict.candidates
        }
        resolutions_by_ref = {
            resolution.conflict_ref: resolution for resolution in context.resolutions
        }

        for conflict in sorted(context.conflicts, key=lambda item: item.conflict_id):
            status = context.conflict_statuses.get(conflict.conflict_id, "open")
            resolution = resolutions_by_ref.get(conflict.conflict_ref)
            claim_ids = tuple(claim.claim_id for claim in conflict.candidates)
            if status == "resolved" and resolution is None:
                omitted_conflict_ids.add(conflict.conflict_id)
                omitted_claim_ids.update(claim_ids)
                continue
            current_resolution_state = tuple(
                item
                for item in context.resolution_history
                if item.conflict_ref == conflict.conflict_ref
                and context.resolution_statuses.get(item.resolution_id) != "superseded"
                and item != resolution
            )
            resolution_evidence_ids = tuple(
                evidence_id
                for item in (*current_resolution_state, resolution)
                if item is not None
                for evidence_id in item.evidence_ids
            )
            lines = [
                f"CONFLICT status={status} {canonical_json_dumps(conflict.to_dict())}",
                *self._claim_and_resource_lines(
                    context, conflict.candidates, resolution_evidence_ids
                ),
            ]
            if resolution is not None:
                lines.append(f"RESOLUTION {canonical_json_dumps(resolution.to_dict())}")
            lines.extend(
                f"RESOLUTION_STATE status={context.resolution_statuses[item.resolution_id]} "
                f"{canonical_json_dumps(item.to_dict())}"
                for item in current_resolution_state
            )
            blocks.append(("conflict", conflict.conflict_id, claim_ids, "\n".join(lines)))

        for claim in sorted(context.claims, key=lambda item: item.claim_id):
            if claim.claim_id in participant_ids:
                continue
            blocks.append(
                (
                    "claim",
                    claim.claim_id,
                    (claim.claim_id,),
                    "\n".join(self._claim_and_resource_lines(context, (claim,))),
                )
            )

        parts: list[str] = []
        used_tokens = 0
        included_claim_ids: set[str] = set()
        included_conflict_ids: set[str] = set()
        for kind, item_id, claim_ids, block in blocks:
            block_tokens = self.count_tokens(block)
            if used_tokens + block_tokens <= self.max_tokens:
                parts.append(block)
                used_tokens += block_tokens
                included_claim_ids.update(claim_ids)
                if kind == "conflict":
                    included_conflict_ids.add(item_id)
            elif kind == "conflict":
                omitted_conflict_ids.add(item_id)
                omitted_claim_ids.update(claim_ids)
            else:
                omitted_claim_ids.add(item_id)

        text = "\n\n".join(parts)
        omitted_claim_ids.difference_update(included_claim_ids)
        omitted_conflict_ids.difference_update(included_conflict_ids)
        return AssembledContext(
            text=text,
            token_count=used_tokens,
            included_claim_ids=tuple(sorted(included_claim_ids)),
            included_conflict_ids=tuple(sorted(included_conflict_ids)),
            omitted_claim_ids=tuple(sorted(omitted_claim_ids)),
            omitted_conflict_ids=tuple(sorted(omitted_conflict_ids)),
        )

    @staticmethod
    def _claim_and_resource_lines(
        context: HydratedContext,
        claims: tuple[Claim, ...],
        extra_evidence_ids: tuple[str, ...] = (),
    ) -> list[str]:
        claim_ids = {claim.claim_id for claim in claims}
        lines = [
            f"CLAIM status={context.claim_statuses.get(claim.claim_id, 'unknown')} "
            f"{canonical_json_dumps(claim.to_dict())}"
            for claim in claims
        ]
        evidence_ids = {
            evidence_id for claim in claims for evidence_id in claim.evidence_ids
        } | set(extra_evidence_ids)
        evidence = sorted(
            (item for item in context.evidence if item.evidence_id in evidence_ids),
            key=lambda item: item.evidence_id,
        )
        lines.extend(f"EVIDENCE {canonical_json_dumps(item.to_dict())}" for item in evidence)
        source_ids = {item.source_id for item in evidence if item.source_id is not None}
        lines.extend(
            f"SOURCE {canonical_json_dumps(item.to_dict())}"
            for item in sorted(context.sources, key=lambda item: item.source_id)
            if item.source_id in source_ids
        )
        lines.extend(
            f"DERIVATION {canonical_json_dumps(item.to_dict())}"
            for item in sorted(context.derivations, key=lambda item: item.derivation_id)
            if claim_ids.intersection((*item.input_claim_ids, *item.output_claim_ids))
        )
        return lines
