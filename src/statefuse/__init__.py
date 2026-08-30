from __future__ import annotations

from .auth import (
    claim_signature_status,
    retraction_signature_status,
    sign_claim,
    sign_retraction,
    verify_claim_signature,
    verify_retraction_signature,
)
from .compaction import (
    CompactionReport,
    compact_oplog,
    compact_oplog_with_report,
    compact_projection_equivalent,
    compact_projection_equivalent_with_report,
)
from .conflict import (
    ConflictDetectionContext,
    ConflictDetector,
    ConflictSet,
    PredicateContractError,
    PredicateRegistry,
    derive_conflict_id,
    derive_conflict_ref,
    make_conflict,
)
from .materialize import MemoryState, materialize
from .memory import Memory, OpIdMode
from .merge import MergeReport, QuarantinedOp, merge, merge_checked, merge_checked_authenticated
from .model import (
    Claim,
    ClaimKey,
    ConflictLifecycleEvent,
    Decision,
    Derivation,
    Evidence,
    ResolutionRecord,
    Source,
    ValidityInterval,
    derive_claim_ref,
)
from .oplog import OpLog
from .ops import (
    AnyOp,
    ClaimAdded,
    ClaimRetracted,
    ConflictLifecycleEventAdded,
    DecisionAdded,
    DerivationAdded,
    EvidenceAdded,
    Op,
    ResolutionAdded,
    SourceAdded,
)
from .resolution_policy import (
    Abstention,
    CausalResolver,
    ConflictResolver,
    LatestWriteWinsResolver,
    PreserveResolver,
    ResolutionAudit,
    ResolutionContext,
    ResolutionResult,
    ResolverRegistry,
    SelectedState,
    UnresolvedConflict,
)
from .resolver import (
    ConservativeHeuristicResolver,
    HeuristicResolver,
    Resolution,
    Resolver,
    ViewConstraints,
)
from .store import InMemoryStore, JsonlStore, OpStore, SQLiteStore
from .view import Projection, build_view

__all__ = [
    "Abstention",
    "AnyOp",
    "CausalResolver",
    "CompactionReport",
    "Claim",
    "ClaimAdded",
    "ClaimKey",
    "ClaimRetracted",
    "claim_signature_status",
    "compact_oplog",
    "compact_oplog_with_report",
    "compact_projection_equivalent",
    "compact_projection_equivalent_with_report",
    "ConservativeHeuristicResolver",
    "ConflictResolver",
    "ConflictDetectionContext",
    "ConflictDetector",
    "ConflictSet",
    "ConflictLifecycleEvent",
    "ConflictLifecycleEventAdded",
    "Decision",
    "DecisionAdded",
    "derive_claim_ref",
    "derive_conflict_id",
    "derive_conflict_ref",
    "Derivation",
    "DerivationAdded",
    "Evidence",
    "EvidenceAdded",
    "HeuristicResolver",
    "InMemoryStore",
    "JsonlStore",
    "LatestWriteWinsResolver",
    "LLMClient",
    "LLMResolver",
    "Memory",
    "MemoryState",
    "MergeReport",
    "Op",
    "OpIdMode",
    "OpLog",
    "OpStore",
    "OpenAIResponsesClient",
    "PredicateContractError",
    "PredicateRegistry",
    "PreserveResolver",
    "Projection",
    "QuarantinedOp",
    "Resolution",
    "ResolutionAdded",
    "ResolutionAudit",
    "ResolutionContext",
    "ResolutionRecord",
    "ResolutionResult",
    "Resolver",
    "ResolverRegistry",
    "retraction_signature_status",
    "sign_claim",
    "sign_retraction",
    "SQLiteStore",
    "Source",
    "SourceAdded",
    "SelectedState",
    "ValidityInterval",
    "ViewConstraints",
    "UnresolvedConflict",
    "verify_claim_signature",
    "verify_retraction_signature",
    "build_view",
    "materialize",
    "make_conflict",
    "merge",
    "merge_checked",
    "merge_checked_authenticated",
]

__version__ = "0.3.0"


def __getattr__(name: str):  # pragma: no cover - trivial dynamic import
    if name in {"LLMClient", "LLMResolver", "OpenAIResponsesClient"}:
        from .resolver_llm import LLMClient, LLMResolver, OpenAIResponsesClient

        exports = {
            "LLMClient": LLMClient,
            "LLMResolver": LLMResolver,
            "OpenAIResponsesClient": OpenAIResponsesClient,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
