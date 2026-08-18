from __future__ import annotations

from .base import (
    AsyncMemoryRepositoryAdapter,
    ExternalReferenceStore,
    MemoryRepositoryAdapter,
)
from .errors import (
    AdapterAuthenticationError,
    AdapterConfigurationError,
    AdapterError,
    AdapterProtocolError,
    AdapterSearchError,
    AdapterUnavailableError,
    AdapterWriteError,
)
from .fake import FakeMemoryRepositoryAdapter
from .graphiti import AsyncGraphitiAdapter, GraphitiAdapter
from .langmem import (
    AsyncLangGraphStoreAdapter,
    AsyncLangMemAdapter,
    LangGraphStoreAdapter,
    LangMemAdapter,
)
from .letta import LettaAdapter
from .mem0 import AsyncMem0Adapter, Mem0Adapter
from .models import (
    ExternalReference,
    ExternalWriteResult,
    HydratedContext,
    RetrievalRecord,
    SearchHit,
    SearchRequest,
    SyncFailure,
    SyncReport,
)
from .projection import (
    AsyncProjectionService,
    ProjectionService,
    hydrate_search_hits,
    project_state,
)
from .registry import InMemoryExternalReferenceStore

__all__ = [
    "AdapterAuthenticationError",
    "AdapterConfigurationError",
    "AdapterError",
    "AdapterProtocolError",
    "AdapterSearchError",
    "AdapterUnavailableError",
    "AdapterWriteError",
    "AsyncGraphitiAdapter",
    "AsyncLangGraphStoreAdapter",
    "AsyncLangMemAdapter",
    "AsyncMem0Adapter",
    "AsyncMemoryRepositoryAdapter",
    "AsyncProjectionService",
    "ExternalReference",
    "ExternalReferenceStore",
    "ExternalWriteResult",
    "FakeMemoryRepositoryAdapter",
    "GraphitiAdapter",
    "HydratedContext",
    "InMemoryExternalReferenceStore",
    "LangGraphStoreAdapter",
    "LangMemAdapter",
    "LettaAdapter",
    "Mem0Adapter",
    "MemoryRepositoryAdapter",
    "ProjectionService",
    "RetrievalRecord",
    "SearchHit",
    "SearchRequest",
    "SyncFailure",
    "SyncReport",
    "hydrate_search_hits",
    "project_state",
]
