# Test an adapter

The reusable sync and async contracts are `AdapterContract` and `AsyncAdapterContract` in
`tests/integrations/test_adapter_contract.py`. In a StateFuse checkout, subclass the matching
contract and provide only the adapter factory:

```python
from tests.integrations.test_adapter_contract import AdapterContract


class TestAcmeAdapterContract(AdapterContract):
    def make_adapter(self):
        return AcmeAdapter(test_client)

    def make_unavailable(self, adapter):
        test_client.fail_requests = True

    def make_available(self, adapter):
        test_client.fail_requests = False
```

Run the shared and isolation tests with:

```bash
python -m pytest -q tests/integrations
```

The contract covers creation, idempotent upsert, updates, normalized search, metadata and
StateFuse-ID round trips, deletion, namespace isolation, outages, recovery, canonical operation
preservation, stale hydration, duplicate results, malformed or missing StateFuse metadata, and
unknown external IDs.

`tests/integrations/test_connectors.py` runs the shared contract against SDK-shaped deterministic
clients for Mem0, LangGraph Store, Letta, and Graphiti. Those doubles use the method signatures and
response shapes of the releases below. They do not fabricate live-service results; service-backed
smoke tests still require the corresponding credentials, model providers, and databases.

## Optional dependency metadata

The extras were prepared from official install documentation and PyPI metadata current on
2026-07-27:

| Extra | Declared range | Metadata release checked |
| --- | --- | --- |
| `mem0` | `mem0ai>=2.0,<3` | 2.0.12 |
| `langmem` | `langmem>=0.0.30,<0.1` | 0.0.30 |
| `letta` | `letta-client>=1.12,<2` | 1.12.1 |
| `graphiti` | `graphiti-core>=0.29,<0.30` | 0.29.2 |

Package names come from the official [Mem0](https://docs.mem0.ai/open-source/python-quickstart),
[LangMem](https://langchain-ai.github.io/langmem/),
[Letta](https://docs.letta.com/api/python), and
[Graphiti](https://help.getzep.com/graphiti/getting-started/quick-start) installation guides.
Release metadata was checked on each package's PyPI project page.

The released wheel sources above were downloaded and inspected for constructor, upsert, search,
delete, and response signatures. The repository contract suite is dependency-free and therefore
runs in core CI. Live Mem0, hosted Letta, and Graphiti/Neo4j integration tests are intentionally
separate because this checkout has no service credentials or graph database. LangGraph Store
implementations can be tested locally by passing the desired store to `LangMemAdapter`.
