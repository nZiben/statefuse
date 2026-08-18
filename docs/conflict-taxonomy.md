# Taxonomy-aware conflicts

StateFuse keeps merge and materialization deterministic. It detects direct structured conflicts
itself and accepts pure application detectors for domain conflicts such as budgets, resource
collisions, dependency cycles, or incompatible plans. Natural-language extraction and semantic
inference belong before materialization; they should append auditable claims and derivations.

| Capability | StateFuse core | Application code |
| --- | --- | --- |
| Same-key functional value mismatch | Detects with registered value equality | Defines predicate schema/normalization |
| Context and validity | Exact context equality and half-open interval overlap | Conditional logic and context hierarchies |
| Instructions, preferences, policies, goals | Labels structured same-key mismatches as normative | Determines semantic incompatibility and authority |
| Budgets, resources, ordering, actions, duplicate work | Stores and resolves detector findings | Supplies deterministic domain detectors |
| Multi-hop contradictions | Detects conflicting derived outputs and records derivations | Performs inference and appends derived claims |
| Free text | Stores evidence and normalized claims | Extracts entities, negation, units, time, modality, and context |

## Structured applicability

Claims can declare a semantic `kind`, context, and validity interval:

```python
from statefuse import Memory, ValidityInterval

memory = Memory(replica_id="scheduler")
memory.add_claim(
    namespace="operations",
    subject="shop",
    predicate="state",
    value="open",
    confidence=0.9,
    evidence_ids=(),
    kind="fact",
    context={"location": "London"},
    validity=ValidityInterval(
        valid_from="2026-08-18T09:00:00Z",
        valid_until="2026-08-18T17:00:00Z",
    ),
)
```

Validity intervals are half-open: `valid_from <= t < valid_until`. Different values only conflict
when both context and validity overlap. Missing context dimensions apply globally.

Common kinds are `fact`, `belief`, `instruction`, `preference`, `policy`, `goal`, `plan`,
`resource`, `action`, `constraint`, and `commitment`. Direct belief mismatches are epistemic;
structured instruction, preference, policy, and goal mismatches are normative. Execution findings
must come from a detector that demonstrates the relevant domain constraint.

Use `Memory.build_view(ViewConstraints(valid_at=..., context=...))` for an applicable snapshot,
or pass the same arguments to `Memory.materialize()`.

Context matching is exact on shared dimensions; it does not infer conditions, exceptions, role
hierarchies, or counterfactual branches. Predicates are functional unless registered as
multi-valued. Authority, causal order, explicitness, human/tool subtypes, and relation/text meaning
are not inferred from labels or wall-clock timestamps. Stale/current behavior requires explicit
validity or a retraction.

An unfiltered materialization keeps one aggregate direct finding per key. Querying by context uses
the finding's incompatible-pair witnesses, and an aggregate finding is not auto-resolved. A view
also exposes non-conflicting context/time alternatives in `Projection.compatible_claims`; request a
specific context and/or time when one applicable value is needed.

## Cross-key and domain conflicts

A detector is a pure callable receiving `ConflictDetectionContext` and returning `ConflictSet`
objects. `make_conflict()` creates deterministic IDs and multi-key findings:

```python
from statefuse import ConflictDetectionContext, Memory, make_conflict


def budget_detector(context: ConflictDetectionContext):
    claims = sorted(context.claims_by_id.values(), key=lambda claim: claim.claim_id)
    capacity = next((claim for claim in claims if claim.key.predicate == "capacity"), None)
    costs = tuple(claim for claim in claims if claim.key.predicate == "cost")
    required = sum(float(claim.value) for claim in costs)
    if capacity is None or required <= float(capacity.value):
        return ()
    return (
        make_conflict(
            candidates=(capacity, *costs),
            key=capacity.key,
            conflict_type="execution.resource.capacity",
            conflict_class="execution",
            conflict_subclass="resource.capacity",
            detector_id="budget/v1",
            reason="Combined cost exceeds available capacity.",
            witness={"required": required, "available": float(capacity.value)},
        ),
    )


memory = Memory(replica_id="planner", conflict_detectors=(budget_detector,))
```

`(detector_id, conflict_type, key)` identifies one stable conflict locus. Multi-key detectors must
pass that anchor explicitly; affected `keys` may then grow without changing the reference. Use a
different anchor or detector ID for independent findings. Candidate IDs identify the current
snapshot, so a committed resolution reopens when an uncovered candidate appears.

Detectors must be deterministic and side-effect free. Do not call an LLM, network service, or wall
clock from a detector. Use ordinary Python graph traversal or arithmetic first; add a solver only
when a concrete domain requires it.

## Multi-hop claims and resolution

Inference runs outside materialization. Append its result as a normal claim and record the input
and output links with `Memory.add_derivation()`. Conflicts involving a derivation with multiple
inputs receive the `dependency_depth=multi_hop` annotation.

Committed resolution outcomes are:

- `select` (legacy default) or `replace`: one current candidate is selected.
- `preserve`: all current candidates are retained and no arbitrary winner is projected.
- `merge`: one application-created merged claim is selected; StateFuse does not synthesize it.
- `abstain`: the conflict is deferred and remains unresolved.

For `merge`, append the combined claim and a `Derivation` linking its inputs before committing the
resolution. This keeps the merge auditable instead of inventing content inside materialization.

`MemoryState.find_conflicts()` filters findings by ID, reference, type, taxonomy class/subclass,
detector, participant claim, key, namespace, context, source, and lifecycle status. `scope` selects
the lifecycle lane used for status evaluation and falls back to the global lane.

## Free text

Free-text ingestion should follow this boundary:

```text
source text -> Evidence -> normalized Claim/Derivation operations -> deterministic detectors
```

An extractor must preserve its model/rule version and confidence as provenance and normalize
entities, coreference, negation, units, time zones, modality, and context. StateFuse deliberately
does not run an extractor during merge or materialization, because replica results must not depend
on a remote or nondeterministic model.
