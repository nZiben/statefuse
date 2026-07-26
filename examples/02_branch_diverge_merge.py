from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from statefuse import InMemoryStore, Memory, ViewConstraints, materialize, merge


def main() -> None:
    store_a = InMemoryStore()
    store_b = InMemoryStore()
    mem_a = Memory(store=store_a, replica_id="agentA")
    mem_b = Memory(store=store_b, replica_id="agentB")

    source_a = mem_a.add_source(
        source_type="user_message", actor_id="alice", message_id="message-a"
    )
    source_b = mem_b.add_source(
        source_type="user_message", actor_id="bob", message_id="message-b"
    )
    evidence_a = mem_a.add_evidence(
        pointer="message://message-a", content="deadline=2026-03-25", source_id=source_a
    )
    evidence_b = mem_b.add_evidence(
        pointer="message://message-b", content="deadline=2026-03-28", source_id=source_b
    )

    claim_a = mem_a.add_claim(
        namespace="project",
        subject="deadline",
        predicate="date",
        value="2026-03-25",
        confidence=0.72,
        evidence_ids=[evidence_a],
    )
    claim_b = mem_b.add_claim(
        namespace="project",
        subject="deadline",
        predicate="date",
        value="2026-03-28",
        confidence=0.76,
        evidence_ids=[evidence_b],
    )

    merged = merge(store_a.load_oplog(), store_b.load_oplog())
    merged_state = materialize(merged)
    conflict = merged_state.conflicts[0]

    canonical = Memory(replica_id="human-review")
    canonical.merge_from(merged)
    canonical.add_resolution(
        conflict_ref=conflict.conflict_ref,
        observed_conflict_id=conflict.conflict_id,
        selected_claim_ids=[claim_a],
        rejected_claim_ids=[claim_b],
        resolution_type="human_review",
        reason="Alice confirmed the approved deadline.",
        actor_id="reviewer-1",
    )

    resolved_state = canonical.materialize()
    resolved_view = canonical.build_view(ViewConstraints())
    key = conflict.key
    assert resolved_view.selected_claims[key].claim_id == claim_a
    assert key in resolved_view.surfaced_conflicts
    assert resolved_state.lifecycle_status_by_conflict_ref_and_scope[
        (conflict.conflict_ref, None)
    ] == "resolved"

    source_c = canonical.add_source(
        source_type="tool_result", system="scheduler", message_id="result-c"
    )
    evidence_c = canonical.add_evidence(
        pointer="tool://scheduler/result-c",
        content="deadline=2026-03-30",
        source_id=source_c,
    )
    canonical.add_claim(
        namespace="project",
        subject="deadline",
        predicate="date",
        value="2026-03-30",
        confidence=0.99,
        evidence_ids=[evidence_c],
    )

    reopened_state = canonical.materialize()
    reopened_view = canonical.build_view(ViewConstraints())
    current_conflict = reopened_state.conflicts_by_ref[conflict.conflict_ref]
    assert current_conflict.conflict_id != conflict.conflict_id
    assert reopened_state.lifecycle_status_by_conflict_ref_and_scope[
        (conflict.conflict_ref, None)
    ] == "reopened"
    assert key in reopened_view.surfaced_conflicts
    assert key not in reopened_view.selected_claims

    print("Initial conflict snapshot:", conflict.conflict_id)
    print("Human-selected claim:", claim_a)
    print("Conflict remained surfaced:", key in resolved_view.surfaced_conflicts)
    print("New snapshot:", current_conflict.conflict_id)
    print("Lifecycle status: reopened")


if __name__ == "__main__":
    main()
