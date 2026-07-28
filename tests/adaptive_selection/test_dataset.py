import copy
import json
from pathlib import Path

import pytest

from experiments.adaptive_selection.dataset import (
    DATASET_BUNDLE_VERSION,
    DatasetBundle,
    canonical_bundle_sha256,
    load_dataset_bundle,
    load_tiny_fixture,
    validate_tiny_fixture,
)

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_experiment.json"
FAMILY_IDS = ("hybrid-network-return-routing", "terraform-drift-state")
EXPECTED_HASH = "34953fdffd11d93921b395a63d312d37b0f3e784aec72295c7feca352801b008"


def raw_fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def parsed_corruption(mutator):
    payload = raw_fixture()
    mutator(payload)
    return DatasetBundle.from_dict(payload)


def test_tiny_fixture_loads_with_declared_order_and_only_mechanical_claims():
    bundle = load_tiny_fixture(FIXTURE)

    assert bundle.dataset_bundle_version == DATASET_BUNDLE_VERSION == "1"
    assert bundle.schema_version == "2"
    assert bundle.family_order == FAMILY_IDS
    assert len(bundle.cases) == 6
    assert len(bundle.adaptation_feedback) == 4
    assert "mechanics only" in bundle.claim_limit.casefold()
    assert (
        "cannot support adaptive-selection efficacy claims"
        in bundle.claim_limit.casefold()
    )

    plans = {plan.task_family_id: plan for plan in bundle.family_plans}
    assert tuple(plans) == FAMILY_IDS
    for family_id in FAMILY_IDS:
        plan = plans[family_id]
        family_cases = [
            case
            for case in bundle.cases
            if case.inputs.profile.task_family_id == family_id
        ]
        assert [case.task_case_id for case in family_cases] == [
            *plan.adaptation_order,
            plan.held_out_case_id,
        ]
        assert [case.split for case in family_cases] == [
            "adaptation",
            "adaptation",
            "held_out",
        ]


def test_fixture_has_token_pressure_partitioned_labels_and_substantive_rubrics():
    bundle = load_tiny_fixture(FIXTURE)

    for case in bundle.cases:
        candidates = case.inputs.candidate_context
        sealed = case.sealed_evaluation
        label_groups = (
            sealed.required_context_item_ids,
            sealed.useful_context_item_ids,
            sealed.misleading_context_item_ids,
            sealed.irrelevant_context_item_ids,
        )
        rubric_ids = {
            criterion.criterion_id for criterion in sealed.scoring_rubric.criteria
        }

        assert 6 <= len(candidates) <= 10
        assert sum(item.token_count for item in candidates) > case.inputs.token_budget
        assert all(label_groups)
        assert set().union(*map(set, label_groups)) == {
            item.context_item_id for item in candidates
        }
        assert rubric_ids == {
            "technical_correctness",
            "required_reasoning_evidence",
            "unsafe_prohibited_actions",
        }
        assert len(sealed.gold_answer.split()) >= 35
        assert sum(c.weight for c in sealed.scoring_rubric.criteria) == pytest.approx(
            1.0
        )


def test_feedback_is_locked_selector_independent_and_adaptation_only():
    bundle = load_tiny_fixture(FIXTURE)
    cases = {case.task_case_id: case for case in bundle.cases}

    assert bundle.reveal_order == tuple(
        event.event_id for event in bundle.adaptation_feedback
    )
    assert set(bundle.planned_run_ids) == {
        case.task_case_id for case in bundle.cases if case.split == "adaptation"
    }
    assert "create matching runs before append" in bundle.runner_precondition.casefold()
    for event in bundle.adaptation_feedback:
        case = cases[event.task_case_id]
        assert case.split == "adaptation"
        assert event.source == "oracle"
        assert event.run_id == bundle.planned_run_ids[event.task_case_id]
        assert event.structured_value["locked"] is True
        assert event.structured_value["selector_independent"] is True
        assert event.structured_value["useful_attributes"]
        assert event.structured_value["harmful_attributes"]

    assert any(
        event.structured_value["no_effect_attributes"]
        for event in bundle.adaptation_feedback
    )
    assert any(
        event.structured_value["shared_feature_trap"]
        for event in bundle.adaptation_feedback
    )


def test_held_out_selector_payloads_exclude_sealed_labels_gold_and_feedback():
    bundle = load_tiny_fixture(FIXTURE)
    forbidden_keys = {
        "sealed_evaluation",
        "gold_answer",
        "scoring_rubric",
        "required_context_item_ids",
        "useful_context_item_ids",
        "misleading_context_item_ids",
        "irrelevant_context_item_ids",
        "adaptation_feedback",
        "feedback",
    }

    def keys(value):
        if isinstance(value, dict):
            return set(value).union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value), set())
        return set()

    for case in bundle.cases:
        if case.split != "held_out":
            continue
        visible = case.selector_inputs().to_dict()
        visible_json = json.dumps(visible, sort_keys=True).casefold()
        assert keys(visible).isdisjoint(forbidden_keys)
        assert case.sealed_evaluation.gold_answer.casefold() not in visible_json
        assert not any(
            event.task_case_id == case.task_case_id
            for event in bundle.adaptation_feedback
        )


def test_canonical_hash_is_complete_and_round_trip_stable(tmp_path):
    bundle = load_tiny_fixture(FIXTURE)
    digest = canonical_bundle_sha256(bundle)
    round_trip_path = tmp_path / "round-trip.json"
    round_trip_path.write_text(
        json.dumps(bundle.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    assert digest == EXPECTED_HASH
    assert canonical_bundle_sha256(load_dataset_bundle(round_trip_path)) == digest

    changed = bundle.to_dict()
    changed["description"] += " changed"
    assert canonical_bundle_sha256(DatasetBundle.from_dict(changed)) != digest


def duplicate_case_id(payload):
    payload["cases"][1]["task_case_id"] = payload["cases"][0]["task_case_id"]


def duplicate_context_id(payload):
    old = payload["cases"][1]["inputs"]["candidate_context"][0]["context_item_id"]
    duplicate = payload["cases"][0]["inputs"]["candidate_context"][0]["context_item_id"]
    payload["cases"][1]["inputs"]["candidate_context"][0]["context_item_id"] = duplicate
    payload["cases"][1]["sealed_evaluation"]["required_context_item_ids"][0] = duplicate
    event = payload["adaptation_feedback"][1]
    event["affected_context_item_ids"] = [
        duplicate if item_id == old else item_id
        for item_id in event["affected_context_item_ids"]
    ]
    event["structured_value"]["useful_context_item_ids"] = [
        duplicate if item_id == old else item_id
        for item_id in event["structured_value"]["useful_context_item_ids"]
    ]


def duplicate_event_id(payload):
    payload["adaptation_feedback"][1]["event_id"] = payload["adaptation_feedback"][0][
        "event_id"
    ]


def bad_split_count(payload):
    payload["cases"][2]["split"] = "adaptation"


def bad_declared_order(payload):
    payload["family_plans"][0]["adaptation_order"].reverse()


def feedback_on_heldout(payload):
    heldout = payload["cases"][2]
    event = payload["adaptation_feedback"][0]
    event["task_case_id"] = heldout["task_case_id"]
    event["task_family_id"] = heldout["inputs"]["profile"]["task_family_id"]


def unknown_affected_item(payload):
    payload["adaptation_feedback"][0]["affected_context_item_ids"][0] = "unknown-item"


def no_token_pressure(payload):
    case = payload["cases"][0]
    case["inputs"]["token_budget"] = sum(
        item["token_count"] for item in case["inputs"]["candidate_context"]
    )


def unpartitioned_candidate(payload):
    payload["cases"][0]["sealed_evaluation"]["irrelevant_context_item_ids"].pop()


def heldout_id_reuse(payload):
    payload["cases"][2]["inputs"]["candidate_context"][0]["context_item_id"] = payload[
        "cases"
    ][0]["inputs"]["candidate_context"][0]["context_item_id"]
    payload["cases"][2]["sealed_evaluation"]["required_context_item_ids"][0] = payload[
        "cases"
    ][2]["inputs"]["candidate_context"][0]["context_item_id"]


def provenance_group_reuse(payload):
    heldout_item = payload["cases"][2]["inputs"]["candidate_context"][0]
    adaptation_item = payload["cases"][0]["inputs"]["candidate_context"][0]
    heldout_item["metadata"]["provenance_group"] = adaptation_item["metadata"][
        "provenance_group"
    ]


def exact_duplicate_content(payload):
    payload["cases"][2]["inputs"]["candidate_context"][0]["content"] = payload["cases"][
        0
    ]["inputs"]["candidate_context"][0]["content"]


def secret_injection(payload):
    payload["cases"][0]["inputs"]["candidate_context"][0][
        "content"
    ] += " password = synthetic-but-forbidden"


def family_profile_mismatch(payload):
    payload["cases"][0]["inputs"]["profile"]["task_family_id"] = "terraform-drift-state"


def reveal_order_mismatch(payload):
    payload["reveal_order"].reverse()


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (duplicate_case_id, "task case IDs must be unique"),
        (duplicate_context_id, "context item IDs must be unique across the bundle"),
        (duplicate_event_id, "feedback event IDs must be unique"),
        (bad_split_count, "exactly two adaptation and one held_out"),
        (bad_declared_order, "declared case order"),
        (feedback_on_heldout, "feedback is permitted only for adaptation cases"),
        (unknown_affected_item, "affected context IDs must refer to that case"),
        (no_token_pressure, "candidate tokens must exceed token_budget"),
        (unpartitioned_candidate, "labels must collectively cover all candidates"),
        (heldout_id_reuse, "held_out context item IDs must be unseen"),
        (provenance_group_reuse, "held_out provenance groups must be unseen"),
        (exact_duplicate_content, "exact normalized duplicate content"),
        (secret_injection, "obvious secret-like pattern"),
        (family_profile_mismatch, "family/profile consistency"),
        (reveal_order_mismatch, "reveal_order must match feedback event order"),
    ],
    ids=lambda value: getattr(value, "__name__", str(value)),
)
def test_corrupted_copies_fail_major_tiny_fixture_invariants(mutator, message):
    with pytest.raises(ValueError, match=message):
        validate_tiny_fixture(parsed_corruption(mutator))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("dataset_bundle_version", None, "dataset_bundle_version is required"),
        ("dataset_bundle_version", "999", "unsupported dataset_bundle_version"),
        ("schema_version", None, "schema_version is required"),
        ("schema_version", "999", "unsupported schema_version"),
    ],
)
def test_missing_or_unsupported_bundle_versions_fail(field, value, message):
    payload = raw_fixture()
    if value is None:
        payload.pop(field)
    else:
        payload[field] = value

    with pytest.raises(ValueError, match=message):
        DatasetBundle.from_dict(payload)


def test_exact_duplicate_detection_does_not_claim_semantic_deduplication():
    bundle = load_tiny_fixture(FIXTURE)

    assert "near-duplicate semantic review is manual" in (
        bundle.validation_limitations.casefold()
    )
    assert "not general secret scanning" in bundle.validation_limitations.casefold()
