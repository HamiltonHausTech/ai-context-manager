import copy
import json
from pathlib import Path

import pytest

from experiments.adaptive_selection.dataset import (
    DATASET_BUNDLE_VERSION,
    DatasetBundle,
    canonical_bundle_sha256,
    count_context_tokens,
    load_dataset_bundle,
    load_tiny_fixture,
    validate_tiny_fixture,
)

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_experiment.json"
FAMILY_IDS = ("hybrid-network-return-routing", "terraform-drift-state")
EXPECTED_HASH = "2b26d517ac1830f692c8d6c2581faed22997a35079922dc5cbbd4e924c54a36c"


def raw_fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def corrupt(mutator):
    payload = raw_fixture()
    mutator(payload)
    return DatasetBundle.from_dict(payload)


def test_stage0_fixture_loads_with_frozen_mechanics_and_manual_review():
    bundle = load_tiny_fixture(FIXTURE)
    assert bundle.dataset_bundle_version == DATASET_BUNDLE_VERSION == "1"
    assert bundle.schema_version == "2"
    assert bundle.family_order == FAMILY_IDS
    assert len(bundle.cases) == 6
    assert len(bundle.adaptation_feedback) == 4
    assert "mechanics only" in bundle.claim_limit.casefold()
    assert "cannot support adaptive-selection efficacy" in bundle.claim_limit.casefold()
    limitations = bundle.validation_limitations.casefold()
    assert "manual domain review" in limitations
    assert "was performed" in limitations
    assert "template siblings" in limitations
    assert "not semantic" in limitations


def test_token_accounting_is_exact_frozen_and_forces_real_tradeoffs():
    bundle = load_tiny_fixture(FIXTURE)
    assert bundle.tokenizer_config["tokenizer_id"] == "stdlib-unicode-regex"
    assert bundle.tokenizer_config["scope"] == "ContextItem.content"
    counts = []
    label_counts = []
    budgets = []
    for case in bundle.cases:
        candidates = case.inputs.candidate_context
        sealed = case.sealed_evaluation
        assert all(
            item.token_count == count_context_tokens(item.content)
            for item in candidates
        )
        assert sum(item.token_count for item in candidates) > case.inputs.token_budget
        nonnegative = set(sealed.required_context_item_ids) | set(
            sealed.useful_context_item_ids
        )
        by_id = {item.context_item_id: item for item in candidates}
        assert (
            sum(by_id[item].token_count for item in nonnegative)
            > case.inputs.token_budget
        )
        counts.append(len(candidates))
        label_counts.append(
            tuple(
                len(getattr(sealed, name))
                for name in (
                    "required_context_item_ids",
                    "useful_context_item_ids",
                    "misleading_context_item_ids",
                    "irrelevant_context_item_ids",
                )
            )
        )
        budgets.append(case.inputs.token_budget)
    assert len(set(counts)) >= 5
    assert len(set(label_counts)) >= 5
    assert len(set(budgets)) == len(budgets)


def test_confidence_and_token_ranges_overlap_all_utility_classes():
    bundle = load_tiny_fixture(FIXTURE)
    values = {label: [] for label in ("required", "useful", "misleading", "irrelevant")}
    for case in bundle.cases:
        by_id = {item.context_item_id: item for item in case.inputs.candidate_context}
        for label in values:
            ids = getattr(case.sealed_evaluation, f"{label}_context_item_ids")
            values[label].extend(
                (by_id[item].confidence, by_id[item].token_count) for item in ids
            )
    confidence_intersection = (
        max(min(value[0] for value in items) for items in values.values()),
        min(max(value[0] for value in items) for items in values.values()),
    )
    token_intersection = (
        max(min(value[1] for value in items) for items in values.values()),
        min(max(value[1] for value in items) for items in values.values()),
    )
    assert confidence_intersection[0] <= confidence_intersection[1]
    assert token_intersection[0] <= token_intersection[1]


def _budget_prefix(items, key, budget):
    selected = set()
    used = 0
    for item in sorted(items, key=key):
        if used + item.token_count <= budget:
            selected.add(item.context_item_id)
            used += item.token_count
    return selected


def test_trivial_visible_heuristics_do_not_solve_every_case():
    bundle = load_tiny_fixture(FIXTURE)
    failures = {"ordinal": 0, "confidence": 0, "shortest": 0, "source": 0}
    useful_fill_failures = dict(failures)
    for case in bundle.cases:
        items = case.inputs.candidate_context
        gold = set(case.sealed_evaluation.required_context_item_ids) | set(
            case.sealed_evaluation.useful_context_item_ids
        )
        useful = set(case.sealed_evaluation.useful_context_item_ids)
        heuristics = {
            "ordinal": _budget_prefix(
                items, lambda item: items.index(item), case.inputs.token_budget
            ),
            "confidence": _budget_prefix(
                items, lambda item: -item.confidence, case.inputs.token_budget
            ),
            "shortest": _budget_prefix(
                items, lambda item: item.token_count, case.inputs.token_budget
            ),
            "source": _budget_prefix(
                items, lambda item: item.source, case.inputs.token_budget
            ),
        }
        for name, selected in heuristics.items():
            failures[name] += selected != gold
            useful_fill_failures[name] += not useful.issubset(selected)
    assert all(value > 0 for value in failures.values())
    assert all(value > 0 for value in useful_fill_failures.values())


def test_candidate_identifiers_sources_and_visible_metadata_are_neutral():
    bundle = load_tiny_fixture(FIXTURE)
    forbidden = (
        "adapt",
        "held",
        "required",
        "useful",
        "mislead",
        "irrelevant",
        "unsafe",
        "stale",
        "destructive",
    )
    for case in bundle.cases:
        assert "case_variant" not in case.inputs.visible_metadata
        for item in case.inputs.candidate_context:
            visible_name = (item.context_item_id + " " + item.source).casefold()
            assert not any(word in visible_name for word in forbidden)
            assert set(item.metadata) == {
                "learning_attributes",
                "control_attributes",
                "format",
            }


def test_provenance_is_case_level_sealed_and_globally_disjoint():
    bundle = load_tiny_fixture(FIXTURE)
    adaptation_groups = set()
    heldout_groups = []
    for case in bundle.cases:
        group = bundle.case_provenance_groups[case.task_case_id]
        assert case.provenance == case.inputs.provenance == group
        assert {item.provenance for item in case.inputs.candidate_context} == {group}
        if case.split == "adaptation":
            adaptation_groups.add(group)
        else:
            heldout_groups.append(group)
    assert not adaptation_groups.intersection(heldout_groups)
    assert len(heldout_groups) == len(set(heldout_groups))


def test_feedback_order_labels_attributes_controls_and_traps_are_exact():
    bundle = load_tiny_fixture(FIXTURE)
    expected_cases = tuple(
        case_id for plan in bundle.family_plans for case_id in plan.adaptation_order
    )
    assert (
        tuple(event.task_case_id for event in bundle.adaptation_feedback)
        == expected_cases
    )
    cases = {case.task_case_id: case for case in bundle.cases}
    for event in bundle.adaptation_feedback:
        case = cases[event.task_case_id]
        sealed = case.sealed_evaluation
        value = event.structured_value
        positive = set(sealed.required_context_item_ids) | set(
            sealed.useful_context_item_ids
        )
        harmful = set(sealed.misleading_context_item_ids)
        assert set(value["useful_context_item_ids"]) == positive
        assert set(value["harmful_context_item_ids"]) == harmful
        assert set(event.affected_context_item_ids) == positive | harmful
        assert type(value["locked"]) is bool and value["locked"] is True
        assert type(value["selector_independent"]) is bool
        assert value["selector_independent"] is True
        assert value["no_effect_attributes"] == ("presentation:teal-header",)
        assert value["shared_feature_trap"] == "format:runbook"


def test_neutral_ontology_has_cross_split_transfer_and_cross_class_controls():
    bundle = load_tiny_fixture(FIXTURE)
    cases = {case.task_case_id: case for case in bundle.cases}
    for plan in bundle.family_plans:
        positive = set()
        harmful = set()
        for case_id in plan.adaptation_order:
            case = cases[case_id]
            by_id = {
                item.context_item_id: item for item in case.inputs.candidate_context
            }
            sealed = case.sealed_evaluation
            for item_id in (
                sealed.required_context_item_ids + sealed.useful_context_item_ids
            ):
                positive.update(by_id[item_id].metadata["learning_attributes"])
            for item_id in sealed.misleading_context_item_ids:
                harmful.update(by_id[item_id].metadata["learning_attributes"])
        held = cases[plan.held_out_case_id]
        by_id = {item.context_item_id: item for item in held.inputs.candidate_context}
        held_positive = {
            attr
            for item_id in held.sealed_evaluation.required_context_item_ids
            + held.sealed_evaluation.useful_context_item_ids
            for attr in by_id[item_id].metadata["learning_attributes"]
        }
        held_harmful = {
            attr
            for item_id in held.sealed_evaluation.misleading_context_item_ids
            for attr in by_id[item_id].metadata["learning_attributes"]
        }
        assert positive <= held_positive
        assert harmful <= held_harmful
    labels_for_teal = set()
    labels_for_runbook = set()
    for case in bundle.cases:
        sealed = case.sealed_evaluation
        labels = {
            item_id: label
            for label in ("required", "useful", "misleading", "irrelevant")
            for item_id in getattr(sealed, f"{label}_context_item_ids")
        }
        for item in case.inputs.candidate_context:
            if "presentation:teal-header" in item.metadata["control_attributes"]:
                labels_for_teal.add(labels[item.context_item_id])
            if item.metadata["format"] == "format:runbook":
                labels_for_runbook.add(labels[item.context_item_id])
    assert labels_for_teal & {"required", "useful"}
    assert labels_for_teal & {"misleading", "irrelevant"}
    assert "useful" in labels_for_runbook and "misleading" in labels_for_runbook


def test_ids_can_be_renamed_without_changing_labels_or_ontology_semantics():
    payload = raw_fixture()
    original_ontology = copy.deepcopy(payload["ontology"])
    for case_index, case in enumerate(payload["cases"]):
        mapping = {
            item["context_item_id"]: f"x{case_index}{item_index}z"
            for item_index, item in enumerate(case["inputs"]["candidate_context"])
        }
        for item in case["inputs"]["candidate_context"]:
            item["context_item_id"] = mapping[item["context_item_id"]]
        sealed = case["sealed_evaluation"]
        for key in (
            "required_context_item_ids",
            "useful_context_item_ids",
            "misleading_context_item_ids",
            "irrelevant_context_item_ids",
        ):
            sealed[key] = [mapping[item] for item in sealed[key]]
        for event in payload["adaptation_feedback"]:
            if event["task_case_id"] == case["task_case_id"]:
                event["affected_context_item_ids"] = [
                    mapping[item] for item in event["affected_context_item_ids"]
                ]
                for key in ("useful_context_item_ids", "harmful_context_item_ids"):
                    event["structured_value"][key] = [
                        mapping[item] for item in event["structured_value"][key]
                    ]
    renamed = DatasetBundle.from_dict(payload)
    validate_tiny_fixture(renamed)
    assert renamed.ontology == DatasetBundle.from_dict(raw_fixture()).ontology
    assert payload["ontology"] == original_ontology


def test_canonical_hash_round_trip_and_direct_constructor_deep_immutability(tmp_path):
    bundle = load_tiny_fixture(FIXTURE)
    assert canonical_bundle_sha256(bundle) == EXPECTED_HASH
    path = tmp_path / "round-trip.json"
    path.write_text(json.dumps(bundle.to_dict(), sort_keys=True), encoding="utf-8")
    assert canonical_bundle_sha256(load_dataset_bundle(path)) == EXPECTED_HASH

    tokenizer = bundle.to_dict()["tokenizer_config"]
    ontology = bundle.to_dict()["ontology"]
    groups = bundle.to_dict()["case_provenance_groups"]
    runs = bundle.to_dict()["planned_run_ids"]
    direct = DatasetBundle(
        **{
            **{
                key: value
                for key, value in bundle.__dict__.items()
                if key
                not in {
                    "tokenizer_config",
                    "ontology",
                    "case_provenance_groups",
                    "planned_run_ids",
                }
            },
            "tokenizer_config": tokenizer,
            "ontology": ontology,
            "case_provenance_groups": groups,
            "planned_run_ids": runs,
        }
    )
    digest = canonical_bundle_sha256(direct)
    tokenizer["scope"] = "changed"
    ontology["definitions"].clear()
    groups.clear()
    runs.clear()
    assert canonical_bundle_sha256(direct) == digest
    with pytest.raises(TypeError):
        direct.ontology["new"] = "value"


def duplicate_cross_family_content(payload):
    payload["cases"][5]["inputs"]["candidate_context"][0]["content"] = payload["cases"][
        0
    ]["inputs"]["candidate_context"][0]["content"]
    payload["cases"][5]["inputs"]["candidate_context"][0]["token_count"] = payload[
        "cases"
    ][0]["inputs"]["candidate_context"][0]["token_count"]


def duplicate_cross_family_provenance(payload):
    group = payload["case_provenance_groups"]["net-adapt-01"]
    payload["case_provenance_groups"]["tf-held-01"] = group
    payload["cases"][5]["provenance"] = group
    payload["cases"][5]["inputs"]["provenance"] = group
    for item in payload["cases"][5]["inputs"]["candidate_context"]:
        item["provenance"] = group


def duplicate_heldout_group(payload):
    group = payload["case_provenance_groups"]["net-held-01"]
    payload["case_provenance_groups"]["tf-held-01"] = group
    payload["cases"][5]["provenance"] = group
    payload["cases"][5]["inputs"]["provenance"] = group
    for item in payload["cases"][5]["inputs"]["candidate_context"]:
        item["provenance"] = group


def reordered_feedback_and_reveal(payload):
    payload["adaptation_feedback"].reverse()
    payload["reveal_order"].reverse()


def swapped_feedback_utility(payload):
    value = payload["adaptation_feedback"][0]["structured_value"]
    value["useful_context_item_ids"], value["harmful_context_item_ids"] = (
        value["harmful_context_item_ids"],
        value["useful_context_item_ids"],
    )


def empty_control(payload):
    payload["adaptation_feedback"][0]["structured_value"]["no_effect_attributes"] = []


def wrong_control_type(payload):
    payload["adaptation_feedback"][0]["structured_value"]["locked"] = 1


def false_trap(payload):
    payload["adaptation_feedback"][0]["structured_value"][
        "shared_feature_trap"
    ] = "format:json"


def feedback_attribute_mismatch(payload):
    payload["adaptation_feedback"][0]["structured_value"]["useful_attributes"].pop()


def one_character_gold(payload):
    payload["cases"][0]["sealed_evaluation"]["gold_answer"] = "x"


def one_character_rubric(payload):
    payload["cases"][0]["sealed_evaluation"]["scoring_rubric"]["instructions"] = "x"


def efficacy_overclaim(payload):
    payload["claim_limit"] = (
        "Mechanics only; cannot support adaptive-selection efficacy, but this validates efficacy and generalization."
    )


def token_mismatch(payload):
    payload["cases"][0]["inputs"]["candidate_context"][0]["token_count"] += 1


def tokenizer_hash_mismatch(payload):
    payload["tokenizer_config"]["config_hash"] = "0" * 64


def no_tradeoff_budget(payload):
    case = payload["cases"][0]
    labels = case["sealed_evaluation"]
    positive = set(
        labels["required_context_item_ids"] + labels["useful_context_item_ids"]
    )
    case["inputs"]["token_budget"] = sum(
        item["token_count"]
        for item in case["inputs"]["candidate_context"]
        if item["context_item_id"] in positive
    )


def secret_in_metadata_key(payload):
    payload["cases"][0]["inputs"]["visible_metadata"]["api_key=abcd1234"] = "redacted"


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (duplicate_cross_family_content, "duplicate content"),
        (duplicate_cross_family_provenance, "adaptation provenance group"),
        (duplicate_heldout_group, "heldout provenance groups"),
        (reordered_feedback_and_reveal, "flattened family adaptation order"),
        (swapped_feedback_utility, "exactly match sealed labels"),
        (empty_control, "no-effect attributes"),
        (wrong_control_type, "strict true"),
        (false_trap, "shared_feature_trap"),
        (feedback_attribute_mismatch, "exhaustive labeled"),
        (one_character_gold, "at least 35 words"),
        (one_character_rubric, "instructions must be substantive"),
        (efficacy_overclaim, "efficacy overclaim"),
        (token_mismatch, "token_count does not match"),
        (tokenizer_hash_mismatch, "tokenizer config hash mismatch"),
        (no_tradeoff_budget, "tradeoff"),
        (secret_in_metadata_key, "secret-like pattern"),
    ],
    ids=lambda value: getattr(value, "__name__", str(value)),
)
def test_corruption_invariants(mutator, message):
    with pytest.raises(ValueError, match=message):
        validate_tiny_fixture(corrupt(mutator))


def test_malformed_nonstring_provenance_and_ontology_raise_value_error():
    payload = raw_fixture()
    payload["case_provenance_groups"]["net-adapt-01"] = 7
    with pytest.raises(ValueError, match="case provenance groups"):
        DatasetBundle.from_dict(payload)
    payload = raw_fixture()
    payload["ontology"]["definitions"]["learning:path-observation"] = []
    with pytest.raises(ValueError, match="ontology attributes"):
        validate_tiny_fixture(DatasetBundle.from_dict(payload))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("dataset_bundle_version", None, "dataset_bundle_version is required"),
        ("dataset_bundle_version", "999", "unsupported dataset_bundle_version"),
        ("schema_version", None, "schema_version is required"),
        ("schema_version", "999", "unsupported schema_version"),
    ],
)
def test_missing_or_unsupported_versions_fail(field, value, message):
    payload = raw_fixture()
    if value is None:
        payload.pop(field)
    else:
        payload[field] = value
    with pytest.raises(ValueError, match=message):
        DatasetBundle.from_dict(payload)
