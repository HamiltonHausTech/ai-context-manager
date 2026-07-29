import copy
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from experiments.adaptive_selection.dataset import (
    DATASET_BUNDLE_VERSION,
    TINY_CLAIM_LIMIT,
    TINY_ONTOLOGY_CONFIG,
    DatasetBundle,
    canonical_bundle_sha256,
    count_context_tokens,
    load_dataset_bundle,
    load_tiny_fixture,
    validate_tiny_fixture,
)

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_experiment.json"
FAMILY_IDS = ("hybrid-network-return-routing", "terraform-drift-state")
EXPECTED_HASH = "05cdb3edbc96d753f44b7161dcae8812679d7776306ebb6887911dda2f7bca32"


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
    assert bundle.claim_limit == TINY_CLAIM_LIMIT
    assert bundle.to_dict()["ontology"] == TINY_ONTOLOGY_CONFIG
    config: Any = TINY_ONTOLOGY_CONFIG
    with pytest.raises(TypeError):
        config["version"] = "changed"
    with pytest.raises(TypeError):
        config["definitions"]["scope:bounded"] = "changed"
    limitations = bundle.validation_limitations.casefold()
    assert "manual domain review" in limitations
    assert "was performed" in limitations
    assert "template siblings" in limitations
    assert "not semantic" in limitations


def test_token_accounting_is_exact_frozen_with_independent_subset_budgets():
    bundle = load_tiny_fixture(FIXTURE)
    assert bundle.tokenizer_config["tokenizer_id"] == "stdlib-unicode-regex"
    assert bundle.tokenizer_config["scope"] == "ContextItem.content"
    counts = []
    deltas = []
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
        positive_total = sum(by_id[item].token_count for item in nonnegative)
        deltas.append(positive_total - case.inputs.token_budget)
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
    assert Counter(counts) == {7: 2, 8: 2, 9: 2}
    assert any(delta < 0 for delta in deltas)
    assert any(delta > 0 for delta in deltas)
    assert len(set(deltas)) >= 4
    assert len(set(label_counts)) >= 4
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
    assert len(bundle.case_provenance_groups) == len(
        set(bundle.case_provenance_groups.values())
    )


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
        assert tuple(value["useful_context_item_ids"]) == (
            sealed.required_context_item_ids + sealed.useful_context_item_ids
        )
        assert tuple(value["harmful_context_item_ids"]) == (
            sealed.misleading_context_item_ids
        )
        assert event.affected_context_item_ids == (
            tuple(value["useful_context_item_ids"])
            + tuple(value["harmful_context_item_ids"])
        )
        by_id = {item.context_item_id: item for item in case.inputs.candidate_context}
        useful_attrs = {
            attr
            for item_id in positive
            for attr in by_id[item_id].metadata["learning_attributes"]
        }
        harmful_attrs = {
            attr
            for item_id in harmful
            for attr in by_id[item_id].metadata["learning_attributes"]
        }
        assert set(value["useful_attributes"]) == useful_attrs - harmful_attrs
        assert set(value["harmful_attributes"]) == harmful_attrs - useful_attrs
        assert set(value["ambiguous_attributes"]) == useful_attrs & harmful_attrs
        assert value["ambiguous_attributes"]
        assert type(value["locked"]) is bool and value["locked"] is True
        assert type(value["selector_independent"]) is bool
        assert value["selector_independent"] is True
        assert value["no_effect_attributes"] == ("presentation:teal-header",)
        assert value["shared_feature_trap"] == "format:runbook"


def test_neutral_ontology_recurs_across_splits_classes_sources_and_metadata():
    bundle = load_tiny_fixture(FIXTURE)
    cases = {case.task_case_id: case for case in bundle.cases}
    attribute_classes = defaultdict(set)
    source_classes = defaultdict(set)
    metadata_classes = defaultdict(set)
    for case in bundle.cases:
        sealed = case.sealed_evaluation
        labels = {
            item_id: label
            for label in ("required", "useful", "misleading", "irrelevant")
            for item_id in getattr(sealed, f"{label}_context_item_ids")
        }
        for item in case.inputs.candidate_context:
            label = labels[item.context_item_id]
            for attribute in item.metadata["learning_attributes"]:
                attribute_classes[attribute].add(label)
            source_classes[item.source].add(label)
            metadata_classes[
                json.dumps(dict(item.metadata), sort_keys=True, default=list)
            ].add(label)
    assert all(len(classes) >= 2 for classes in attribute_classes.values())
    assert all(len(classes) >= 2 for classes in source_classes.values())
    assert any(len(classes) >= 2 for classes in metadata_classes.values())

    for plan in bundle.family_plans:
        adaptation_attributes = {
            attribute
            for case_id in plan.adaptation_order
            for item in cases[case_id].inputs.candidate_context
            for attribute in item.metadata["learning_attributes"]
        }
        held_attributes = {
            attribute
            for item in cases[plan.held_out_case_id].inputs.candidate_context
            for attribute in item.metadata["learning_attributes"]
        }
        assert held_attributes <= adaptation_attributes

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


def claim_effectiveness_bypass(payload):
    payload["claim_limit"] = TINY_CLAIM_LIMIT + " It demonstrates effectiveness."


def token_mismatch(payload):
    payload["cases"][0]["inputs"]["candidate_context"][0]["token_count"] += 1


def tokenizer_hash_mismatch(payload):
    payload["tokenizer_config"]["config_hash"] = "0" * 64


def all_positive_budget_deltas(payload):
    for index, case in enumerate(payload["cases"]):
        labels = case["sealed_evaluation"]
        positive = set(
            labels["required_context_item_ids"] + labels["useful_context_item_ids"]
        )
        positive_total = sum(
            item["token_count"]
            for item in case["inputs"]["candidate_context"]
            if item["context_item_id"] in positive
        )
        case["inputs"]["token_budget"] = positive_total - index - 1


def same_budget_delta(payload):
    for index, case in enumerate(payload["cases"]):
        labels = case["sealed_evaluation"]
        positive = set(
            labels["required_context_item_ids"] + labels["useful_context_item_ids"]
        )
        positive_total = sum(
            item["token_count"]
            for item in case["inputs"]["candidate_context"]
            if item["context_item_id"] in positive
        )
        delta = -1 if index % 2 else 1
        case["inputs"]["token_budget"] = positive_total - delta


def secret_in_metadata_key(payload):
    payload["cases"][0]["inputs"]["visible_metadata"]["api_key=abcd1234"] = "redacted"


def separate_secret_mapping_pair(payload):
    payload["cases"][0]["inputs"]["visible_metadata"]["api_key"] = "abcd1234"


def duplicate_feedback_attribute(payload):
    values = payload["adaptation_feedback"][0]["structured_value"]
    values["ambiguous_attributes"].append(values["ambiguous_attributes"][0])


def duplicate_affected_id(payload):
    event = payload["adaptation_feedback"][0]
    event["affected_context_item_ids"].append(event["affected_context_item_ids"][0])


def duplicate_sealed_label(payload):
    labels = payload["cases"][0]["sealed_evaluation"]
    labels["required_context_item_ids"].append(labels["required_context_item_ids"][0])


def count_position_shortcut(payload):
    first = payload["cases"][0]["sealed_evaluation"]
    paired_case = payload["cases"][3]
    paired = paired_case["sealed_evaluation"]
    moved = paired["irrelevant_context_item_ids"].pop(0)
    paired["misleading_context_item_ids"].append(moved)
    pattern = [
        label
        for item in payload["cases"][0]["inputs"]["candidate_context"]
        for label in ("required", "useful", "misleading", "irrelevant")
        if item["context_item_id"] in first[f"{label}_context_item_ids"]
    ]
    queues = {
        label: list(paired[f"{label}_context_item_ids"])
        for label in ("required", "useful", "misleading", "irrelevant")
    }
    by_id = {
        item["context_item_id"]: item
        for item in paired_case["inputs"]["candidate_context"]
    }
    paired_case["inputs"]["candidate_context"] = [
        by_id[queues[label].pop(0)] for label in pattern
    ]


def isolate_learning_attribute(payload):
    for case in payload["cases"]:
        labels = case["sealed_evaluation"]
        irrelevant = set(labels["irrelevant_context_item_ids"])
        for item in case["inputs"]["candidate_context"]:
            if item["context_item_id"] in irrelevant:
                item["metadata"]["learning_attributes"] = [
                    "basis:assumed" if value == "scope:global" else value
                    for value in item["metadata"]["learning_attributes"]
                ]


def isolate_recurring_source(payload):
    for case in payload["cases"]:
        required = set(case["sealed_evaluation"]["required_context_item_ids"])
        for item in case["inputs"]["candidate_context"]:
            if item["context_item_id"] in required:
                item["source"] = "record-mono"


def make_metadata_lookup_perfect(payload):
    item = next(
        item
        for item in payload["cases"][5]["inputs"]["candidate_context"]
        if item["context_item_id"] == "x5r9"
    )
    item["metadata"]["learning_attributes"].reverse()


def copied_generic_rubric(payload):
    rubric = payload["cases"][1]["sealed_evaluation"]["scoring_rubric"]
    generic = (
        "Review the supplied evidence carefully and award credit only for a clear, "
        "supported, safe, and complete response. Case anchors: ia-22, interfaces A "
        "and B, asymmetric-session."
    )
    rubric["instructions"] = generic
    for criterion in rubric["criteria"]:
        criterion["description"] = generic


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (duplicate_cross_family_content, "duplicate content"),
        (duplicate_cross_family_provenance, "provenance group must be unique"),
        (duplicate_heldout_group, "provenance group must be unique"),
        (reordered_feedback_and_reveal, "flattened family adaptation order"),
        (swapped_feedback_utility, "exactly match sealed labels"),
        (empty_control, "no-effect attributes"),
        (wrong_control_type, "strict true"),
        (false_trap, "shared_feature_trap"),
        (feedback_attribute_mismatch, "exactly partition"),
        (duplicate_feedback_attribute, "attribute sequences must be duplicate-free"),
        (duplicate_affected_id, "affected_context_item_ids must be unique"),
        (duplicate_sealed_label, "required_context_item_ids must be unique"),
        (one_character_gold, "at least 35 words"),
        (one_character_rubric, "instructions must be substantive"),
        (efficacy_overclaim, "exactly state mechanics-only"),
        (claim_effectiveness_bypass, "exactly state mechanics-only"),
        (token_mismatch, "token_count does not match"),
        (tokenizer_hash_mismatch, "tokenizer config hash mismatch"),
        (all_positive_budget_deltas, "shortfalls and surpluses"),
        (same_budget_delta, "varied magnitudes"),
        (count_position_shortcut, "count and ordinal position"),
        (isolate_learning_attribute, "learning attribute must span"),
        (isolate_recurring_source, "recurring source must span"),
        (make_metadata_lookup_perfect, "metadata tuple lookup"),
        (secret_in_metadata_key, "secret-like pattern"),
        (separate_secret_mapping_pair, "secret-like mapping pair"),
    ],
    ids=lambda value: getattr(value, "__name__", str(value)),
)
def test_corruption_invariants(mutator, message):
    with pytest.raises(ValueError, match=message):
        validate_tiny_fixture(corrupt(mutator))


def _set_case_provenance(payload, case_index, group):
    case = payload["cases"][case_index]
    payload["case_provenance_groups"][case["task_case_id"]] = group
    case["provenance"] = group
    case["inputs"]["provenance"] = group
    for item in case["inputs"]["candidate_context"]:
        item["provenance"] = group


def _recompute_feedback_attributes(payload, case_id):
    case = next(case for case in payload["cases"] if case["task_case_id"] == case_id)
    sealed = case["sealed_evaluation"]
    by_id = {
        item["context_item_id"]: item for item in case["inputs"]["candidate_context"]
    }
    useful_ids = sealed["required_context_item_ids"] + sealed["useful_context_item_ids"]
    harmful_ids = sealed["misleading_context_item_ids"]
    useful = {
        attribute
        for item_id in useful_ids
        for attribute in by_id[item_id]["metadata"]["learning_attributes"]
    }
    harmful = {
        attribute
        for item_id in harmful_ids
        for attribute in by_id[item_id]["metadata"]["learning_attributes"]
    }
    event = next(
        event
        for event in payload["adaptation_feedback"]
        if event["task_case_id"] == case_id
    )
    event["structured_value"]["useful_attributes"] = sorted(useful - harmful)
    event["structured_value"]["harmful_attributes"] = sorted(harmful - useful)
    event["structured_value"]["ambiguous_attributes"] = sorted(useful & harmful)


def test_candidate_count_distribution_guard_is_direct():
    payload = raw_fixture()
    case = payload["cases"][2]
    removed_id = case["sealed_evaluation"]["irrelevant_context_item_ids"].pop()
    case["inputs"]["candidate_context"] = [
        item
        for item in case["inputs"]["candidate_context"]
        if item["context_item_id"] != removed_id
    ]
    with pytest.raises(ValueError, match="7, 8, and 9 exactly twice"):
        validate_tiny_fixture(DatasetBundle.from_dict(payload))


def test_contiguous_label_role_block_guard_is_direct():
    payload = raw_fixture()
    case = payload["cases"][2]
    labels = case["sealed_evaluation"]
    ordered_ids = [
        item_id
        for label in ("required", "useful", "misleading", "irrelevant")
        for item_id in labels[f"{label}_context_item_ids"]
    ]
    by_id = {
        item["context_item_id"]: item for item in case["inputs"]["candidate_context"]
    }
    case["inputs"]["candidate_context"] = [by_id[item_id] for item_id in ordered_ids]
    with pytest.raises(ValueError, match="contiguous role blocks"):
        validate_tiny_fixture(DatasetBundle.from_dict(payload))


def test_total_candidate_tokens_must_exceed_budget_directly():
    payload = raw_fixture()
    case = payload["cases"][0]
    case["inputs"]["token_budget"] = sum(
        item["token_count"] for item in case["inputs"]["candidate_context"]
    )
    with pytest.raises(ValueError, match="candidate tokens must exceed token_budget"):
        validate_tiny_fixture(DatasetBundle.from_dict(payload))


def test_heldout_attribute_coverage_guard_is_direct():
    payload = raw_fixture()
    for case in payload["cases"][:2]:
        for item in case["inputs"]["candidate_context"]:
            attributes = item["metadata"]["learning_attributes"]
            item["metadata"]["learning_attributes"] = list(
                dict.fromkeys(
                    "scope:bounded" if value == "scope:global" else value
                    for value in attributes
                )
            )
        _recompute_feedback_attributes(payload, case["task_case_id"])
    with pytest.raises(
        ValueError,
        match="heldout learning attributes must be represented in adaptation",
    ):
        validate_tiny_fixture(DatasetBundle.from_dict(payload))


def test_missing_ambiguous_partition_guard_is_direct_and_duplicate_free():
    payload = raw_fixture()
    payload["adaptation_feedback"][0]["structured_value"]["ambiguous_attributes"] = []
    with pytest.raises(
        ValueError, match="exactly partition labeled learning attributes"
    ):
        validate_tiny_fixture(DatasetBundle.from_dict(payload))


def test_duplicate_token_budget_guard_is_direct():
    payload = raw_fixture()
    payload["cases"][1]["inputs"]["token_budget"] = payload["cases"][3]["inputs"][
        "token_budget"
    ]
    with pytest.raises(ValueError, match="token budgets must be independent"):
        validate_tiny_fixture(DatasetBundle.from_dict(payload))


def test_same_split_provenance_reuse_is_rejected_globally():
    payload = raw_fixture()
    group = payload["case_provenance_groups"]["net-adapt-01"]
    _set_case_provenance(payload, 1, group)
    with pytest.raises(ValueError, match="provenance group must be unique for all six"):
        validate_tiny_fixture(DatasetBundle.from_dict(payload))


def test_same_split_normalized_content_duplicate_is_rejected_globally():
    payload = raw_fixture()
    source = next(
        item
        for item in payload["cases"][0]["inputs"]["candidate_context"]
        if item["context_item_id"] == "p5d0"
    )
    target = next(
        item
        for item in payload["cases"][1]["inputs"]["candidate_context"]
        if item["context_item_id"] == "a9t3"
    )
    assert source["token_count"] == target["token_count"] == 23
    target["content"] = "  " + source["content"].swapcase() + "  "
    with pytest.raises(ValueError, match="candidate content must be globally unique"):
        validate_tiny_fixture(DatasetBundle.from_dict(payload))


def test_overlap_preserving_teal_control_substitution_hits_exact_pin():
    payload = raw_fixture()
    for case in payload["cases"]:
        for item in case["inputs"]["candidate_context"]:
            controls = item["metadata"]["control_attributes"]
            item["metadata"]["control_attributes"] = [
                (
                    "presentation:plain-header"
                    if value == "presentation:teal-header"
                    else "presentation:teal-header"
                )
                for value in controls
            ]
    for event in payload["adaptation_feedback"]:
        event["structured_value"]["no_effect_attributes"] = [
            "presentation:plain-header"
        ]
    with pytest.raises(
        ValueError, match="no_effect_attributes must exactly equal the teal control"
    ):
        validate_tiny_fixture(DatasetBundle.from_dict(payload))


def test_genuinely_shared_trap_substitution_hits_exact_pin():
    payload = raw_fixture()
    event = payload["adaptation_feedback"][3]
    assert event["task_case_id"] == "tf-adapt-02"
    event["structured_value"]["shared_feature_trap"] = "format:note"
    with pytest.raises(
        ValueError, match="shared_feature_trap must exactly equal format:runbook"
    ):
        validate_tiny_fixture(DatasetBundle.from_dict(payload))


def test_generic_rubric_with_all_destination_anchors_hits_exact_pin():
    payload = raw_fixture()
    copied_generic_rubric(payload)
    rubric = payload["cases"][1]["sealed_evaluation"]["scoring_rubric"]
    assert all(
        anchor.casefold() in text.casefold()
        for text in [rubric["instructions"]]
        + [criterion["description"] for criterion in rubric["criteria"]]
        for anchor in ("ia-22", "interfaces A and B", "asymmetric-session")
    )
    with pytest.raises(ValueError, match="frozen per-case rubric fingerprint"):
        validate_tiny_fixture(DatasetBundle.from_dict(payload))


def _recompute_config_hash(config):
    body = {key: value for key, value in config.items() if key != "config_hash"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"))
    config["config_hash"] = hashlib.sha256(encoded.encode()).hexdigest()


@pytest.mark.parametrize("field", ["ontology_id", "version", "definitions"])
def test_ontology_exact_pin_rejects_recomputed_id_version_or_definition(field):
    payload = raw_fixture()
    if field == "definitions":
        payload["ontology"][field]["scope:bounded"] += " Altered."
    else:
        payload["ontology"][field] += "-altered"
    _recompute_config_hash(payload["ontology"])
    with pytest.raises(ValueError, match="exactly match the frozen"):
        validate_tiny_fixture(DatasetBundle.from_dict(payload))


def test_malformed_nonstring_provenance_and_ontology_raise_value_error():
    payload = raw_fixture()
    payload["case_provenance_groups"]["net-adapt-01"] = 7
    with pytest.raises(ValueError, match="case provenance groups"):
        DatasetBundle.from_dict(payload)
    payload = raw_fixture()
    payload["ontology"]["definitions"]["scope:bounded"] = []
    _recompute_config_hash(payload["ontology"])
    with pytest.raises(ValueError, match="exactly match the frozen"):
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
