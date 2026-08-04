import ast
import copy
import hashlib
import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest

from experiments.adaptive_selection.context_sensitivity_calibration import (
    CONDITIONS,
    CONTRACT_PATH,
    CONTRACT_STATUS,
    CONTRACT_VERSION,
    EXECUTION_ORDER,
    EXPECTED_EVIDENCE_LINE_TEMPLATE,
    EXPECTED_INSTRUCTIONS,
    EXPECTED_SCHEMA,
    EXPECTED_TEMPLATE,
    PINNED_CONTRACT_SHA256,
    ContractValidationError,
    build_dry_run_summary,
    canonical_bytes,
    load_contract,
    main,
    project_cost,
    render_requests,
    score_annotations,
    validate_contract,
)

ROOT = Path(__file__).parents[2]
CANONICAL_CONTRACT = ROOT / CONTRACT_PATH
MODULE = ROOT / "experiments/adaptive_selection/context_sensitivity_calibration.py"


def contract():
    return load_contract(CANONICAL_CONTRACT)[0]


def cloned():
    return copy.deepcopy(contract())


def assert_invalid(value, match=None):
    with pytest.raises(ContractValidationError, match=match):
        validate_contract(value)


def test_canonical_contract_validates_and_renders_exactly_nine_cells():
    frozen, raw = load_contract(CANONICAL_CONTRACT)
    requests = render_requests(frozen)

    assert frozen["contract_version"] == CONTRACT_VERSION
    assert frozen["status"] == CONTRACT_STATUS
    assert len(raw) == CANONICAL_CONTRACT.stat().st_size
    assert hashlib.sha256(raw).hexdigest() == PINNED_CONTRACT_SHA256
    assert len(requests) == 9
    assert all("metadata" not in body for body in requests)


def test_pinned_hash_rejects_evidence_provenance_mutation(tmp_path):
    value = cloned()
    value["scenarios"][0]["cells"][0]["evidence"][0]["source_ref"] = "unrelated locator"
    mutated = tmp_path / "mutated.json"
    mutated.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(ContractValidationError, match="pinned review candidate"):
        load_contract(mutated)


def test_rendering_is_deterministic_and_returns_independent_copies():
    frozen = contract()
    first = render_requests(frozen)
    second = render_requests(frozen)

    assert canonical_bytes(first) == canonical_bytes(second)
    first[0]["text"]["format"]["schema"]["properties"]["diagnosis"]["maxLength"] = 1
    assert second[0]["text"]["format"] == EXPECTED_SCHEMA
    assert render_requests(frozen) == second


def test_response_schema_uses_supported_bounds_and_fits_output_budget():
    encoded = json.dumps(EXPECTED_SCHEMA, sort_keys=True)
    assert "minLength" not in encoded
    assert "maxLength" not in encoded
    assert "uniqueItems" not in encoded
    properties = EXPECTED_SCHEMA["schema"]["properties"]
    assert properties["diagnosis"]["pattern"] == r"^[ -~]{1,400}$"
    assert properties["supporting_evidence_numbers"]["maxItems"] == 5
    assert properties["missing_evidence"]["maxItems"] == 3
    assert properties["next_safe_actions"]["maxItems"] == 3
    assert properties["actions_to_avoid"]["maxItems"] == 3
    assert EXPECTED_TEMPLATE["max_output_tokens"] == 2048
    worst_case = {
        "diagnosis": "~" * 400,
        "supporting_evidence_numbers": [1, 2, 3, 4, 5],
        "missing_evidence": ["~" * 120] * 3,
        "confidence": "medium",
        "next_safe_actions": ["~" * 160] * 3,
        "actions_to_avoid": ["~" * 160] * 3,
    }
    assert len(canonical_bytes(worst_case)) <= EXPECTED_TEMPLATE["max_output_tokens"]


def test_exact_request_isolation_and_shape():
    frozen = contract()
    lookup = {
        cell["cell_id"]: (scenario, cell)
        for scenario in frozen["scenarios"]
        for cell in scenario["cells"]
    }
    for cell_id, body in zip(EXECUTION_ORDER, render_requests(frozen)):
        scenario, cell = lookup[cell_id]
        numbered = "\n".join(
            EXPECTED_EVIDENCE_LINE_TEMPLATE.format(
                index=index,
                observed_at=item["observed_at"],
                content=item["content"],
            )
            for index, item in enumerate(cell["evidence"], 1)
        )
        assert body["instructions"] == EXPECTED_INSTRUCTIONS
        assert body["input"] == "Task:\n{}\n\nEvidence:\n{}".format(
            scenario["task_prompt"], numbered
        )
        assert body["text"]["format"] == EXPECTED_SCHEMA
        assert set(body) == {
            "model",
            "max_output_tokens",
            "parallel_tool_calls",
            "reasoning",
            "service_tier",
            "store",
            "stream",
            "text",
            "tools",
            "truncation",
            "instructions",
            "input",
        }
        visible = canonical_bytes(body).decode("utf-8").casefold()
        for forbidden in (
            "source_role",
            "source_provenance",
            "rubric",
            "forbidden_prompt_phrases",
            "decision_rule",
            "execution_order",
            *CONDITIONS,
        ):
            assert forbidden.casefold() not in visible
        for other_scenario in frozen["scenarios"]:
            for other_cell in other_scenario["cells"]:
                for evidence in other_cell["evidence"]:
                    assert evidence["evidence_id"].casefold() not in visible
                    assert evidence["source_ref"].casefold() not in visible
        assert all(item["content"] in body["input"] for item in cell["evidence"])
        assert all(item["observed_at"] in body["input"] for item in cell["evidence"])


@pytest.mark.parametrize("field", ["contract_version", "status"])
def test_exact_version_and_status_are_required(field):
    value = cloned()
    value[field] += "-changed"
    assert_invalid(value)


def test_contract_byte_drift_is_rejected_before_structural_validation(tmp_path):
    target = tmp_path / "contract.json"
    target.write_bytes(CANONICAL_CONTRACT.read_bytes() + b"\n")
    with pytest.raises(ContractValidationError, match="pinned review candidate"):
        load_contract(target)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda value: value["rendering"].update(
                evidence_line_template="{index}. {content}"
            ),
            "evidence line template",
        ),
        (
            lambda value: value["assessment"]["criterion_status_scores"].update(
                partially_met="0.75"
            ),
            "assessment",
        ),
        (
            lambda value: value["decision_rule"]["parameters"].update(
                minimum_passing_families_for_continue=1
            ),
            "decision rule parameters",
        ),
    ],
)
def test_timestamp_rendering_scoring_and_decision_rules_are_frozen(mutation, match):
    value = cloned()
    mutation(value)
    assert_invalid(value, match)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["scenarios"].pop(),
        lambda value: value["scenarios"][0]["cells"].pop(),
        lambda value: value["scenarios"][0]["cells"][0].update(
            condition="withheld_context"
        ),
        lambda value: value["scenarios"][1].update(
            family=value["scenarios"][0]["family"]
        ),
        lambda value: value["scenarios"][0]["cells"][0].update(
            cell_id=value["scenarios"][1]["cells"][0]["cell_id"]
        ),
        lambda value: value["execution_order"].reverse(),
        lambda value: value["execution_order"].pop(),
    ],
)
def test_design_cardinality_uniqueness_and_fixed_order_mutations_fail(mutation):
    value = cloned()
    mutation(value)
    assert_invalid(value)


@pytest.mark.parametrize(
    "field",
    [
        "condition_labels_forbidden_in_requests",
        "correct_requires_decisive_evidence",
        "former_consulting_and_day_job_material_forbidden",
        "polished_wiki_or_assistant_conclusions_forbidden",
        "primary_artifacts_only",
        "source_roles_forbidden_in_requests",
        "stale_requires_historical_evidence_and_forbids_decisive_evidence",
        "withheld_forbids_decisive_evidence",
    ],
)
def test_every_anti_taint_gate_must_remain_true(field):
    value = cloned()
    value["anti_taint"][field] = False
    assert_invalid(value, "anti-taint")


def _cell(value, condition):
    return next(
        cell
        for scenario in value["scenarios"]
        for cell in scenario["cells"]
        if cell["condition"] == condition
    )


@pytest.mark.parametrize(
    "condition,new_role,error",
    [
        ("correct_context", "supporting", "requires decisive"),
        ("withheld_context", "decisive", "forbid decisive"),
        ("stale_context", "decisive", "forbid decisive"),
    ],
)
def test_decisive_role_rules_fail_under_mutation(condition, new_role, error):
    value = cloned()
    target = _cell(value, condition)
    for evidence in target["evidence"]:
        evidence["source_role"] = new_role
    assert_invalid(value, error)


def test_stale_requires_historical_evidence():
    value = cloned()
    target = _cell(value, "stale_context")
    for evidence in target["evidence"]:
        evidence["source_role"] = "supporting"
    assert_invalid(value, "requires historical")


def test_evidence_ids_must_be_globally_unique():
    value = cloned()
    value["scenarios"][1]["cells"][0]["evidence"][0]["evidence_id"] = value[
        "scenarios"
    ][0]["cells"][0]["evidence"][0]["evidence_id"]
    assert_invalid(value, "globally unique")


def test_cell_evidence_count_must_fit_numbered_citation_schema():
    value = cloned()
    target = value["scenarios"][0]["cells"][0]
    extra = copy.deepcopy(target["evidence"][0])
    extra["evidence_id"] = "mh-c-6"
    extra["source_ref"] = "synthetic mutation source"
    target["evidence"].append(extra)
    assert_invalid(value, "at most five")


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda value: value["scenarios"][0].update(source_provenance=[]),
            "provenance",
        ),
        (
            lambda value: value["scenarios"][0]["source_provenance"][0].pop(
                "sanitization"
            ),
            "provenance",
        ),
        (
            lambda value: value["scenarios"][0]["rubric"]["criteria"][0].update(
                weight="0.34"
            ),
            "sum exactly",
        ),
        (
            lambda value: value["scenarios"][0]["rubric"]["criteria"][0].update(
                weight="NaN"
            ),
            "positive and finite",
        ),
    ],
)
def test_provenance_and_exact_rubric_weight_rules(mutation, match):
    value = cloned()
    mutation(value)
    assert_invalid(value, match)


def test_each_rubric_requires_one_nonempty_anchor_per_condition():
    value = cloned()
    del value["scenarios"][0]["rubric"]["condition_anchors"]["withheld_context"]
    assert_invalid(value, "one nonempty anchor")

    value = cloned()
    value["scenarios"][0]["rubric"]["condition_anchors"]["stale_context"] = ""
    assert_invalid(value, "one nonempty anchor")


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda value: value["execution_authority"].update(
                network_requests_authorized_by_this_contract=1
            ),
            "execution_authority",
        ),
        (
            lambda value: value["execution_authority"].update(
                future_execution_requires_owner_approval=False
            ),
            "execution_authority",
        ),
        (
            lambda value: value["execution_authority"].update(
                future_execution_requires_independent_scientific_and_security_review=False
            ),
            "execution_authority",
        ),
        (
            lambda value: value["provider_configuration"].update(max_retries=1),
            "provider_configuration",
        ),
        (
            lambda value: value["provider_configuration"].update(timeout_seconds=29.0),
            "provider_configuration",
        ),
        (
            lambda value: value["provider_configuration"].update(
                requested_model="gpt-5.6"
            ),
            "provider_configuration",
        ),
        (
            lambda value: value["provider_configuration"][
                "request_body_template"
            ].update(store=True),
            "provider_configuration",
        ),
        (
            lambda value: value["provider_configuration"][
                "request_body_template"
            ].update(tools=[{"type": "web_search"}]),
            "provider_configuration",
        ),
    ],
)
def test_authority_review_and_provider_configuration_are_frozen(mutation, match):
    value = cloned()
    mutation(value)
    assert_invalid(value, match)


def test_strict_response_schema_mutation_fails():
    value = cloned()
    value["provider_configuration"]["request_body_template"]["text"]["format"][
        "strict"
    ] = False
    assert_invalid(value, "provider_configuration")


def test_blind_assessment_and_deterministic_status_scores_are_frozen():
    value = cloned()
    value["assessment"]["criterion_status_scores"]["partially_met"] = "0.75"
    assert_invalid(value, "assessment")


@pytest.mark.parametrize(
    "text,match",
    [
        ("The whole host, not just the lock", "conclusion phrase"),
        ("rubric", "internal field"),
        ("correct_context", "condition label"),
        ('The source label is "decisive".', "source-role label"),
        ("This exposed SK-PRIVATE material", "private identifier"),
        ("private-ssid-canary", "private identifier"),
    ],
)
def test_provider_visible_anti_taint_rules_are_case_insensitive(text, match):
    value = cloned()
    value["scenarios"][0]["cells"][0]["evidence"][0]["content"] = text
    assert_invalid(value, match)


def test_evidence_identifier_cannot_leak_through_content():
    value = cloned()
    item = value["scenarios"][0]["cells"][0]["evidence"][0]
    item["content"] = "Accidental label " + item["evidence_id"]
    assert_invalid(value, "Evidence ID|evidence ID")


def test_foreign_scenario_identifier_and_source_reference_cannot_leak():
    value = cloned()
    foreign = value["scenarios"][1]["cells"][0]["evidence"][0]
    target = value["scenarios"][0]["cells"][0]["evidence"][0]
    target["content"] = "Accidental foreign label " + foreign["evidence_id"]
    assert_invalid(value, "Evidence ID|evidence ID")

    value = cloned()
    foreign = value["scenarios"][1]["cells"][0]["evidence"][0]
    target = value["scenarios"][0]["cells"][0]["evidence"][0]
    target["content"] = "Accidental foreign locator " + foreign["source_ref"]
    assert_invalid(value, "source reference")


def test_foreign_scenario_conclusion_phrase_cannot_leak():
    value = cloned()
    foreign_phrase = value["scenarios"][1]["forbidden_prompt_phrases"][0]
    value["scenarios"][0]["cells"][0]["evidence"][0]["content"] = foreign_phrase
    assert_invalid(value, "forbidden conclusion phrase")


def test_cost_projection_uses_utf8_bytes_overhead_output_cap_and_decimal_rates():
    frozen = contract()
    requests = render_requests(frozen)
    projection = project_cost(frozen, requests)
    input_rate = Decimal(frozen["budget"]["provisional_input_per_million"])
    output_rate = Decimal(frozen["budget"]["provisional_output_per_million"])
    expected_total = Decimal(0)

    for body, cell in zip(requests, projection["cells"]):
        byte_count = len(canonical_bytes(body))
        assert cell["request_bytes"] == byte_count
        assert cell["projected_input_tokens"] == byte_count + 1024
        assert cell["projected_output_tokens"] == 2048
        expected_total += (
            Decimal(byte_count + 1024) * input_rate + Decimal(2048) * output_rate
        ) / Decimal(1_000_000)
    assert Decimal(projection["total_projected_max_cost"]) == expected_total
    assert expected_total <= Decimal("1.00")


def test_cost_cap_is_enforced():
    value = cloned()
    value["budget"]["provisional_output_per_million"] = "1000.00"
    validate_contract(value)
    with pytest.raises(ContractValidationError, match="exceeds"):
        project_cost(value, render_requests(value))


def test_budget_and_request_output_caps_must_match():
    value = cloned()
    value["budget"]["max_output_tokens_per_cell"] = 900
    assert_invalid(value, "output-token cap")


def _annotations_for_passing_scenarios(
    frozen, passing_scenario_ids, *, critical_cell_id=None
):
    records = []
    for scenario in frozen["scenarios"]:
        criterion_ids = [
            item["criterion_id"] for item in scenario["rubric"]["criteria"]
        ]
        passes = scenario["scenario_id"] in passing_scenario_ids
        for cell in scenario["cells"]:
            status = (
                "met"
                if cell["condition"] == "correct_context" or not passes
                else "not_met"
            )
            records.append(
                {
                    "cell_id": cell["cell_id"],
                    "criteria": {
                        criterion_id: status for criterion_id in criterion_ids
                    },
                    "critical_finding": cell["cell_id"] == critical_cell_id,
                }
            )
    return records


def test_scoring_derives_continue_narrow_and_stop_without_post_hoc_math():
    frozen = contract()
    scenario_ids = [item["scenario_id"] for item in frozen["scenarios"]]
    assert (
        score_annotations(
            frozen, _annotations_for_passing_scenarios(frozen, set(scenario_ids[:2]))
        )["verdict"]
        == "continue"
    )
    assert (
        score_annotations(
            frozen, _annotations_for_passing_scenarios(frozen, {scenario_ids[0]})
        )["verdict"]
        == "narrow"
    )
    assert (
        score_annotations(frozen, _annotations_for_passing_scenarios(frozen, set()))[
            "verdict"
        ]
        == "stop_or_redesign_once"
    )


def test_correct_context_critical_finding_caps_score_and_forces_stop():
    frozen = contract()
    scenario_ids = [item["scenario_id"] for item in frozen["scenarios"]]
    correct_cell = frozen["scenarios"][0]["cells"][0]["cell_id"]
    result = score_annotations(
        frozen,
        _annotations_for_passing_scenarios(
            frozen, set(scenario_ids), critical_cell_id=correct_cell
        ),
    )
    scored = next(item for item in result["cells"] if item["cell_id"] == correct_cell)
    assert scored["raw_score"] == "1.000"
    assert scored["normalized_score"] == "0.200"
    assert result["verdict"] == "stop_or_redesign_once"


def test_calibrated_uncertainty_earns_safety_but_not_resolution_credit():
    frozen = contract()
    safety_criteria = {
        "personal-host-network-v1": {"uncertainty", "safe_action", "avoid_harm"},
        "provider-packaging-v1": {"authority", "evidence_integrity", "uncertainty"},
        "git-main-divergence-v1": {"uncertainty"},
    }
    records = _annotations_for_passing_scenarios(frozen, set())
    cells = {
        cell["cell_id"]: (scenario["scenario_id"], cell["condition"])
        for scenario in frozen["scenarios"]
        for cell in scenario["cells"]
    }
    for record in records:
        scenario_id, condition = cells[record["cell_id"]]
        credited = (
            set(record["criteria"])
            if condition == "correct_context"
            else safety_criteria[scenario_id]
        )
        record["criteria"] = {
            criterion_id: "met" if criterion_id in credited else "unresolved"
            for criterion_id in record["criteria"]
        }

    result = score_annotations(frozen, records)
    assert result["verdict"] == "continue"
    assert result["passing_family_count"] == 3


def test_scoring_rejects_incomplete_or_unfrozen_annotations():
    frozen = contract()
    records = _annotations_for_passing_scenarios(frozen, set())
    with pytest.raises(ContractValidationError, match="exactly nine"):
        score_annotations(frozen, records[:-1])
    malformed = copy.deepcopy(records)
    malformed[0].clear()
    with pytest.raises(ContractValidationError, match="frozen field set"):
        score_annotations(frozen, malformed)
    records[0]["criteria"][next(iter(records[0]["criteria"]))] = "invented"
    with pytest.raises(ContractValidationError, match="not frozen"):
        score_annotations(frozen, records)


def test_dry_run_summary_is_deterministic_and_contains_no_prompt_content():
    first = build_dry_run_summary(CANONICAL_CONTRACT)
    second = build_dry_run_summary(CANONICAL_CONTRACT)
    assert first == second
    assert first["mode"] == "offline_dry_run"
    assert first["network_requests_authorized"] == 0
    assert first["cell_count"] == 9
    assert first["contract_sha256"].startswith("sha256:")
    assert all(cell["request_sha256"].startswith("sha256:") for cell in first["cells"])
    serialized = json.dumps(first)
    assert "front-door" not in serialized
    assert "Task:" not in serialized
    assert "Evidence:" not in serialized
    assert '"instructions":' not in serialized
    assert '"input":' not in serialized


def test_module_has_no_network_or_execution_surface():
    source = MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots = set()
    function_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function_names.add(node.name)
    assert "openai" not in imported_roots
    assert "httpx" not in imported_roots
    assert "requests" not in imported_roots
    assert "urllib" not in imported_roots
    assert not any(
        token in name.casefold()
        for name in function_names
        for token in ("execute", "dispatch", "network", "provider_call", "live")
    )
    assert "api.openai.com" not in source


def test_cli_defaults_to_prompt_free_dry_run(capsys):
    assert main(["--contract", str(CANONICAL_CONTRACT)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output == build_dry_run_summary(CANONICAL_CONTRACT)


def test_cli_offers_no_live_mode():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.adaptive_selection.context_sensitivity_calibration",
            "--live",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "unrecognized arguments: --live" in result.stderr
    assert result.stdout == ""
