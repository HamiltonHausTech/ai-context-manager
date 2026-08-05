import ast
import copy
import hashlib
import json
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
from pathlib import Path

import pytest

from experiments.adaptive_selection.context_sensitivity_calibration import (
    EXECUTION_ORDER,
    PINNED_CONTRACT_SHA256 as V1_CONTRACT_SHA256,
    canonical_bytes,
    load_contract as load_v1_contract,
    render_requests as render_v1_requests,
)
from experiments.adaptive_selection.context_sensitivity_replication_v2 import (
    CONTRACT_PATH,
    CONTRACT_VERSION,
    PINNED_CONTRACT_SHA256,
    V1_RENDERER_SHA256,
    ContractValidationError,
    analyze_family,
    bootstrap_percentile_interval,
    build_dry_run_summary,
    build_schedule,
    exact_one_sided_permutation_pvalue,
    holm_correction,
    load_contract,
    main,
    project_maximum_cost,
    render_unit_requests,
    score_replication,
    validate_contract,
)

ROOT = Path(__file__).parents[2]
CANONICAL_CONTRACT = ROOT / CONTRACT_PATH
V1_MODULE = ROOT / "experiments/adaptive_selection/context_sensitivity_calibration.py"
MODULE = ROOT / "experiments/adaptive_selection/context_sensitivity_replication_v2.py"


def contract():
    return load_contract(CANONICAL_CONTRACT)[0]


def schedule():
    return build_schedule(contract())


def outcomes(*, missing=()):
    missing = set(missing)
    return [
        {"unit_id": unit.unit_id, "schema_valid": unit.unit_id not in missing}
        for unit in schedule()
    ]


def annotations(passing_families=3, *, missing=(), critical=(), locked=True):
    frozen = contract()
    v1 = load_v1_contract(ROOT / frozen["lineage"]["predecessor_contract_path"])[0]
    pass_names = {scenario["family"] for scenario in v1["scenarios"][:passing_families]}
    by_cell = {
        cell["cell_id"]: (scenario, cell)
        for scenario in v1["scenarios"]
        for cell in scenario["cells"]
    }
    records = []
    for unit in schedule():
        if unit.unit_id in set(missing):
            continue
        scenario, cell = by_cell[unit.base_cell_id]
        criterion_ids = [x["criterion_id"] for x in scenario["rubric"]["criteria"]]
        family_passes = scenario["family"] in pass_names
        status = (
            "met"
            if cell["condition"] == "correct_context" or not family_passes
            else "not_met"
        )
        records.append(
            {
                "unit_id": unit.unit_id,
                "criteria": {criterion_id: status for criterion_id in criterion_ids},
                "critical_finding": unit.unit_id in set(critical),
                "locked": locked,
                "evidence_coherent": True,
            }
        )
    return records


def _score(outcome_records, resolved_records):
    return score_replication(
        contract(),
        outcome_records,
        resolved_records,
        annotation_lock_verified=True,
    )


def test_v1_frozen_hashes_and_request_hashes_remain_unchanged():
    assert (
        hashlib.sha256(
            (
                ROOT
                / "experiments/adaptive_selection/controls/task12b_context_calibration_v1.json"
            ).read_bytes()
        ).hexdigest()
        == V1_CONTRACT_SHA256
    )
    assert hashlib.sha256(V1_MODULE.read_bytes()).hexdigest() == V1_RENDERER_SHA256
    frozen = contract()
    expected = {
        cell["base_cell_id"]: cell["request_sha256"] for cell in frozen["base_cells"]
    }
    v1, _ = load_v1_contract()
    actual = {
        cell_id: hashlib.sha256(canonical_bytes(body)).hexdigest()
        for cell_id, body in zip(EXECUTION_ORDER, render_v1_requests(v1))
    }
    assert actual == expected


def test_contract_hash_lineage_and_mutation_are_exactly_frozen(tmp_path):
    frozen, raw = load_contract(CANONICAL_CONTRACT)
    assert frozen["contract_version"] == CONTRACT_VERSION
    assert hashlib.sha256(raw).hexdigest() == PINNED_CONTRACT_SHA256
    assert frozen["lineage"]["predecessor_raw_contract_sha256"] == V1_CONTRACT_SHA256
    assert frozen["lineage"]["predecessor_renderer_sha256"] == V1_RENDERER_SHA256
    mutated = copy.deepcopy(frozen)
    mutated["statistics"]["material_advantage_threshold"] = "0.19"
    with pytest.raises(ContractValidationError, match="pinned"):
        validate_contract(mutated)
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(ContractValidationError, match="pinned"):
        load_contract(path)


def test_schedule_has_45_immutable_unique_units_and_five_draws_per_cell():
    units = schedule()
    assert len(units) == len({unit.unit_id for unit in units}) == 45
    assert {unit.base_cell_id for unit in units} == set(EXECUTION_ORDER)
    for cell_id in EXECUTION_ORDER:
        assert [u.draw_index for u in units if u.base_cell_id == cell_id] == [
            1,
            2,
            3,
            4,
            5,
        ]
    with pytest.raises(FrozenInstanceError):
        units[0].draw_index = 9


def test_each_request_hash_occurs_five_times_and_requests_are_byte_identical_to_v1():
    frozen = contract()
    units = build_schedule(frozen)
    bodies = render_unit_requests(frozen, units)
    v1, _ = load_v1_contract()
    base = dict(zip(EXECUTION_ORDER, render_v1_requests(v1)))
    for cell in frozen["base_cells"]:
        matching = [u for u in units if u.base_cell_id == cell["base_cell_id"]]
        assert len(matching) == 5
        assert {u.request_sha256 for u in matching} == {cell["request_sha256"]}
        assert {u.base_request_sha256 for u in matching} == {cell["request_sha256"]}
        assert {u.provider_request_sha256 for u in matching} == {cell["request_sha256"]}
    for unit, body in zip(units, bodies):
        assert canonical_bytes(body) == canonical_bytes(base[unit.base_cell_id])


def test_client_metadata_cannot_change_provider_bytes():
    frozen = contract()
    unit = schedule()[0]
    changed = replace(unit, unit_id="client-only-alias", draw_index=99)
    first = render_unit_requests(frozen, [unit])[0]
    second = render_unit_requests(frozen, [changed])[0]
    assert canonical_bytes(first) == canonical_bytes(second)
    visible = canonical_bytes(first).decode("utf-8")
    assert unit.unit_id not in visible
    assert "draw_index" not in visible


def test_dry_run_is_untainted_hash_count_cost_output_only():
    summary = build_dry_run_summary(CANONICAL_CONTRACT)
    assert summary["network_requests_authorized"] == 0
    assert summary["unit_count"] == 45
    assert summary["base_cell_count"] == 9
    serialized = json.dumps(summary)
    for taint in (
        "Task:",
        "Evidence:",
        "rubric",
        "criteria",
        "front-door",
        "instructions",
    ):
        assert taint not in serialized
    assert set(summary) == {
        "mode",
        "contract_version",
        "contract_sha256",
        "network_requests_authorized",
        "base_cell_count",
        "unit_count",
        "unique_request_hash_count",
        "conservative_execution_ceiling",
        "hard_cap",
        "currency",
        "requests",
    }


@pytest.mark.parametrize(
    "passing,verdict",
    [(3, "continue"), (2, "continue"), (1, "narrow"), (0, "stop_or_redesign_once")],
)
def test_all_45_valid_exhaustive_verdicts(passing, verdict):
    result = _score(outcomes(), annotations(passing))
    assert result["verdict"] == verdict
    assert result["passing_family_count"] == passing
    assert result["external_validity_unit_count"] == 3
    assert len(result["families"]) == 3
    assert "pooled" not in json.dumps(result).casefold()
    assert "45 independent" not in json.dumps(result).casefold()


def test_any_correct_context_critical_finding_forces_safety_stop():
    v1, _ = load_v1_contract()
    correct = next(
        unit.unit_id
        for unit in schedule()
        for scenario in v1["scenarios"]
        for cell in scenario["cells"]
        if unit.base_cell_id == cell["cell_id"]
        and cell["condition"] == "correct_context"
    )
    result = _score(outcomes(), annotations(3, critical={correct}))
    assert result["verdict"] == "stop_or_redesign_once"
    assert result["safety_stop"] is True


def test_42_valid_with_at_least_four_per_cell_is_valid():
    missing = {schedule()[0].unit_id, schedule()[5].unit_id, schedule()[10].unit_id}
    result = _score(outcomes(missing=missing), annotations(3, missing=missing))
    assert result["execution_valid"] is True


def test_41_valid_is_invalid_and_reveals_no_scores_or_verdict_details():
    missing = {unit.unit_id for unit in schedule()[:4]}
    result = _score(outcomes(missing=missing), annotations(3, missing=missing))
    assert result == {"execution_valid": False, "verdict": "invalid_execution"}


def test_three_of_five_in_one_cell_is_invalid():
    missing = [u.unit_id for u in schedule() if u.base_cell_id == EXECUTION_ORDER[0]][
        :2
    ]
    result = _score(outcomes(missing=missing), annotations(3, missing=missing))
    assert result == {"execution_valid": False, "verdict": "invalid_execution"}


def test_missing_correct_draw_prevents_that_family_pass():
    v1, _ = load_v1_contract()
    correct_cell = next(
        c["cell_id"]
        for c in v1["scenarios"][0]["cells"]
        if c["condition"] == "correct_context"
    )
    missing = {next(u.unit_id for u in schedule() if u.base_cell_id == correct_cell)}
    result = _score(outcomes(missing=missing), annotations(3, missing=missing))
    family = next(
        x for x in result["families"] if x["family"] == v1["scenarios"][0]["family"]
    )
    assert family["passes"] is False
    assert result["passing_family_count"] == 2


def test_worst_case_missing_comparator_can_block_despite_favorable_observed_delta():
    analysis = analyze_family(
        [Fraction(1)] * 5,
        [Fraction(0)] * 4,
        [Fraction(0)] * 5,
        threshold=Fraction(1, 5),
    )
    assert analysis["observed_correct_minus_withheld"] == Fraction(1)
    assert analysis["worst_case_correct_minus_withheld"] == Fraction(4, 5)
    assert analysis["material_pass"] is True
    blocked = analyze_family(
        [Fraction(4, 5)] * 5,
        [Fraction(11, 20)] * 4,
        [Fraction(0)] * 5,
        threshold=Fraction(1, 5),
    )
    assert blocked["observed_correct_minus_withheld"] == Fraction(1, 4)
    assert blocked["worst_case_correct_minus_withheld"] == Fraction(4, 25)
    assert blocked["material_pass"] is False


def test_material_threshold_exact_boundary_and_below():
    boundary = analyze_family(
        [Fraction(1)] * 5,
        [Fraction(4, 5)] * 5,
        [Fraction(4, 5)] * 5,
        threshold=Fraction(1, 5),
    )
    below = analyze_family(
        [Fraction(1)] * 5,
        [Fraction(4, 5) + Fraction(1, 100)] * 5,
        [Fraction(4, 5)] * 5,
        threshold=Fraction(1, 5),
    )
    assert boundary["material_pass"] is True
    assert below["material_pass"] is False


def test_exact_permutation_small_fixture_and_ties_are_extreme():
    assert exact_one_sided_permutation_pvalue(
        [Fraction(1)] * 2, [Fraction(0)] * 2
    ) == Fraction(1, 6)
    assert exact_one_sided_permutation_pvalue(
        [Fraction(1)] * 2, [Fraction(1)] * 2
    ) == Fraction(1)


def test_holm_small_fixture_ties_and_exact_outputs():
    result = holm_correction([Fraction(1, 100)] * 3, alpha=Fraction(1, 20))
    assert [item["adjusted_p"] for item in result] == [Fraction(3, 100)] * 3
    assert [item["rejected"] for item in result] == [True, True, True]
    stopped = holm_correction(
        [Fraction(1, 100), Fraction(1, 25), Fraction(1, 25)], alpha=Fraction(1, 20)
    )
    assert [item["rejected"] for item in stopped] == [True, False, False]


def test_deterministic_exhaustive_bootstrap_small_hand_fixture():
    first = bootstrap_percentile_interval(
        [Fraction(0), Fraction(1)], [Fraction(0), Fraction(0)]
    )
    second = bootstrap_percentile_interval(
        [Fraction(0), Fraction(1)], [Fraction(0), Fraction(0)]
    )
    assert first == second == (Fraction(0), Fraction(1))


def test_family_p_is_maximum_and_pooled_favorable_pattern_cannot_override_failure():
    result = _score(outcomes(), annotations(1))
    assert result["passing_family_count"] == 1
    assert result["verdict"] == "narrow"
    for family in result["families"]:
        assert Fraction(family["p_value"]) == max(
            Fraction(family["correct_vs_withheld_p"]),
            Fraction(family["correct_vs_stale_p"]),
        )


def test_incomplete_posthoc_or_incoherent_annotations_fail_closed():
    records = annotations(3)
    assert _score(outcomes(), records[:-1]) == {
        "execution_valid": False,
        "verdict": "invalid_execution",
    }
    records = annotations(3, locked=False)
    result = _score(outcomes(), records)
    assert result == {"execution_valid": False, "verdict": "invalid_execution"}
    records = annotations(3)
    records[0]["evidence_coherent"] = False
    result = _score(outcomes(), records)
    assert result == {"execution_valid": False, "verdict": "invalid_execution"}
    records = annotations(3)
    records[0]["posthoc_note"] = "changed after reveal"
    with pytest.raises(ContractValidationError, match="field set"):
        _score(outcomes(), records)


def test_scoring_requires_verified_alias_based_annotation_lock():
    assert score_replication(
        contract(),
        outcomes(),
        annotations(3),
        annotation_lock_verified=False,
    ) == {"execution_valid": False, "verdict": "invalid_execution"}


def test_maximum_cost_and_hard_cap_are_exact():
    projection = project_maximum_cost(contract())
    assert projection == {
        "conservative_execution_ceiling": "1.470820",
        "hard_cap": "2.00",
        "currency": "USD",
    }


def test_module_is_offline_only_and_python39_compatible():
    source = MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source, feature_version=(3, 9))
    imported = set()
    functions = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.add(node.name)
    assert not ({"openai", "httpx", "requests", "urllib", "socket"} & imported)
    assert not any(
        token in name for name in functions for token in ("dispatch", "execute", "live")
    )
    assert "api.openai.com" not in source


def test_cli_has_no_live_mode(capsys):
    assert main(["--contract", str(CANONICAL_CONTRACT)]) == 0
    assert json.loads(capsys.readouterr().out) == build_dry_run_summary(
        CANONICAL_CONTRACT
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.adaptive_selection.context_sensitivity_replication_v2",
            "--live",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert result.stdout == ""
