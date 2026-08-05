"""Offline scientific layer for Task 12b v2 repeated-draw replication.

The module schedules and scores frozen requests.  It has no provider client,
transport, authority state, credential access, or network execution surface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from itertools import combinations, product
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

from experiments.adaptive_selection import context_sensitivity_calibration as v1

CONTRACT_PATH = Path(
    "experiments/adaptive_selection/controls/task12b_context_replication_v2.json"
)
CONTRACT_VERSION = "task12b-context-sensitivity-repeated-draws-v2"
PINNED_CONTRACT_SHA256 = (
    "e967ce9872afec25da2b9803ed494545ae36725bbef128772e7fecf98f55de06"
)
PINNED_CANONICAL_SHA256 = (
    "9a5d11b67ff0ebe63ef43f3d39c23c9dc3bee318afbfb404d8d8f18ec2eb1ad4"
)
V1_CONTRACT_SHA256 = v1.PINNED_CONTRACT_SHA256
V1_RENDERER_SHA256 = "6616ec0f8f8621490d0e2f83d5472d049881158cce59f685098fd593f62889ea"
_ROOT = Path(__file__).resolve().parents[2]
_RESOLVED_ANNOTATION_FIELDS = {
    "unit_id",
    "criteria",
    "critical_finding",
    "locked",
    "evidence_coherent",
}


class ContractValidationError(ValueError):
    """The v2 contract or supplied offline scientific record is invalid."""


def _fail(message: str) -> None:
    raise ContractValidationError(message)


def canonical_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON bytes."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _exact_json_object(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _resolved(path: Path) -> Path:
    return path if path.is_absolute() else _ROOT / path


def load_contract(path: Path = CONTRACT_PATH) -> Tuple[Dict[str, Any], bytes]:
    """Load and exact-validate the raw hash-frozen v2 contract."""
    try:
        raw = _resolved(path).read_bytes()
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_exact_json_object)
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise ContractValidationError("contract must be readable UTF-8 JSON") from exc
    if _sha256(raw) != PINNED_CONTRACT_SHA256:
        _fail("contract bytes do not match the pinned v2 contract")
    validate_contract(value)
    return cast(Dict[str, Any], value), raw


def _verify_v1_lineage(contract: Mapping[str, Any]) -> Tuple[Dict[str, Any], bytes]:
    lineage = contract["lineage"]
    v1_path = _resolved(Path(lineage["predecessor_contract_path"]))
    try:
        predecessor_raw = v1_path.read_bytes()
        renderer_raw = Path(v1.__file__).read_bytes()
    except OSError as exc:
        raise ContractValidationError(
            "frozen predecessor files must be readable"
        ) from exc
    if _sha256(predecessor_raw) != V1_CONTRACT_SHA256:
        _fail("predecessor raw contract hash changed")
    if _sha256(renderer_raw) != V1_RENDERER_SHA256:
        _fail("predecessor renderer hash changed")
    predecessor, _ = v1.load_contract(v1_path)
    return predecessor, predecessor_raw


def validate_contract(contract: Mapping[str, Any]) -> None:
    """Exact-validate all v2 values and their immutable v1 lineage."""
    if type(contract) is not dict:
        _fail("contract root must be an object")
    try:
        digest = _sha256(canonical_bytes(contract))
    except (TypeError, ValueError) as exc:
        raise ContractValidationError("contract must be canonical JSON") from exc
    if digest != PINNED_CANONICAL_SHA256:
        _fail("contract values do not match the pinned v2 contract")
    if contract.get("contract_version") != CONTRACT_VERSION:
        _fail("contract version changed")
    lineage = contract["lineage"]
    if (
        lineage["predecessor_raw_contract_sha256"] != V1_CONTRACT_SHA256
        or lineage["predecessor_renderer_sha256"] != V1_RENDERER_SHA256
        or lineage["replacement_is_not_a_retry"] is not True
        or lineage["predecessor_execution_disposition"]
        != "consumed_invalid_harness_execution_outputs_excluded_and_sealed_unscored"
    ):
        _fail("predecessor lineage changed")
    authority = contract["execution_authority"]
    if (
        authority["network_requests_authorized_by_this_contract"] != 0
        or authority["offline_scientific_layer_only"] is not True
        or authority["provider_execution_surface_present"] is not False
    ):
        _fail("v2 grants no network authority")
    predecessor, _ = _verify_v1_lineage(contract)
    base_requests = dict(zip(v1.EXECUTION_ORDER, v1.render_requests(predecessor)))
    expected_cells = []
    lookup = {
        cell["cell_id"]: (scenario["family"], cell["condition"])
        for scenario in predecessor["scenarios"]
        for cell in scenario["cells"]
    }
    for cell_id in v1.EXECUTION_ORDER:
        family, condition = lookup[cell_id]
        expected_cells.append(
            {
                "base_cell_id": cell_id,
                "condition": condition,
                "family": family,
                "request_sha256": _sha256(canonical_bytes(base_requests[cell_id])),
            }
        )
    if contract["base_cells"] != expected_cells:
        _fail("base cells or predecessor request hashes changed")


@dataclass(frozen=True)
class ScheduledUnit:
    """One immutable, client-side repeated draw of a frozen v1 base request."""

    unit_id: str
    base_cell_id: str
    draw_index: int
    base_request_sha256: str
    provider_request_sha256: str

    @property
    def request_sha256(self) -> str:
        """Compatibility alias for the identical base/provider request hash."""
        return self.provider_request_sha256


def build_schedule(contract: Mapping[str, Any]) -> List[ScheduledUnit]:
    """Build the fixed 45-unit cell-major schedule."""
    validate_contract(contract)
    units = []
    for cell in contract["base_cells"]:
        for draw_index in contract["schedule"]["draw_indices"]:
            units.append(
                ScheduledUnit(
                    unit_id="{}-draw-{}".format(cell["base_cell_id"], draw_index),
                    base_cell_id=cell["base_cell_id"],
                    draw_index=draw_index,
                    base_request_sha256=cell["request_sha256"],
                    provider_request_sha256=cell["request_sha256"],
                )
            )
    if len(units) != 45 or len({unit.unit_id for unit in units}) != 45:
        _fail("scheduler must produce 45 unique units")
    return units


def render_unit_requests(
    contract: Mapping[str, Any], units: Sequence[ScheduledUnit] = ()
) -> List[Dict[str, Any]]:
    """Reuse v1 rendering and prove every provider byte is a frozen base request."""
    validate_contract(contract)
    requested_units = list(units) if units else build_schedule(contract)
    predecessor, _ = _verify_v1_lineage(contract)
    base: Dict[str, Dict[str, Any]] = dict(
        zip(v1.EXECUTION_ORDER, v1.render_requests(predecessor))
    )
    expected_hashes = {
        cell["base_cell_id"]: cell["request_sha256"] for cell in contract["base_cells"]
    }
    bodies = []
    for unit in requested_units:
        if unit.base_cell_id not in base:
            _fail("scheduled unit references an unknown base cell")
        body = deepcopy(base[unit.base_cell_id])
        data = canonical_bytes(body)
        if _sha256(data) != expected_hashes[unit.base_cell_id]:
            _fail("unit request differs from frozen v1 canonical bytes")
        bodies.append(body)
    return bodies


def project_maximum_cost(contract: Mapping[str, Any]) -> Dict[str, str]:
    """Return the exact preregistered ceiling and hard cap."""
    validate_contract(contract)
    budget = contract["budget"]
    ceiling = Decimal(budget["conservative_execution_ceiling"])
    cap = Decimal(budget["hard_cap"])
    if ceiling != Decimal("1.470820") or cap != Decimal("2.00") or ceiling > cap:
        _fail("budget ceiling or hard cap changed")
    return {
        "conservative_execution_ceiling": format(ceiling, ".6f"),
        "hard_cap": format(cap, ".2f"),
        "currency": budget["currency"],
    }


def _mean(values: Sequence[Fraction]) -> Fraction:
    if not values:
        _fail("a statistical group cannot be empty")
    return sum(values, Fraction(0)) / len(values)


def exact_one_sided_permutation_pvalue(
    correct: Sequence[Fraction], comparator: Sequence[Fraction]
) -> Fraction:
    """Exact label-permutation p-value for H1: mean(correct)>mean(comparator).

    Equality is counted as extreme, making ties conservative and deterministic.
    """
    left = [Fraction(value) for value in correct]
    right = [Fraction(value) for value in comparator]
    if not left or not right:
        _fail("permutation groups cannot be empty")
    pooled = left + right
    observed = _mean(left) - _mean(right)
    extreme = 0
    total = 0
    indices = range(len(pooled))
    for selected_tuple in combinations(indices, len(left)):
        selected = set(selected_tuple)
        perm_left = [pooled[index] for index in indices if index in selected]
        perm_right = [pooled[index] for index in indices if index not in selected]
        if _mean(perm_left) - _mean(perm_right) >= observed:
            extreme += 1
        total += 1
    return Fraction(extreme, total)


def holm_correction(
    p_values: Sequence[Fraction], *, alpha: Fraction = Fraction(1, 20)
) -> List[Dict[str, Any]]:
    """Apply exact Holm step-down correction, preserving original order."""
    values = [Fraction(value) for value in p_values]
    if not values or any(value < 0 or value > 1 for value in values):
        _fail("Holm p-values must be a nonempty sequence in [0,1]")
    ordered = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    count = len(values)
    adjusted_by_index: Dict[int, Fraction] = {}
    rejected_by_index: Dict[int, bool] = {}
    running_adjusted = Fraction(0)
    still_rejecting = True
    for rank, (index, value) in enumerate(ordered):
        running_adjusted = max(
            running_adjusted, min(Fraction(1), (count - rank) * value)
        )
        adjusted_by_index[index] = running_adjusted
        rejected = still_rejecting and value <= alpha / (count - rank)
        rejected_by_index[index] = rejected
        if not rejected:
            still_rejecting = False
    return [
        {
            "p_value": values[index],
            "adjusted_p": adjusted_by_index[index],
            "rejected": rejected_by_index[index],
        }
        for index in range(count)
    ]


def _bootstrap_mean_counts(values: Sequence[Fraction]) -> Counter:
    samples = [Fraction(value) for value in values]
    if not samples:
        _fail("bootstrap groups cannot be empty")
    result: Counter = Counter()
    for selected in product(samples, repeat=len(samples)):
        result[_mean(selected)] += 1
    return result


def _weighted_quantile(counts: Counter, numerator: int, denominator: int) -> Fraction:
    total = sum(counts.values())
    rank = max(1, ceil(Fraction(numerator * total, denominator)))
    seen = 0
    for value in sorted(counts):
        seen += counts[value]
        if seen >= rank:
            return Fraction(value)
    raise AssertionError("weighted quantile rank must be reachable")


def bootstrap_percentile_interval(
    correct: Sequence[Fraction], comparator: Sequence[Fraction]
) -> Tuple[Fraction, Fraction]:
    """Deterministic exhaustive independent-group 95% percentile interval."""
    correct_counts = _bootstrap_mean_counts(correct)
    comparator_counts = _bootstrap_mean_counts(comparator)
    deltas: Counter = Counter()
    for correct_mean, correct_count in correct_counts.items():
        for comparator_mean, comparator_count in comparator_counts.items():
            deltas[correct_mean - comparator_mean] += correct_count * comparator_count
    return (
        _weighted_quantile(deltas, 25, 1000),
        _weighted_quantile(deltas, 975, 1000),
    )


def analyze_family(
    correct: Sequence[Fraction],
    withheld: Sequence[Fraction],
    stale: Sequence[Fraction],
    *,
    threshold: Fraction = Fraction(1, 5),
) -> Dict[str, Any]:
    """Compute exact observed effects, five-draw bounds, p-values, and intervals."""
    groups = [[Fraction(x) for x in values] for values in (correct, withheld, stale)]
    if any(len(values) < 4 or len(values) > 5 for values in groups):
        _fail("family groups must contain four or five valid draws")
    correct_values, withheld_values, stale_values = groups
    observed_withheld = _mean(correct_values) - _mean(withheld_values)
    observed_stale = _mean(correct_values) - _mean(stale_values)

    def worst_case(comparator: Sequence[Fraction]) -> Fraction:
        correct_lower = sum(correct_values, Fraction(0)) / 5
        comparator_upper = (
            sum(comparator, Fraction(0)) + (5 - len(comparator)) * Fraction(1)
        ) / 5
        return correct_lower - comparator_upper

    worst_withheld = worst_case(withheld_values)
    worst_stale = worst_case(stale_values)
    p_withheld = exact_one_sided_permutation_pvalue(correct_values, withheld_values)
    p_stale = exact_one_sided_permutation_pvalue(correct_values, stale_values)
    return {
        "observed_correct_minus_withheld": observed_withheld,
        "observed_correct_minus_stale": observed_stale,
        "worst_case_correct_minus_withheld": worst_withheld,
        "worst_case_correct_minus_stale": worst_stale,
        "correct_vs_withheld_p": p_withheld,
        "correct_vs_stale_p": p_stale,
        "family_p": max(p_withheld, p_stale),
        "correct_vs_withheld_bootstrap": bootstrap_percentile_interval(
            correct_values, withheld_values
        ),
        "correct_vs_stale_bootstrap": bootstrap_percentile_interval(
            correct_values, stale_values
        ),
        "all_correct_draws_present": len(correct_values) == 5,
        "material_pass": (
            len(correct_values) == 5
            and observed_withheld >= threshold
            and observed_stale >= threshold
            and worst_withheld >= threshold
            and worst_stale >= threshold
        ),
    }


def _fraction_text(value: Fraction) -> str:
    return (
        str(value.numerator)
        if value.denominator == 1
        else "{}/{}".format(value.numerator, value.denominator)
    )


def _interval_text(interval: Tuple[Fraction, Fraction]) -> List[str]:
    return [_fraction_text(interval[0]), _fraction_text(interval[1])]


def _invalid() -> Dict[str, Any]:
    return {"execution_valid": False, "verdict": "invalid_execution"}


def score_replication(
    contract: Mapping[str, Any],
    outcomes: Sequence[Mapping[str, Any]],
    resolved_annotations: Sequence[Mapping[str, Any]],
    *,
    annotation_lock_verified: bool,
) -> Dict[str, Any]:
    """Score canonical records only after an alias-based lock was verified.

    The execution layer must verify the immutable blind annotation lock and then
    resolve its random assessment aliases to unit IDs. This scientific layer
    deliberately has no access to the private alias mapping or lock files.
    """
    validate_contract(contract)
    if annotation_lock_verified is not True:
        return _invalid()
    units = build_schedule(contract)
    unit_ids = {unit.unit_id for unit in units}
    if (
        type(outcomes) is not list
        or len(outcomes) != 45
        or any(
            type(record) is not dict or set(record) != {"unit_id", "schema_valid"}
            for record in outcomes
        )
    ):
        _fail("outcomes must use the exact 45-record field set")
    if any(type(record["schema_valid"]) is not bool for record in outcomes):
        _fail("schema_valid must be boolean")
    outcome_ids = [record["unit_id"] for record in outcomes]
    if set(outcome_ids) != unit_ids or len(set(outcome_ids)) != 45:
        _fail("outcomes must cover every scheduled unit exactly once")
    valid_ids = {record["unit_id"] for record in outcomes if record["schema_valid"]}
    by_cell = {
        cell_id: [unit for unit in units if unit.base_cell_id == cell_id]
        for cell_id in v1.EXECUTION_ORDER
    }
    if len(valid_ids) < 42 or any(
        sum(unit.unit_id in valid_ids for unit in cell_units) < 4
        for cell_units in by_cell.values()
    ):
        return _invalid()

    if type(resolved_annotations) is not list or any(
        type(record) is not dict or set(record) != _RESOLVED_ANNOTATION_FIELDS
        for record in resolved_annotations
    ):
        _fail("resolved annotation record must use the frozen field set")
    annotation_ids = [record["unit_id"] for record in resolved_annotations]
    if set(annotation_ids) != valid_ids or len(annotation_ids) != len(valid_ids):
        return _invalid()
    if any(
        record["locked"] is not True or record["evidence_coherent"] is not True
        for record in resolved_annotations
    ):
        return _invalid()

    predecessor, _ = _verify_v1_lineage(contract)
    cell_lookup = {
        cell["cell_id"]: (scenario, cell)
        for scenario in predecessor["scenarios"]
        for cell in scenario["cells"]
    }
    unit_lookup = {unit.unit_id: unit for unit in units}
    status_scores = {
        name: Fraction(value)
        for name, value in contract["assessment"]["criterion_status_scores"].items()
    }
    critical_cap = Fraction(contract["assessment"]["critical_finding_cap"])
    scored: Dict[str, Fraction] = {}
    critical: Dict[str, bool] = {}
    for record in resolved_annotations:
        unit = unit_lookup[record["unit_id"]]
        scenario, _ = cell_lookup[unit.base_cell_id]
        criteria = record["criteria"]
        rubric = scenario["rubric"]["criteria"]
        criterion_ids = {criterion["criterion_id"] for criterion in rubric}
        if type(criteria) is not dict or set(criteria) != criterion_ids:
            _fail("annotation criteria must cover the frozen rubric exactly")
        if any(
            type(status) is not str or status not in status_scores
            for status in criteria.values()
        ):
            _fail("annotation criterion status is not frozen")
        if type(record["critical_finding"]) is not bool:
            _fail("critical_finding must be boolean")
        raw = sum(
            (
                Fraction(criterion["weight"])
                * status_scores[criteria[criterion["criterion_id"]]]
                for criterion in rubric
            ),
            Fraction(0),
        )
        scored[unit.unit_id] = (
            min(raw, critical_cap) if record["critical_finding"] else raw
        )
        critical[unit.unit_id] = record["critical_finding"]

    threshold = Fraction(contract["statistics"]["material_advantage_threshold"])
    analyses = []
    safety_stop = False
    for scenario in predecessor["scenarios"]:
        condition_cells = {
            cell["condition"]: cell["cell_id"] for cell in scenario["cells"]
        }
        values = {}
        for condition, cell_id in condition_cells.items():
            cell_units = by_cell[cell_id]
            values[condition] = [
                scored[unit.unit_id] for unit in cell_units if unit.unit_id in valid_ids
            ]
        correct_ids = [
            unit.unit_id
            for unit in by_cell[condition_cells["correct_context"]]
            if unit.unit_id in valid_ids
        ]
        has_critical = any(critical[unit_id] for unit_id in correct_ids)
        safety_stop = safety_stop or has_critical
        analysis = analyze_family(
            values["correct_context"],
            values["withheld_context"],
            values["stale_context"],
            threshold=threshold,
        )
        analysis["family"] = scenario["family"]
        analysis["correct_critical"] = has_critical
        analyses.append(analysis)

    holm = holm_correction(
        [analysis["family_p"] for analysis in analyses],
        alpha=Fraction(contract["statistics"]["alpha"]),
    )
    family_results = []
    for analysis, correction in zip(analyses, holm):
        passes = (
            analysis["material_pass"]
            and correction["rejected"]
            and not analysis["correct_critical"]
        )
        family_results.append(
            {
                "family": analysis["family"],
                "observed_correct_minus_withheld": _fraction_text(
                    analysis["observed_correct_minus_withheld"]
                ),
                "observed_correct_minus_stale": _fraction_text(
                    analysis["observed_correct_minus_stale"]
                ),
                "worst_case_correct_minus_withheld": _fraction_text(
                    analysis["worst_case_correct_minus_withheld"]
                ),
                "worst_case_correct_minus_stale": _fraction_text(
                    analysis["worst_case_correct_minus_stale"]
                ),
                "correct_vs_withheld_p": _fraction_text(
                    analysis["correct_vs_withheld_p"]
                ),
                "correct_vs_stale_p": _fraction_text(analysis["correct_vs_stale_p"]),
                "p_value": _fraction_text(analysis["family_p"]),
                "holm_adjusted_p": _fraction_text(correction["adjusted_p"]),
                "holm_rejected": correction["rejected"],
                "correct_vs_withheld_bootstrap_95": _interval_text(
                    analysis["correct_vs_withheld_bootstrap"]
                ),
                "correct_vs_stale_bootstrap_95": _interval_text(
                    analysis["correct_vs_stale_bootstrap"]
                ),
                "passes": passes,
            }
        )
    passing_count = sum(result["passes"] for result in family_results)
    if safety_stop:
        verdict = "stop_or_redesign_once"
    elif passing_count >= 2:
        verdict = "continue"
    elif passing_count == 1:
        verdict = "narrow"
    else:
        verdict = "stop_or_redesign_once"
    return {
        "execution_valid": True,
        "external_validity_unit_count": 3,
        "families": family_results,
        "passing_family_count": passing_count,
        "safety_stop": safety_stop,
        "verdict": verdict,
    }


def build_dry_run_summary(path: Path = CONTRACT_PATH) -> Dict[str, Any]:
    """Build a prompt-free, evidence-free, rubric-free deterministic summary."""
    contract, raw = load_contract(path)
    units = build_schedule(contract)
    requests = render_unit_requests(contract, units)
    request_counts = Counter(_sha256(canonical_bytes(body)) for body in requests)
    costs = project_maximum_cost(contract)
    return {
        "mode": "offline_dry_run",
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": "sha256:" + _sha256(raw),
        "network_requests_authorized": 0,
        "base_cell_count": 9,
        "unit_count": len(units),
        "unique_request_hash_count": len(request_counts),
        "conservative_execution_ceiling": costs["conservative_execution_ceiling"],
        "hard_cap": costs["hard_cap"],
        "currency": costs["currency"],
        "requests": [
            {"request_sha256": "sha256:" + digest, "scheduled_count": count}
            for digest, count in sorted(request_counts.items())
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Print the offline dry-run summary; there is intentionally no live mode."""
    parser = argparse.ArgumentParser(
        description="Validate Task 12b v2 offline; never contacts a provider."
    )
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    args = parser.parse_args(argv)
    print(json.dumps(build_dry_run_summary(args.contract), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
