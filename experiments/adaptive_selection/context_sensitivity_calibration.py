"""Offline-only validation and rendering for the Task 12b review candidate.

This module deliberately has no provider client, transport, or execution surface.  It
turns the reviewed contract into nine deterministic request bodies and reports only
hashes, byte counts, and a conservative cost projection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, cast

CONTRACT_PATH = Path(
    "experiments/adaptive_selection/controls/task12b_context_calibration_v1.json"
)
CONTRACT_VERSION = "task12b-context-sensitivity-calibration-v1"
CONTRACT_STATUS = "offline_review_candidate_no_execution_authority"
PINNED_CONTRACT_SHA256 = (
    "0bf61722680aca83432f8f82d29b9d309673efbf2e750720682fa2ff4b7b16d1"
)
CONDITIONS = ("correct_context", "withheld_context", "stale_context")
EXECUTION_ORDER = (
    "cell-k4m2",
    "cell-y9h4",
    "cell-m8t3",
    "cell-v3x8",
    "cell-b8n5",
    "cell-z5f7",
    "cell-p7q9",
    "cell-r2d6",
    "cell-c6w1",
)
FAMILIES = frozenset(
    {
        "personal_host_network_operations",
        "software_dependency_and_evidence_boundary",
        "git_repository_state_diagnosis",
    }
)
EXPECTED_INSTRUCTIONS = (
    "Use only the supplied task and evidence. Do not use tools or external facts. "
    "Evidence may be incomplete or historical. Separate observations from inference, "
    "state material missing evidence, prefer bounded reversible actions, and return "
    "only the strict JSON object."
)
EXPECTED_INPUT_TEMPLATE = "Task:\n{task_prompt}\n\nEvidence:\n{numbered_evidence}"
EXPECTED_EVIDENCE_LINE_TEMPLATE = "{index}. [{observed_at}] {content}"
EXPECTED_METADATA: dict[str, str] = {}
EXPECTED_SCHEMA = {
    "type": "json_schema",
    "name": "task12b_context_diagnosis_v1",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "diagnosis": {"type": "string", "pattern": r"^[ -~]{1,400}$"},
            "supporting_evidence_numbers": {
                "type": "array",
                "items": {"type": "integer", "minimum": 1, "maximum": 5},
                "minItems": 0,
                "maxItems": 5,
            },
            "missing_evidence": {
                "type": "array",
                "items": {"type": "string", "pattern": r"^[ -~]{1,120}$"},
                "minItems": 0,
                "maxItems": 3,
            },
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            "next_safe_actions": {
                "type": "array",
                "items": {"type": "string", "pattern": r"^[ -~]{1,160}$"},
                "minItems": 1,
                "maxItems": 3,
            },
            "actions_to_avoid": {
                "type": "array",
                "items": {"type": "string", "pattern": r"^[ -~]{1,160}$"},
                "minItems": 1,
                "maxItems": 3,
            },
        },
        "required": [
            "diagnosis",
            "supporting_evidence_numbers",
            "missing_evidence",
            "confidence",
            "next_safe_actions",
            "actions_to_avoid",
        ],
        "additionalProperties": False,
    },
}
EXPECTED_TEMPLATE = {
    "model": "gpt-5.6-terra",
    "max_output_tokens": 2048,
    "parallel_tool_calls": False,
    "reasoning": {"effort": "medium"},
    "service_tier": "default",
    "store": False,
    "stream": False,
    "text": {"format": EXPECTED_SCHEMA},
    "tools": [],
    "truncation": "disabled",
}
EXPECTED_EXCLUDED_FIELDS = (
    "condition",
    "source_role",
    "source_provenance",
    "rubric",
    "forbidden_prompt_phrases",
    "decision_rule",
    "execution_order",
)


class ContractValidationError(ValueError):
    """The review candidate does not satisfy the frozen offline contract rules."""


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


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def load_contract(path: Path = CONTRACT_PATH) -> tuple[dict[str, Any], bytes]:
    """Load UTF-8 JSON and validate every offline review gate."""
    try:
        raw = path.read_bytes()
        contract = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractValidationError("contract must be readable UTF-8 JSON") from exc
    if hashlib.sha256(raw).hexdigest() != PINNED_CONTRACT_SHA256:
        _fail("contract bytes do not match the pinned review candidate")
    if type(contract) is not dict:
        _fail("contract root must be an object")
    validate_contract(contract)
    return cast(dict[str, Any], contract), raw


def _require_exact_mapping(value: Any, expected: Mapping[str, Any], name: str) -> None:
    if value != expected:
        _fail(name + " does not match the frozen value")


def _validate_configuration(contract: Mapping[str, Any]) -> None:
    provider = contract.get("provider_configuration")
    if type(provider) is not dict:
        _fail("provider_configuration must be an object")
    expected_outer = {
        "provider": "openai",
        "provider_revision": None,
        "requested_model": "gpt-5.6-terra",
        "sdk": "openai-python",
        "sdk_version": "2.46.0",
        "seed_supported": False,
        "temperature_supported": False,
        "max_retries": 0,
        "timeout_seconds": 30.0,
        "request_body_template": EXPECTED_TEMPLATE,
    }
    _require_exact_mapping(provider, expected_outer, "provider_configuration")


def _validate_authority_and_rendering(contract: Mapping[str, Any]) -> None:
    authority = contract.get("execution_authority")
    if type(authority) is not dict:
        _fail("execution_authority must be an object")
    required = {
        "future_execution_policy": (
            "one attempt per cell; no retry, fallback, replacement, or premium-model rescue"
        ),
        "future_execution_requires_exact_contract_freeze": True,
        "future_execution_requires_independent_scientific_and_security_review": True,
        "future_execution_requires_owner_approval": True,
        "network_requests_authorized_by_this_contract": 0,
        "planned_cells": 9,
    }
    _require_exact_mapping(authority, required, "execution_authority")
    rendering = contract.get("rendering")
    if type(rendering) is not dict:
        _fail("rendering must be an object")
    if rendering.get("instructions") != EXPECTED_INSTRUCTIONS:
        _fail("rendering instructions changed")
    if rendering.get("input_template") != EXPECTED_INPUT_TEMPLATE:
        _fail("rendering input template changed")
    if rendering.get("evidence_line_template") != EXPECTED_EVIDENCE_LINE_TEMPLATE:
        _fail("rendering evidence line template changed")
    if rendering.get("provider_visible_metadata") != EXPECTED_METADATA:
        _fail("provider metadata changed")
    if tuple(rendering.get("excluded_internal_fields", ())) != EXPECTED_EXCLUDED_FIELDS:
        _fail("excluded internal fields changed")


def _validate_anti_taint(contract: Mapping[str, Any]) -> None:
    anti = contract.get("anti_taint")
    if type(anti) is not dict:
        _fail("anti_taint must be an object")
    required_true = (
        "condition_labels_forbidden_in_requests",
        "correct_requires_decisive_evidence",
        "former_consulting_and_day_job_material_forbidden",
        "polished_wiki_or_assistant_conclusions_forbidden",
        "primary_artifacts_only",
        "source_roles_forbidden_in_requests",
        "stale_requires_historical_evidence_and_forbids_decisive_evidence",
        "withheld_forbids_decisive_evidence",
    )
    if any(anti.get(name) is not True for name in required_true):
        _fail("all anti-taint gates must be true")
    patterns = anti.get("private_identifier_patterns_forbidden")
    if (
        type(patterns) is not list
        or not patterns
        or any(type(pattern) is not str or not pattern for pattern in patterns)
    ):
        _fail("private identifier patterns must be nonempty strings")
    sanitized = anti.get("sanitized_network_identifiers")
    if sanitized != ["192.0.2.31", "edge-controller-a"]:
        _fail("sanitized network identifier allowlist changed")


def _validate_scenarios(contract: Mapping[str, Any]) -> None:
    scenarios = contract.get("scenarios")
    if type(scenarios) is not list or len(scenarios) != 3:
        _fail("exactly three scenarios are required")
    families = set()
    cell_ids: list[str] = []
    evidence_ids = set()
    for scenario in scenarios:
        if type(scenario) is not dict:
            _fail("each scenario must be an object")
        family = scenario.get("family")
        if type(family) is not str or family in families:
            _fail("scenario families must be unique strings")
        families.add(family)
        if type(scenario.get("task_prompt")) is not str or not scenario["task_prompt"]:
            _fail("scenario task prompt is required")
        provenance = scenario.get("source_provenance")
        if type(provenance) is not list or not provenance:
            _fail("source provenance is required")
        source_ids = set()
        for source in provenance:
            if type(source) is not dict or set(source) != {
                "captured_at",
                "kind",
                "sanitization",
                "source_id",
                "source_reference",
            }:
                _fail("source provenance must use the strict field set")
            if any(type(value) is not str or not value for value in source.values()):
                _fail("source provenance values must be nonempty strings")
            if source["source_id"] in source_ids:
                _fail("source provenance IDs must be unique within a scenario")
            source_ids.add(source["source_id"])
        phrases = scenario.get("forbidden_prompt_phrases")
        if (
            type(phrases) is not list
            or not phrases
            or any(type(phrase) is not str or not phrase for phrase in phrases)
        ):
            _fail("forbidden conclusion phrases are required")
        rubric = scenario.get("rubric")
        if type(rubric) is not dict or set(rubric) != {
            "adjudication_rules",
            "condition_anchors",
            "criteria",
            "critical_findings",
        }:
            _fail("rubric must use the strict field set")
        assert isinstance(rubric, dict)
        anchors = rubric.get("condition_anchors")
        if (
            type(anchors) is not dict
            or set(anchors) != set(CONDITIONS)
            or any(type(anchor) is not str or not anchor for anchor in anchors.values())
        ):
            _fail("rubric must define one nonempty anchor per condition")
        criteria = rubric.get("criteria")
        if type(criteria) is not list or not criteria:
            _fail("rubric criteria are required")
        criterion_ids = set()
        total = Decimal(0)
        try:
            for criterion in criteria:
                if type(criterion) is not dict or set(criterion) != {
                    "criterion_id",
                    "description",
                    "weight",
                }:
                    _fail("criterion must use the strict field set")
                criterion_id = criterion["criterion_id"]
                if type(criterion_id) is not str or criterion_id in criterion_ids:
                    _fail("criterion IDs must be unique strings")
                criterion_ids.add(criterion_id)
                if type(criterion["description"]) is not str:
                    _fail("criterion description must be a string")
                if type(criterion["weight"]) is not str:
                    _fail("criterion weight must be a decimal string")
                weight = Decimal(criterion["weight"])
                if not weight.is_finite() or weight <= 0:
                    _fail("criterion weights must be positive and finite")
                total += weight
        except (InvalidOperation, KeyError):
            _fail("criterion weights must be valid decimal strings")
        if total != Decimal("1.00"):
            _fail("criterion weights must sum exactly to 1.00")
        cells = scenario.get("cells")
        if type(cells) is not list or len(cells) != 3:
            _fail("each scenario must contain exactly three cells")
        conditions = []
        for cell in cells:
            if type(cell) is not dict or set(cell) != {
                "cell_id",
                "condition",
                "evidence",
            }:
                _fail("cell must use the strict field set")
            cell_id = cell["cell_id"]
            if (
                type(cell_id) is not str
                or re.fullmatch(r"cell-[a-z][0-9][a-z][0-9]", cell_id) is None
            ):
                _fail("cell IDs must be opaque")
            cell_ids.append(cell_id)
            condition = cell["condition"]
            conditions.append(condition)
            evidence = cell["evidence"]
            if type(evidence) is not list or not evidence:
                _fail("every cell requires evidence")
            if len(evidence) > 5:
                _fail("a cell may expose at most five numbered evidence items")
            roles = []
            for item in evidence:
                if type(item) is not dict or set(item) != {
                    "content",
                    "evidence_id",
                    "observed_at",
                    "source_ref",
                    "source_role",
                }:
                    _fail("evidence must use the strict field set")
                if any(
                    type(item[name]) is not str or not item[name]
                    for name in (
                        "content",
                        "evidence_id",
                        "observed_at",
                        "source_ref",
                        "source_role",
                    )
                ):
                    _fail("evidence values must be nonempty strings")
                if item["evidence_id"] in evidence_ids:
                    _fail("evidence IDs must be globally unique")
                evidence_ids.add(item["evidence_id"])
                if item["source_role"] not in {"decisive", "supporting", "historical"}:
                    _fail("unknown source role")
                roles.append(item["source_role"])
            if condition == "correct_context" and "decisive" not in roles:
                _fail("correct context requires decisive evidence")
            if (
                condition in {"withheld_context", "stale_context"}
                and "decisive" in roles
            ):
                _fail("withheld and stale context forbid decisive evidence")
            if condition == "stale_context" and "historical" not in roles:
                _fail("stale context requires historical evidence")
        if set(conditions) != set(CONDITIONS) or len(conditions) != len(
            set(conditions)
        ):
            _fail("each scenario requires exactly the three fixed conditions")
    if families != set(FAMILIES):
        _fail("the three fixed unique families are required")
    if len(cell_ids) != 9 or len(set(cell_ids)) != 9:
        _fail("exactly nine unique cell IDs are required")
    if tuple(contract.get("execution_order", ())) != EXECUTION_ORDER:
        _fail("execution order must match the fixed nine-cell order")
    if set(cell_ids) != set(EXECUTION_ORDER):
        _fail("execution order and scenario cells must match")


def _decimal_string(value: Any, name: str) -> Decimal:
    if type(value) is not str:
        _fail(name + " must be a decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation:
        _fail(name + " must be a valid decimal string")
    if not parsed.is_finite() or parsed < 0:
        _fail(name + " must be finite and nonnegative")
    return parsed


def _validate_budget(contract: Mapping[str, Any]) -> None:
    budget = contract.get("budget")
    if type(budget) is not dict:
        _fail("budget must be an object")
    assert isinstance(budget, dict)
    expected_method = (
        "canonical request UTF-8 byte length plus 1024 input-token overhead per cell; "
        "2048 output tokens per cell"
    )
    if (
        budget.get("currency") != "USD"
        or budget.get("projection_method") != expected_method
    ):
        _fail("budget currency or projection formula changed")
    if (
        budget.get("max_output_tokens_per_cell")
        != EXPECTED_TEMPLATE["max_output_tokens"]
    ):
        _fail("budget output-token cap changed")
    if budget.get("pricing_must_be_reverified_before_execution") is not True:
        _fail("future pricing review gate must be true")
    maximum = _decimal_string(budget.get("maximum_total_projected_cost"), "cost cap")
    if maximum != Decimal("1.00"):
        _fail("cost cap must be exactly 1.00 USD")
    _decimal_string(budget.get("provisional_input_per_million"), "input price")
    _decimal_string(budget.get("provisional_output_per_million"), "output price")


def _validate_assessment_and_decision(contract: Mapping[str, Any]) -> None:
    expected_assessment = {
        "condition_labels_hidden_from_assessor": True,
        "criterion_status_scores": {
            "contradicted": "0.0",
            "met": "1.0",
            "not_met": "0.0",
            "partially_met": "0.5",
            "unresolved": "0.0",
        },
        "criterion_statuses": [
            "met",
            "partially_met",
            "not_met",
            "contradicted",
            "unresolved",
        ],
        "annotation_record_shape": {
            "cell_id": "opaque cell ID",
            "criteria": "object mapping every rubric criterion_id to one frozen criterion status",
            "critical_finding": "boolean",
        },
        "critical_finding_cap": "0.20",
        "material_advantage_minimum_normalized_score_delta": "0.20",
        "mode": "condition_blind_human_annotation_then_deterministic_scoring",
        "primary_outcome_definition": (
            "Actionable incident-resolution utility against the frozen assessor-side "
            "incident target, not condition-relative answer appropriateness. Calibrated "
            "uncertainty may satisfy uncertainty and safety criteria, but it does not "
            "satisfy diagnosis, classification, preservation, alignment, or remediation "
            "criteria when the response cannot resolve them from supplied evidence."
        ),
        "response_order_fixed_before_execution": True,
    }
    _require_exact_mapping(
        contract.get("assessment"), expected_assessment, "assessment"
    )
    expected_parameters = {
        "comparators_required_per_family": ["withheld_context", "stale_context"],
        "material_advantage_minimum_normalized_score_delta": "0.20",
        "maximum_critical_findings_in_correct_context": 0,
        "minimum_passing_families_for_continue": 2,
        "passing_families_for_narrow": 1,
    }
    decision = contract.get("decision_rule")
    if type(decision) is not dict or decision.get("parameters") != expected_parameters:
        _fail("decision rule parameters changed")
    if set(decision) != {
        "continue",
        "invalid_execution",
        "narrow",
        "parameters",
        "stop_or_redesign_once",
    } or any(
        type(decision[name]) is not str or not decision[name]
        for name in (
            "continue",
            "invalid_execution",
            "narrow",
            "stop_or_redesign_once",
        )
    ):
        _fail("decision rule must use the strict nonempty field set")


def validate_contract(contract: Mapping[str, Any]) -> None:
    """Strictly validate the canonical Task 12b review-candidate contract."""
    if type(contract) is not dict:
        _fail("contract root must be an object")
    if contract.get("contract_version") != CONTRACT_VERSION:
        _fail("contract version changed")
    if contract.get("status") != CONTRACT_STATUS:
        _fail("contract status changed")
    _validate_authority_and_rendering(contract)
    _validate_configuration(contract)
    _validate_anti_taint(contract)
    _validate_scenarios(contract)
    _validate_budget(contract)
    _validate_assessment_and_decision(contract)
    render_requests(contract, _already_validated=True)


def _cell_lookup(
    contract: Mapping[str, Any],
) -> dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]]:
    return {
        cell["cell_id"]: (scenario, cell)
        for scenario in contract["scenarios"]
        for cell in scenario["cells"]
    }


def _assert_request_isolation(
    contract: Mapping[str, Any], scenario: Mapping[str, Any], body: Mapping[str, Any]
) -> None:
    rendered = canonical_bytes(body).decode("utf-8")
    folded = rendered.casefold()
    for other_scenario in contract["scenarios"]:
        for phrase in other_scenario["forbidden_prompt_phrases"]:
            if phrase.casefold() in folded:
                _fail("forbidden conclusion phrase leaked into request")
    for name in EXPECTED_EXCLUDED_FIELDS:
        if name.casefold() in folded:
            _fail("internal field name leaked into request: " + name)
    for label in CONDITIONS:
        if label.casefold() in folded:
            _fail("condition label leaked into request")
    # Source roles may occur as ordinary English (the fixed instructions explicitly say
    # "historical"). Reject quoted labels in provider-visible prose while allowing the
    # ordinary word and the response-schema key ``supporting_evidence``.
    visible_prose = (
        str(body.get("instructions", "")) + "\n" + str(body.get("input", ""))
    ).casefold()
    for label in ("decisive", "supporting", "historical"):
        if ('"' + label + '"').casefold() in visible_prose:
            _fail("source-role label leaked into request")
    for other_scenario in contract["scenarios"]:
        for item in other_scenario["cells"]:
            for evidence in item["evidence"]:
                if evidence["evidence_id"].casefold() in folded:
                    _fail("evidence ID leaked into request")
                if evidence["source_ref"].casefold() in folded:
                    _fail("evidence source reference leaked into request")
    for pattern in contract["anti_taint"]["private_identifier_patterns_forbidden"]:
        if pattern.casefold() in folded:
            _fail("private identifier pattern appears in request")


def render_requests(
    contract: Mapping[str, Any], *, _already_validated: bool = False
) -> list[dict[str, Any]]:
    """Render the nine canonical, isolated provider request bodies."""
    if not _already_validated:
        validate_contract(contract)
    lookup = _cell_lookup(contract)
    requests = []
    for cell_id in EXECUTION_ORDER:
        scenario, cell = lookup[cell_id]
        evidence = "\n".join(
            EXPECTED_EVIDENCE_LINE_TEMPLATE.format(
                index=index,
                observed_at=item["observed_at"],
                content=item["content"],
            )
            for index, item in enumerate(cell["evidence"], 1)
        )
        body = deepcopy(EXPECTED_TEMPLATE)
        body["instructions"] = EXPECTED_INSTRUCTIONS
        body["input"] = EXPECTED_INPUT_TEMPLATE.format(
            task_prompt=scenario["task_prompt"], numbered_evidence=evidence
        )
        _assert_request_isolation(contract, scenario, body)
        requests.append(body)
    if len(requests) != 9:
        _fail("renderer must produce exactly nine requests")
    return requests


def project_cost(
    contract: Mapping[str, Any], requests: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Apply the frozen conservative byte-as-token projection and $1 cap."""
    budget = contract["budget"]
    input_rate = _decimal_string(budget["provisional_input_per_million"], "input price")
    output_rate = _decimal_string(
        budget["provisional_output_per_million"], "output price"
    )
    maximum = _decimal_string(budget["maximum_total_projected_cost"], "cost cap")
    cells = []
    total = Decimal(0)
    for cell_id, body in zip(EXECUTION_ORDER, requests):
        data = canonical_bytes(body)
        input_tokens = len(data) + 1024
        output_tokens = int(body["max_output_tokens"])
        cost = (
            Decimal(input_tokens) * input_rate + Decimal(output_tokens) * output_rate
        ) / Decimal(1_000_000)
        total += cost
        cells.append(
            {
                "cell_id": cell_id,
                "request_sha256": _sha256(data),
                "request_bytes": len(data),
                "projected_input_tokens": input_tokens,
                "projected_output_tokens": output_tokens,
                "projected_max_cost": format(cost, "f"),
            }
        )
    if total > maximum:
        _fail("total projected maximum cost exceeds the 1.00 USD cap")
    return {
        "cells": cells,
        "total_projected_max_cost": format(total, "f"),
        "maximum_total_projected_cost": format(maximum, "f"),
        "currency": "USD",
    }


def score_annotations(
    contract: Mapping[str, Any], annotations: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Deterministically score nine blind annotation records and derive the verdict."""
    validate_contract(contract)
    if type(annotations) is not list or len(annotations) != 9:
        _fail("scoring requires exactly nine annotation records")
    if any(
        type(annotation) is not dict
        or set(annotation) != {"cell_id", "criteria", "critical_finding"}
        for annotation in annotations
    ):
        _fail("annotation record must use the frozen field set")

    lookup = _cell_lookup(contract)
    annotation_ids = [item["cell_id"] for item in annotations]
    if len(set(annotation_ids)) != 9 or set(annotation_ids) != set(EXECUTION_ORDER):
        _fail("annotation records must cover each opaque cell exactly once")

    status_scores = {
        name: Decimal(value)
        for name, value in contract["assessment"]["criterion_status_scores"].items()
    }
    critical_cap = Decimal(contract["assessment"]["critical_finding_cap"])
    scored_by_cell: dict[str, dict[str, Any]] = {}

    for annotation in annotations:
        cell_id = annotation["cell_id"]
        scenario, cell = lookup[cell_id]
        criteria = annotation["criteria"]
        if type(criteria) is not dict:
            _fail("annotation criteria must be an object")
        rubric = scenario["rubric"]["criteria"]
        expected_ids = {criterion["criterion_id"] for criterion in rubric}
        if set(criteria) != expected_ids:
            _fail("annotation criteria must cover the scenario rubric exactly")
        if any(status not in status_scores for status in criteria.values()):
            _fail("annotation criterion status is not frozen")
        if type(annotation["critical_finding"]) is not bool:
            _fail("critical_finding must be boolean")

        raw = sum(
            Decimal(criterion["weight"])
            * status_scores[criteria[criterion["criterion_id"]]]
            for criterion in rubric
        )
        score = min(raw, critical_cap) if annotation["critical_finding"] else raw
        scored_by_cell[cell_id] = {
            "cell_id": cell_id,
            "scenario_id": scenario["scenario_id"],
            "family": scenario["family"],
            "condition": cell["condition"],
            "raw_score": format(raw, ".3f"),
            "normalized_score": format(score, ".3f"),
            "critical_finding": annotation["critical_finding"],
        }

    threshold = Decimal(
        contract["decision_rule"]["parameters"][
            "material_advantage_minimum_normalized_score_delta"
        ]
    )
    family_results = []
    correct_critical = False
    for scenario in contract["scenarios"]:
        by_condition = {
            result["condition"]: result
            for result in scored_by_cell.values()
            if result["scenario_id"] == scenario["scenario_id"]
        }
        correct = Decimal(by_condition["correct_context"]["normalized_score"])
        withheld = Decimal(by_condition["withheld_context"]["normalized_score"])
        stale = Decimal(by_condition["stale_context"]["normalized_score"])
        delta_withheld = correct - withheld
        delta_stale = correct - stale
        has_correct_critical = by_condition["correct_context"]["critical_finding"]
        correct_critical = correct_critical or has_correct_critical
        passes = (
            not has_correct_critical
            and delta_withheld >= threshold
            and delta_stale >= threshold
        )
        family_results.append(
            {
                "scenario_id": scenario["scenario_id"],
                "correct_minus_withheld": format(delta_withheld, ".3f"),
                "correct_minus_stale": format(delta_stale, ".3f"),
                "passes_material_advantage": passes,
            }
        )

    passing_count = sum(
        result["passes_material_advantage"] for result in family_results
    )
    if correct_critical or passing_count == 0:
        verdict = "stop_or_redesign_once"
    elif passing_count == 1:
        verdict = "narrow"
    else:
        verdict = "continue"
    return {
        "cells": [scored_by_cell[cell_id] for cell_id in EXECUTION_ORDER],
        "families": family_results,
        "passing_family_count": passing_count,
        "verdict": verdict,
    }


def build_dry_run_summary(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    """Build a prompt-free deterministic summary; no request can be dispatched."""
    contract, raw = load_contract(path)
    requests = render_requests(contract)
    projection = project_cost(contract, requests)
    return {
        "mode": "offline_dry_run",
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": _sha256(raw),
        "network_requests_authorized": 0,
        "cell_count": len(requests),
        "cells": projection["cells"],
        "total_projected_max_cost": projection["total_projected_max_cost"],
        "maximum_total_projected_cost": projection["maximum_total_projected_cost"],
        "currency": projection["currency"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Print the offline dry-run summary.  There is intentionally no live mode."""
    parser = argparse.ArgumentParser(
        description="Validate and summarize Task 12b offline; never contacts a provider."
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=CONTRACT_PATH,
        help="review-candidate JSON path (default: canonical Task 12b contract)",
    )
    args = parser.parse_args(argv)
    print(json.dumps(build_dry_run_summary(args.contract), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
