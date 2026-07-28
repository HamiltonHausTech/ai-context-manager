"""Typed loading and strict validation for the tiny adaptive-selection corpus.

The fixture validated here is a deterministic harness check.  It establishes bundle
versioning, order, partition, leakage, and hashing mechanics only; it cannot support
adaptive-selection efficacy claims.  Exact normalized duplicate checks are mechanical:
near-duplicate and semantic/template-sibling review remains a manual domain task.
"""

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Dict, Mapping, Sequence, Tuple, Union

from .schema import FeedbackEvent, SCHEMA_VERSION, TaskCase

DATASET_BUNDLE_VERSION = "1"
PathLike = Union[str, Path]


@dataclass(frozen=True)
class FamilyPlan:
    """Declared within-family case order for the controlled tiny fixture."""

    task_family_id: str
    adaptation_order: Tuple[str, ...]
    held_out_case_id: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FamilyPlan":
        if not isinstance(data, Mapping):
            raise ValueError("family plan must be an object")
        required = {"task_family_id", "adaptation_order", "held_out_case_id"}
        missing = required.difference(data)
        if missing:
            raise ValueError(f"family plan missing required fields: {sorted(missing)}")
        if set(data) != required:
            raise ValueError("family plan contains unsupported fields")
        order = data["adaptation_order"]
        if not _is_sequence(order):
            raise ValueError("adaptation_order must be a sequence")
        values = tuple(order)
        if not all(isinstance(value, str) and value.strip() for value in values):
            raise ValueError("adaptation_order must contain nonempty IDs")
        family_id = data["task_family_id"]
        held_out = data["held_out_case_id"]
        if not isinstance(family_id, str) or not family_id.strip():
            raise ValueError("task_family_id must be nonempty")
        if not isinstance(held_out, str) or not held_out.strip():
            raise ValueError("held_out_case_id must be nonempty")
        return cls(family_id, values, held_out)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_family_id": self.task_family_id,
            "adaptation_order": list(self.adaptation_order),
            "held_out_case_id": self.held_out_case_id,
        }


@dataclass(frozen=True)
class DatasetBundle:
    """Versioned wire bundle containing cases and separately revealed feedback."""

    dataset_bundle_version: str
    schema_version: str
    dataset_version: str
    description: str
    claim_limit: str
    family_order: Tuple[str, ...]
    family_plans: Tuple[FamilyPlan, ...]
    cases: Tuple[TaskCase, ...]
    adaptation_feedback: Tuple[FeedbackEvent, ...]
    reveal_order: Tuple[str, ...]
    planned_run_ids: Mapping[str, str]
    runner_precondition: str
    validation_limitations: str
    provenance: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DatasetBundle":
        if not isinstance(data, Mapping):
            raise ValueError("dataset bundle must be an object")
        if "dataset_bundle_version" not in data:
            raise ValueError("dataset_bundle_version is required")
        if data["dataset_bundle_version"] != DATASET_BUNDLE_VERSION:
            raise ValueError(
                "unsupported dataset_bundle_version: "
                f"{data['dataset_bundle_version']}"
            )
        if "schema_version" not in data:
            raise ValueError("schema_version is required")
        if data["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {data['schema_version']}")

        required = {
            "dataset_bundle_version",
            "schema_version",
            "dataset_version",
            "description",
            "claim_limit",
            "family_order",
            "family_plans",
            "cases",
            "adaptation_feedback",
            "reveal_order",
            "planned_run_ids",
            "runner_precondition",
            "validation_limitations",
            "provenance",
        }
        missing = required.difference(data)
        if missing:
            raise ValueError(
                f"dataset bundle missing required fields: {sorted(missing)}"
            )
        extra = set(data).difference(required)
        if extra:
            raise ValueError(
                f"dataset bundle contains unsupported fields: {sorted(extra)}"
            )

        for key in (
            "dataset_version",
            "description",
            "claim_limit",
            "runner_precondition",
            "validation_limitations",
            "provenance",
        ):
            if not isinstance(data[key], str) or not data[key].strip():
                raise ValueError(f"{key} must be nonempty")
        for key in (
            "family_order",
            "family_plans",
            "cases",
            "adaptation_feedback",
            "reveal_order",
        ):
            if not _is_sequence(data[key]):
                raise ValueError(f"{key} must be a sequence")
        if not isinstance(data["planned_run_ids"], Mapping):
            raise ValueError("planned_run_ids must be an object")
        planned = dict(data["planned_run_ids"])
        if not all(
            isinstance(key, str)
            and key.strip()
            and isinstance(value, str)
            and value.strip()
            for key, value in planned.items()
        ):
            raise ValueError("planned_run_ids must map nonempty IDs to nonempty IDs")

        return cls(
            dataset_bundle_version=data["dataset_bundle_version"],
            schema_version=data["schema_version"],
            dataset_version=data["dataset_version"],
            description=data["description"],
            claim_limit=data["claim_limit"],
            family_order=tuple(data["family_order"]),
            family_plans=tuple(
                FamilyPlan.from_dict(item) for item in data["family_plans"]
            ),
            cases=tuple(TaskCase.from_dict(item) for item in data["cases"]),
            adaptation_feedback=tuple(
                FeedbackEvent.from_dict(item) for item in data["adaptation_feedback"]
            ),
            reveal_order=tuple(data["reveal_order"]),
            planned_run_ids=MappingProxyType(planned),
            runner_precondition=data["runner_precondition"],
            validation_limitations=data["validation_limitations"],
            provenance=data["provenance"],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_bundle_version": self.dataset_bundle_version,
            "schema_version": self.schema_version,
            "dataset_version": self.dataset_version,
            "description": self.description,
            "claim_limit": self.claim_limit,
            "family_order": list(self.family_order),
            "family_plans": [plan.to_dict() for plan in self.family_plans],
            "cases": [case.to_dict() for case in self.cases],
            "adaptation_feedback": [
                event.to_dict() for event in self.adaptation_feedback
            ],
            "reveal_order": list(self.reveal_order),
            "planned_run_ids": dict(self.planned_run_ids),
            "runner_precondition": self.runner_precondition,
            "validation_limitations": self.validation_limitations,
            "provenance": self.provenance,
        }


_SECRET_PATTERNS = (
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(
        r"\b(?:password|passwd|token|api[_ -]?key|secret)\b\s*[:=]\s*"
        r"(?:['\"])?[^\s,;]{4,}",
        re.IGNORECASE,
    ),
)


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _normalized_content(value: str) -> str:
    return " ".join(value.casefold().split())


def _strings(value: Any) -> Tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        return tuple(text for item in value.values() for text in _strings(item))
    if _is_sequence(value):
        return tuple(text for item in value for text in _strings(item))
    return ()


def load_dataset_bundle(path: PathLike) -> DatasetBundle:
    """Load and type-check JSON without inferring missing or unsupported versions."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return DatasetBundle.from_dict(payload)


def load_tiny_fixture(path: PathLike) -> DatasetBundle:
    """Load a bundle and apply the controlled six-case fixture constraints."""

    bundle = load_dataset_bundle(path)
    validate_tiny_fixture(bundle)
    return bundle


def canonical_bundle_sha256(bundle: DatasetBundle) -> str:
    """Hash every bundle field using canonical UTF-8 JSON serialization."""

    encoded = json.dumps(
        bundle.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_tiny_fixture(bundle: DatasetBundle) -> None:
    """Validate the six-case controlled corpus, not arbitrary future datasets."""

    expected_families = (
        "hybrid-network-return-routing",
        "terraform-drift-state",
    )
    if bundle.family_order != expected_families:
        raise ValueError(f"family_order must be {expected_families}")
    if (
        len(bundle.family_plans) != 2
        or tuple(plan.task_family_id for plan in bundle.family_plans)
        != bundle.family_order
    ):
        raise ValueError("family plans must be unique and follow family_order")
    if len(bundle.cases) != 6:
        raise ValueError("tiny fixture must contain exactly six cases")

    case_ids = tuple(case.task_case_id for case in bundle.cases)
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("task case IDs must be unique")
    cases_by_id = {case.task_case_id: case for case in bundle.cases}

    declared_family_by_case: Dict[str, str] = {}
    for plan in bundle.family_plans:
        declared_ids = plan.adaptation_order + (plan.held_out_case_id,)
        for case_id in declared_ids:
            if case_id in declared_family_by_case:
                raise ValueError(
                    "declared case IDs must occur in exactly one family plan"
                )
            declared_family_by_case[case_id] = plan.task_family_id
    if set(declared_family_by_case) != set(case_ids):
        raise ValueError("family plans must declare every case exactly once")
    for case in bundle.cases:
        if (
            case.inputs.profile.task_family_id
            != declared_family_by_case[case.task_case_id]
        ):
            raise ValueError("family/profile consistency violation")
        if case.dataset_version != bundle.dataset_version:
            raise ValueError("case dataset_version must match bundle dataset_version")

    expected_case_order = tuple(
        case_id
        for plan in bundle.family_plans
        for case_id in plan.adaptation_order + (plan.held_out_case_id,)
    )
    for plan in bundle.family_plans:
        family_cases = [
            case
            for case in bundle.cases
            if declared_family_by_case[case.task_case_id] == plan.task_family_id
        ]
        counts = Counter(case.split for case in family_cases)
        if counts != {"adaptation": 2, "held_out": 1}:
            raise ValueError(
                "each family must contain exactly two adaptation and one held_out case"
            )
        if len(plan.adaptation_order) != 2:
            raise ValueError("adaptation_order must contain exactly two case IDs")
        if any(
            cases_by_id[item].split != "adaptation" for item in plan.adaptation_order
        ):
            raise ValueError("adaptation_order may contain only adaptation cases")
        if cases_by_id[plan.held_out_case_id].split != "held_out":
            raise ValueError("held_out_case_id must identify the held_out case")
    if case_ids != expected_case_order:
        raise ValueError("cases must follow the declared case order")

    adaptation_ids = {
        case.task_case_id for case in bundle.cases if case.split == "adaptation"
    }
    if set(bundle.planned_run_ids) != adaptation_ids:
        raise ValueError("planned_run_ids must cover exactly the adaptation cases")
    if len(bundle.adaptation_feedback) != 4:
        raise ValueError("tiny fixture must contain exactly four feedback events")
    event_ids = tuple(event.event_id for event in bundle.adaptation_feedback)
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("feedback event IDs must be unique")
    if bundle.reveal_order != event_ids:
        raise ValueError("reveal_order must match feedback event order")

    for event in bundle.adaptation_feedback:
        if event.task_case_id not in cases_by_id:
            raise ValueError("feedback task_case_id must refer to a case")
        case = cases_by_id[event.task_case_id]
        if case.split != "adaptation":
            raise ValueError("feedback is permitted only for adaptation cases")

    events_by_case = Counter(event.task_case_id for event in bundle.adaptation_feedback)
    if set(events_by_case) != adaptation_ids or any(
        count != 1 for count in events_by_case.values()
    ):
        raise ValueError("each adaptation case must have exactly one feedback event")
    for event in bundle.adaptation_feedback:
        case = cases_by_id[event.task_case_id]
        if event.task_family_id != case.inputs.profile.task_family_id:
            raise ValueError("feedback family must match its case profile")
        if event.run_id != bundle.planned_run_ids[event.task_case_id]:
            raise ValueError("feedback run_id must match the synthetic planned run ID")
        candidate_ids = {item.context_item_id for item in case.inputs.candidate_context}
        if not set(event.affected_context_item_ids).issubset(candidate_ids):
            raise ValueError("feedback affected context IDs must refer to that case")
        if event.source != "oracle" or event.signal_type != "context_utility":
            raise ValueError(
                "adaptation feedback must be locked oracle context_utility"
            )
        value = event.structured_value
        required_feedback_fields = {
            "locked",
            "selector_independent",
            "useful_context_item_ids",
            "harmful_context_item_ids",
            "useful_attributes",
            "harmful_attributes",
            "no_effect_attributes",
            "shared_feature_trap",
        }
        if not isinstance(value, Mapping) or not required_feedback_fields.issubset(
            value
        ):
            raise ValueError(
                "oracle feedback must identify useful and harmful items/attributes"
            )
        if value["locked"] is not True or value["selector_independent"] is not True:
            raise ValueError("oracle feedback must be locked and selector-independent")
        identified = set(value["useful_context_item_ids"]) | set(
            value["harmful_context_item_ids"]
        )
        if identified != set(event.affected_context_item_ids):
            raise ValueError(
                "feedback affected IDs must equal identified useful/harmful items"
            )

    all_context_ids = []
    for case in bundle.cases:
        candidates = case.inputs.candidate_context
        if not 6 <= len(candidates) <= 10:
            raise ValueError("each tiny-fixture case must have 6-10 candidates")
        if sum(item.token_count for item in candidates) <= case.inputs.token_budget:
            raise ValueError("candidate tokens must exceed token_budget")
        sealed = case.sealed_evaluation
        groups = (
            sealed.required_context_item_ids,
            sealed.useful_context_item_ids,
            sealed.misleading_context_item_ids,
            sealed.irrelevant_context_item_ids,
        )
        if any(not group for group in groups):
            raise ValueError("each controlled label group must be nonempty")
        candidate_ids = {item.context_item_id for item in candidates}
        partitioned_ids = set().union(*(set(group) for group in groups))
        if partitioned_ids != candidate_ids:
            raise ValueError("sealed labels must collectively cover all candidates")
        rubric_ids = {
            criterion.criterion_id for criterion in sealed.scoring_rubric.criteria
        }
        if rubric_ids != {
            "technical_correctness",
            "required_reasoning_evidence",
            "unsafe_prohibited_actions",
        }:
            raise ValueError(
                "rubric must cover technical, evidence, and unsafe criteria"
            )
        all_context_ids.extend(item.context_item_id for item in candidates)

    for plan in bundle.family_plans:
        adaptation_cases = [cases_by_id[item] for item in plan.adaptation_order]
        held_out = cases_by_id[plan.held_out_case_id]
        adaptation_items = [
            item for case in adaptation_cases for item in case.inputs.candidate_context
        ]
        held_out_items = list(held_out.inputs.candidate_context)
        adaptation_ids_for_family = {item.context_item_id for item in adaptation_items}
        if adaptation_ids_for_family.intersection(
            item.context_item_id for item in held_out_items
        ):
            raise ValueError("held_out context item IDs must be unseen in adaptation")
        adaptation_groups = {
            item.metadata.get("provenance_group") for item in adaptation_items
        }
        held_out_groups = {
            item.metadata.get("provenance_group") for item in held_out_items
        }
        if None in adaptation_groups or None in held_out_groups:
            raise ValueError("every candidate needs a provenance_group")
        if adaptation_groups.intersection(held_out_groups):
            raise ValueError("held_out provenance groups must be unseen in adaptation")
        adaptation_content = {
            _normalized_content(item.content) for item in adaptation_items
        }
        if adaptation_content.intersection(
            _normalized_content(item.content) for item in held_out_items
        ):
            raise ValueError(
                "exact normalized duplicate content across adaptation/held_out"
            )

    if len(set(all_context_ids)) != len(all_context_ids):
        raise ValueError("context item IDs must be unique across the bundle")

    for text in _strings(bundle.to_dict()):
        if any(pattern.search(text) for pattern in _SECRET_PATTERNS):
            raise ValueError("fixture contains an obvious secret-like pattern")
