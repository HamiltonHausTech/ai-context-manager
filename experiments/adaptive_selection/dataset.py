"""Strict loading for the deterministic Stage 0 adaptive-selection corpus.

This fixture validates harness mechanics only and cannot support adaptive-selection
 efficacy claims. Exact duplicate checks are mechanical; semantic and template-sibling
review remains a manual domain responsibility.
"""

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Dict, Mapping, Sequence, Tuple, Union

from .schema import FeedbackEvent, SCHEMA_VERSION, TaskCase

DATASET_BUNDLE_VERSION = "1"
PathLike = Union[str, Path]
TOKENIZER_ID = "stdlib-unicode-regex"
TOKENIZER_VERSION = "1"
TOKENIZER_SCOPE = "ContextItem.content"
TOKENIZER_PATTERN = r"(?u)\w+(?:[./:-]\w+)*"


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _nonempty_string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _string_tuple(name: str, value: Any) -> Tuple[str, ...]:
    if not _is_sequence(value):
        raise ValueError(f"{name} must be a sequence")
    result = tuple(value)
    if any(not isinstance(item, str) or not item.strip() for item in result):
        raise ValueError(f"{name} must contain nonempty strings")
    return result


def _freeze_json(value: Any, path: str) -> Any:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite")
        return value
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ValueError(f"{path} object keys must be strings")
        return MappingProxyType(
            {key: _freeze_json(value[key], f"{path}.{key}") for key in sorted(value)}
        )
    if type(value) in (list, tuple):
        return tuple(_freeze_json(item, f"{path}[]") for item in value)
    raise ValueError(f"{path} contains unsupported JSON value")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _config_hash(value: Mapping[str, Any]) -> str:
    body = {key: _thaw(item) for key, item in value.items() if key != "config_hash"}
    encoded = json.dumps(
        body, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def count_context_tokens(content: str) -> int:
    """Count tokens using the frozen stdlib-only Unicode regex convention."""

    _nonempty_string("content", content)
    return len(re.findall(TOKENIZER_PATTERN, content))


@dataclass(frozen=True)
class FamilyPlan:
    task_family_id: str
    adaptation_order: Tuple[str, ...]
    held_out_case_id: str

    def __post_init__(self) -> None:
        _nonempty_string("task_family_id", self.task_family_id)
        object.__setattr__(
            self,
            "adaptation_order",
            _string_tuple("adaptation_order", self.adaptation_order),
        )
        _nonempty_string("held_out_case_id", self.held_out_case_id)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FamilyPlan":
        if not isinstance(data, Mapping):
            raise ValueError("family plan must be an object")
        required = {"task_family_id", "adaptation_order", "held_out_case_id"}
        if set(data) != required:
            raise ValueError("family plan must contain exactly its required fields")
        return cls(
            data["task_family_id"], data["adaptation_order"], data["held_out_case_id"]
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_family_id": self.task_family_id,
            "adaptation_order": list(self.adaptation_order),
            "held_out_case_id": self.held_out_case_id,
        }


@dataclass(frozen=True)
class DatasetBundle:
    dataset_bundle_version: str
    schema_version: str
    dataset_version: str
    description: str
    claim_limit: str
    tokenizer_config: Mapping[str, Any]
    ontology: Mapping[str, Any]
    case_provenance_groups: Mapping[str, str]
    family_order: Tuple[str, ...]
    family_plans: Tuple[FamilyPlan, ...]
    cases: Tuple[TaskCase, ...]
    adaptation_feedback: Tuple[FeedbackEvent, ...]
    reveal_order: Tuple[str, ...]
    planned_run_ids: Mapping[str, str]
    runner_precondition: str
    validation_limitations: str
    provenance: str

    def __post_init__(self) -> None:
        if self.dataset_bundle_version != DATASET_BUNDLE_VERSION:
            raise ValueError(
                f"unsupported dataset_bundle_version: {self.dataset_bundle_version}"
            )
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        for name in (
            "dataset_version",
            "description",
            "claim_limit",
            "runner_precondition",
            "validation_limitations",
            "provenance",
        ):
            _nonempty_string(name, getattr(self, name))
        object.__setattr__(
            self, "family_order", _string_tuple("family_order", self.family_order)
        )
        object.__setattr__(
            self, "reveal_order", _string_tuple("reveal_order", self.reveal_order)
        )
        plans = tuple(self.family_plans)
        if not all(isinstance(item, FamilyPlan) for item in plans):
            raise ValueError("family_plans must contain FamilyPlan records")
        object.__setattr__(self, "family_plans", plans)
        cases = tuple(self.cases)
        if not all(isinstance(item, TaskCase) for item in cases):
            raise ValueError("cases must contain TaskCase records")
        object.__setattr__(self, "cases", cases)
        events = tuple(self.adaptation_feedback)
        if not all(isinstance(item, FeedbackEvent) for item in events):
            raise ValueError("adaptation_feedback must contain FeedbackEvent records")
        object.__setattr__(self, "adaptation_feedback", events)
        for name in (
            "tokenizer_config",
            "ontology",
            "case_provenance_groups",
            "planned_run_ids",
        ):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise ValueError(f"{name} must be an object")
            object.__setattr__(self, name, _freeze_json(value, name))
        if any(
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(value, str)
            or not value.strip()
            for key, value in self.case_provenance_groups.items()
        ):
            raise ValueError("case provenance groups must map nonempty strings")
        if any(
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(value, str)
            or not value.strip()
            for key, value in self.planned_run_ids.items()
        ):
            raise ValueError("planned_run_ids must map nonempty strings")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DatasetBundle":
        if not isinstance(data, Mapping):
            raise ValueError("dataset bundle must be an object")
        for key in ("dataset_bundle_version", "schema_version"):
            if key not in data:
                raise ValueError(f"{key} is required")
        required = {
            "dataset_bundle_version",
            "schema_version",
            "dataset_version",
            "description",
            "claim_limit",
            "tokenizer_config",
            "ontology",
            "case_provenance_groups",
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
            "family_order",
            "family_plans",
            "cases",
            "adaptation_feedback",
            "reveal_order",
        ):
            if not _is_sequence(data[key]):
                raise ValueError(f"{key} must be a sequence")
        return cls(
            dataset_bundle_version=data["dataset_bundle_version"],
            schema_version=data["schema_version"],
            dataset_version=data["dataset_version"],
            description=data["description"],
            claim_limit=data["claim_limit"],
            tokenizer_config=data["tokenizer_config"],
            ontology=data["ontology"],
            case_provenance_groups=data["case_provenance_groups"],
            family_order=data["family_order"],
            family_plans=tuple(
                FamilyPlan.from_dict(item) for item in data["family_plans"]
            ),
            cases=tuple(TaskCase.from_dict(item) for item in data["cases"]),
            adaptation_feedback=tuple(
                FeedbackEvent.from_dict(item) for item in data["adaptation_feedback"]
            ),
            reveal_order=data["reveal_order"],
            planned_run_ids=data["planned_run_ids"],
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
            "tokenizer_config": _thaw(self.tokenizer_config),
            "ontology": _thaw(self.ontology),
            "case_provenance_groups": _thaw(self.case_provenance_groups),
            "family_order": list(self.family_order),
            "family_plans": [item.to_dict() for item in self.family_plans],
            "cases": [item.to_dict() for item in self.cases],
            "adaptation_feedback": [
                item.to_dict() for item in self.adaptation_feedback
            ],
            "reveal_order": list(self.reveal_order),
            "planned_run_ids": _thaw(self.planned_run_ids),
            "runner_precondition": self.runner_precondition,
            "validation_limitations": self.validation_limitations,
            "provenance": self.provenance,
        }


_SECRET_PATTERNS = (
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(
        r"\b(?:password|passwd|token|api[_ -]?key|secret)\b\s*[:=]\s*(?:['\"])?[^\s,;]{4,}",
        re.I,
    ),
)


def _normalized_content(value: str) -> str:
    return " ".join(value.casefold().split())


def _strings(value: Any) -> Tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        return tuple(
            text
            for key, item in value.items()
            for text in (_strings(key) + _strings(item))
        )
    if _is_sequence(value):
        return tuple(text for item in value for text in _strings(item))
    return ()


def load_dataset_bundle(path: PathLike) -> DatasetBundle:
    with Path(path).open("r", encoding="utf-8") as handle:
        return DatasetBundle.from_dict(json.load(handle))


def load_tiny_fixture(path: PathLike) -> DatasetBundle:
    bundle = load_dataset_bundle(path)
    validate_tiny_fixture(bundle)
    return bundle


def canonical_bundle_sha256(bundle: DatasetBundle) -> str:
    encoded = json.dumps(
        bundle.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _candidate_attributes(item: Any) -> Tuple[set, set, str]:
    metadata = item.metadata
    if set(metadata) != {"learning_attributes", "control_attributes", "format"}:
        raise ValueError("candidate metadata must use the neutral ontology fields")
    learning = set(
        _string_tuple("learning_attributes", metadata["learning_attributes"])
    )
    controls = set(_string_tuple("control_attributes", metadata["control_attributes"]))
    fmt = _nonempty_string("format", metadata["format"])
    return learning, controls, fmt


def validate_tiny_fixture(bundle: DatasetBundle) -> None:
    expected_families = ("hybrid-network-return-routing", "terraform-drift-state")
    if bundle.family_order != expected_families:
        raise ValueError(f"family_order must be {expected_families}")
    if (
        len(bundle.family_plans) != 2
        or tuple(p.task_family_id for p in bundle.family_plans) != bundle.family_order
    ):
        raise ValueError("family plans must be unique and follow family_order")
    if len(bundle.cases) != 6:
        raise ValueError("tiny fixture must contain exactly six cases")

    claim = bundle.claim_limit.casefold()
    if (
        "mechanics only" not in claim
        or "cannot support adaptive-selection efficacy" not in claim
    ):
        raise ValueError(
            "claim_limit must state mechanics only and reject efficacy support"
        )
    if re.search(
        r"\b(?:proves?|demonstrates?|establishes?|validates?)\b.{0,30}\b(?:efficacy|generalization|effectiveness)\b",
        claim,
    ):
        raise ValueError("claim_limit contains an efficacy overclaim")

    tokenizer = bundle.tokenizer_config
    required_tokenizer = {
        "tokenizer_id": TOKENIZER_ID,
        "version": TOKENIZER_VERSION,
        "scope": TOKENIZER_SCOPE,
        "pattern": TOKENIZER_PATTERN,
    }
    if any(
        tokenizer.get(key) != value for key, value in required_tokenizer.items()
    ) or set(tokenizer) != set(required_tokenizer) | {"config_hash"}:
        raise ValueError("tokenizer configuration does not match the frozen convention")
    if tokenizer.get("config_hash") != _config_hash(tokenizer):
        raise ValueError("tokenizer config hash mismatch")

    ontology = bundle.ontology
    if set(ontology) != {"ontology_id", "version", "definitions", "config_hash"}:
        raise ValueError("ontology must contain its frozen fields")
    _nonempty_string("ontology_id", ontology["ontology_id"])
    _nonempty_string("ontology version", ontology["version"])
    definitions = ontology["definitions"]
    if not isinstance(definitions, Mapping) or not definitions:
        raise ValueError("ontology definitions must be a nonempty object")
    if any(
        not isinstance(k, str)
        or not k.strip()
        or not isinstance(v, str)
        or len(v.split()) < 3
        for k, v in definitions.items()
    ):
        raise ValueError(
            "ontology attributes and definitions must be substantive strings"
        )
    if ontology["config_hash"] != _config_hash(ontology):
        raise ValueError("ontology config hash mismatch")

    case_ids = tuple(case.task_case_id for case in bundle.cases)
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("task case IDs must be unique")
    cases_by_id = {case.task_case_id: case for case in bundle.cases}
    declared: Dict[str, str] = {}
    expected_case_order = []
    expected_adaptation_order = []
    for plan in bundle.family_plans:
        if len(plan.adaptation_order) != 2:
            raise ValueError("adaptation_order must contain exactly two case IDs")
        expected_adaptation_order.extend(plan.adaptation_order)
        for case_id in plan.adaptation_order + (plan.held_out_case_id,):
            if case_id in declared:
                raise ValueError(
                    "declared case IDs must occur in exactly one family plan"
                )
            declared[case_id] = plan.task_family_id
            expected_case_order.append(case_id)
    if set(declared) != set(case_ids) or tuple(expected_case_order) != case_ids:
        raise ValueError("cases must follow the declared case order")
    if set(bundle.case_provenance_groups) != set(case_ids):
        raise ValueError("case provenance groups must cover every case")

    adaptation_ids = set(expected_adaptation_order)
    for plan in bundle.family_plans:
        family_cases = [
            cases_by_id[item]
            for item in plan.adaptation_order + (plan.held_out_case_id,)
        ]
        if Counter(case.split for case in family_cases) != {
            "adaptation": 2,
            "held_out": 1,
        }:
            raise ValueError(
                "each family must contain exactly two adaptation and one held_out case"
            )
        if (
            any(
                cases_by_id[item].split != "adaptation"
                for item in plan.adaptation_order
            )
            or cases_by_id[plan.held_out_case_id].split != "held_out"
        ):
            raise ValueError("family plan split assignments are invalid")

    if set(bundle.planned_run_ids) != adaptation_ids:
        raise ValueError("planned_run_ids must cover exactly the adaptation cases")
    if len(bundle.adaptation_feedback) != 4:
        raise ValueError("tiny fixture must contain exactly four feedback events")
    event_ids = tuple(event.event_id for event in bundle.adaptation_feedback)
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("feedback event IDs must be unique")
    if bundle.reveal_order != event_ids:
        raise ValueError("reveal_order must match feedback event order")
    if tuple(event.task_case_id for event in bundle.adaptation_feedback) != tuple(
        expected_adaptation_order
    ):
        raise ValueError("feedback order must equal flattened family adaptation order")

    all_context_ids = []
    adaptation_groups = set()
    heldout_groups = []
    adaptation_contents = set()
    heldout_contents = set()
    attrs_by_case: Dict[str, Dict[str, set]] = {}
    forbidden_id_words = re.compile(
        r"adapt|held|unsafe|stale|destructive|irrelevant|mislead|useful|required", re.I
    )

    for case in bundle.cases:
        if case.inputs.profile.task_family_id != declared[case.task_case_id]:
            raise ValueError("family/profile consistency violation")
        if case.dataset_version != bundle.dataset_version:
            raise ValueError("case dataset_version must match bundle dataset_version")
        group = bundle.case_provenance_groups[case.task_case_id]
        _nonempty_string("provenance group", group)
        if (
            case.provenance != group
            or case.inputs.provenance != group
            or any(item.provenance != group for item in case.inputs.candidate_context)
        ):
            raise ValueError(
                "case and candidate provenance must match the sealed case group"
            )
        if case.split == "adaptation":
            adaptation_groups.add(group)
        else:
            heldout_groups.append(group)

        candidates = case.inputs.candidate_context
        if not 6 <= len(candidates) <= 10:
            raise ValueError("each tiny-fixture case must have 6-10 candidates")
        total_tokens = sum(item.token_count for item in candidates)
        if total_tokens <= case.inputs.token_budget:
            raise ValueError("candidate tokens must exceed token_budget")
        sealed = case.sealed_evaluation
        groups = {
            "required": sealed.required_context_item_ids,
            "useful": sealed.useful_context_item_ids,
            "misleading": sealed.misleading_context_item_ids,
            "irrelevant": sealed.irrelevant_context_item_ids,
        }
        if any(not values for values in groups.values()):
            raise ValueError("each controlled label group must be nonempty")
        candidate_by_id = {item.context_item_id: item for item in candidates}
        if set().union(*(set(values) for values in groups.values())) != set(
            candidate_by_id
        ):
            raise ValueError("sealed labels must collectively cover all candidates")
        nonnegative = set(groups["required"]) | set(groups["useful"])
        if (
            sum(candidate_by_id[item].token_count for item in nonnegative)
            <= case.inputs.token_budget
        ):
            raise ValueError(
                "token budget must force a tradeoff among nonnegative items"
            )
        for item in candidates:
            if forbidden_id_words.search(
                item.context_item_id
            ) or forbidden_id_words.search(item.source):
                raise ValueError("candidate IDs and sources must be neutral")
            if item.token_count != count_context_tokens(item.content):
                raise ValueError(
                    "candidate token_count does not match tokenizer convention"
                )
            learning, controls, fmt = _candidate_attributes(item)
            for attribute in learning | controls | {fmt}:
                if attribute not in definitions:
                    raise ValueError("candidate attribute is absent from ontology")
            all_context_ids.append(item.context_item_id)
            normalized = _normalized_content(item.content)
            (
                adaptation_contents if case.split == "adaptation" else heldout_contents
            ).add(normalized)
        attrs_by_case[case.task_case_id] = {
            label: set().union(
                *(_candidate_attributes(candidate_by_id[item])[0] for item in ids)
            )
            for label, ids in groups.items()
        }
        rubric = sealed.scoring_rubric
        if len(sealed.gold_answer.split()) < 35:
            raise ValueError("gold answer must contain at least 35 words")
        if len(rubric.instructions.split()) < 12:
            raise ValueError(
                "rubric instructions must be substantive and case-specific"
            )
        expected_criteria = {
            "technical_correctness",
            "required_reasoning_evidence",
            "unsafe_prohibited_actions",
        }
        if {item.criterion_id for item in rubric.criteria} != expected_criteria:
            raise ValueError("rubric must contain the required three criterion IDs")
        if any(len(item.description.split()) < 10 for item in rubric.criteria):
            raise ValueError(
                "rubric criterion descriptions must be substantive and case-specific"
            )
        if not math.isclose(
            sum(float(item.weight) for item in rubric.criteria), 1.0, abs_tol=1e-9
        ):
            raise ValueError("rubric weights must sum to 1")

    if len(set(all_context_ids)) != len(all_context_ids):
        raise ValueError("context item IDs must be unique across the bundle")
    if adaptation_groups.intersection(heldout_groups):
        raise ValueError(
            "no adaptation provenance group may appear in heldout cases globally"
        )
    if len(set(heldout_groups)) != len(heldout_groups):
        raise ValueError("heldout provenance groups must be globally unique")
    if adaptation_contents.intersection(heldout_contents):
        raise ValueError(
            "exact normalized duplicate content across adaptation/heldout globally"
        )

    for event in bundle.adaptation_feedback:
        case = cases_by_id[event.task_case_id]
        if event.source != "oracle" or event.signal_type != "context_utility":
            raise ValueError(
                "adaptation feedback must be locked oracle context_utility"
            )
        if (
            event.task_family_id != case.inputs.profile.task_family_id
            or event.run_id != bundle.planned_run_ids[event.task_case_id]
        ):
            raise ValueError("feedback identity fields must match the planned case")
        value = event.structured_value
        fields = {
            "locked",
            "selector_independent",
            "useful_context_item_ids",
            "harmful_context_item_ids",
            "useful_attributes",
            "harmful_attributes",
            "no_effect_attributes",
            "shared_feature_trap",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError(
                "oracle feedback must contain exactly the locked control fields"
            )
        if (
            type(value["locked"]) is not bool
            or type(value["selector_independent"]) is not bool
            or value["locked"] is not True
            or value["selector_independent"] is not True
        ):
            raise ValueError("oracle feedback booleans must be strict true values")
        useful_ids = _string_tuple(
            "useful_context_item_ids", value["useful_context_item_ids"]
        )
        harmful_ids = _string_tuple(
            "harmful_context_item_ids", value["harmful_context_item_ids"]
        )
        sealed = case.sealed_evaluation
        expected_useful = set(sealed.required_context_item_ids) | set(
            sealed.useful_context_item_ids
        )
        expected_harmful = set(sealed.misleading_context_item_ids)
        if set(useful_ids) != expected_useful or set(harmful_ids) != expected_harmful:
            raise ValueError(
                "feedback useful/harmful IDs must exactly match sealed labels"
            )
        if set(event.affected_context_item_ids) != expected_useful | expected_harmful:
            raise ValueError(
                "feedback affected IDs must equal useful and harmful union"
            )
        candidate_by_id = {
            item.context_item_id: item for item in case.inputs.candidate_context
        }
        expected_useful_attrs = set().union(
            *(
                _candidate_attributes(candidate_by_id[item])[0]
                for item in expected_useful
            )
        )
        expected_harmful_attrs = set().union(
            *(
                _candidate_attributes(candidate_by_id[item])[0]
                for item in expected_harmful
            )
        )
        if (
            set(_string_tuple("useful_attributes", value["useful_attributes"]))
            != expected_useful_attrs
            or set(_string_tuple("harmful_attributes", value["harmful_attributes"]))
            != expected_harmful_attrs
        ):
            raise ValueError(
                "feedback attributes must be exhaustive labeled learning-attribute unions"
            )
        controls = set(
            _string_tuple("no_effect_attributes", value["no_effect_attributes"])
        )
        trap = _nonempty_string("shared_feature_trap", value["shared_feature_trap"])
        positive_controls = set().union(
            *(
                _candidate_attributes(candidate_by_id[item])[1]
                for item in expected_useful
            )
        )
        negative_controls = set().union(
            *(
                _candidate_attributes(candidate_by_id[item])[1]
                for item in expected_harmful | set(sealed.irrelevant_context_item_ids)
            )
        )
        if not controls or not controls.issubset(positive_controls & negative_controls):
            raise ValueError(
                "no-effect attributes must span positive and negative utility classes"
            )
        positive_formats = {
            _candidate_attributes(candidate_by_id[item])[2] for item in expected_useful
        }
        misleading_formats = {
            _candidate_attributes(candidate_by_id[item])[2] for item in expected_harmful
        }
        if trap not in positive_formats & misleading_formats:
            raise ValueError(
                "shared_feature_trap must occur in positive and misleading candidates"
            )

    for plan in bundle.family_plans:
        adaptation_positive = set().union(
            *(
                attrs_by_case[item]["required"] | attrs_by_case[item]["useful"]
                for item in plan.adaptation_order
            )
        )
        adaptation_harmful = set().union(
            *(attrs_by_case[item]["misleading"] for item in plan.adaptation_order)
        )
        held = attrs_by_case[plan.held_out_case_id]
        if not adaptation_positive.issubset(held["required"] | held["useful"]):
            raise ValueError(
                "heldout positive candidates must carry adaptation useful attributes"
            )
        if not adaptation_harmful.issubset(held["misleading"]):
            raise ValueError(
                "heldout misleading candidates must carry adaptation harmful attributes"
            )

    for text in _strings(bundle.to_dict()):
        if any(pattern.search(text) for pattern in _SECRET_PATTERNS):
            raise ValueError("fixture contains an obvious secret-like pattern")
