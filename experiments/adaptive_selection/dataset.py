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
TINY_CLAIM_LIMIT = (
    "This fixture validates harness mechanics only and cannot support "
    "adaptive-selection efficacy claims, empirical generalization, causal "
    "effectiveness, production readiness, or safety claims."
)
TINY_ONTOLOGY_ID = "stage0-neutral-context-attributes"
TINY_ONTOLOGY_VERSION = "2"
TINY_ONTOLOGY_DEFINITIONS = MappingProxyType(
    {
        **{
            f"format:{name}": "Neutral document presentation and structural format attribute."
            for name in (
                "capture",
                "checklist",
                "diagnostic",
                "diagram",
                "diff",
                "forum",
                "inventory",
                "json",
                "list",
                "memo",
                "note",
                "policy",
                "record",
                "runbook",
                "script",
                "table",
            )
        },
        "signal:path-correlation": (
            "Records a correlation between concrete paths or observations."
        ),
        "signal:authority-identity": (
            "Records an identity or authority relationship for comparison."
        ),
        "scope:bounded": "Describes evidence or an action with a bounded scope.",
        "scope:global": "Describes evidence or an action with a global scope.",
        "basis:observed": "States that the record is based on an observed condition.",
        "basis:assumed": "States that the record is based on an assumed condition.",
        "action:reconcile": "Describes reconciling two states or observations.",
        "action:replace": "Describes replacing one state or target with another.",
        "presentation:plain-header": (
            "Document is rendered with a plain presentation header."
        ),
        "presentation:teal-header": (
            "Document is rendered with a teal presentation header."
        ),
    }
)
TINY_ONTOLOGY_CONFIG_HASH = (
    "516106794bb78088152a71aafa5316b1102a43dcd8e5b5798173cb90fa356578"
)
TINY_ONTOLOGY_CONFIG = MappingProxyType(
    {
        "ontology_id": TINY_ONTOLOGY_ID,
        "version": TINY_ONTOLOGY_VERSION,
        "definitions": TINY_ONTOLOGY_DEFINITIONS,
        "config_hash": TINY_ONTOLOGY_CONFIG_HASH,
    }
)
TINY_RUBRIC_ANCHOR_TERMS = {
    "net-adapt-01": ("10.42.8.0/24", "va-31", "vpn-07"),
    "net-adapt-02": ("ia-22", "interfaces A and B", "asymmetric-session"),
    "net-held-01": ("192.0.2.0/24", "pc-14", "bt-09"),
    "tf-adapt-01": ("security-lab", "L-330", "provider 5.4"),
    "tf-adapt-02": ("old_service", "service address", "backend lineage"),
    "tf-held-01": ("lab-west", "L-204", "L-117"),
}
_TINY_RUBRIC_FINGERPRINTS = {
    "net-adapt-01": {
        "instructions": "6571464e1f1434b3c7b75389af36660b525c038bae86e646926bdf1152ec49c6",
        "technical_correctness": "a5f6d266c586e0a0d00db7964b4fac18a7a5afffa174cc4f507edb3db22544d2",
        "required_reasoning_evidence": "3e73c5fb2830593e96faa32a1e1d1d053cba991e5beebc588e3176f45e0efbc6",
        "unsafe_prohibited_actions": "a10067d60665f201e1cf10605e0a9ed61cbca1675a2c848a02d82dd66be7af99",
    },
    "net-adapt-02": {
        "instructions": "664856d9adf69c9959b35bc0f7d0ab05b4bf43be9aea5144021bd0cbcaff15c8",
        "technical_correctness": "b91a4fe14b26c73ea242bf421babeb74dfa37920af129b59c3e2bb6bb51b6da7",
        "required_reasoning_evidence": "be54d62d842b73dd90ee442875c82fe37f0c7b37f73459c91c632e58bc81b4ff",
        "unsafe_prohibited_actions": "2c041b14db494971582b19207bdc67cce0172e3c8bb1652480d312ab63629a21",
    },
    "net-held-01": {
        "instructions": "1a6c119f76f528462f67966af50c5a2d11fb792e081b19486975312eaf9b18da",
        "technical_correctness": "022004985f02af780404a8df002cf3980aad7f16b909808bb0b7bf12e881663f",
        "required_reasoning_evidence": "4af76dadbc732394aeedbd3620ef4d07328a7047a1652588275eda908ee0c020",
        "unsafe_prohibited_actions": "6053d3924866465803e7c1a6f6ace76461e4631614c5361d92a4428ba6073368",
    },
    "tf-adapt-01": {
        "instructions": "11f1cc6e4e1c61353128720f3b0be1a26bd8cfe3cc97f4f537954c5ecbc4ed10",
        "technical_correctness": "e21c091768d69a0a1247f4da44e7cdba2299a281ff27b260605db87e9aef7670",
        "required_reasoning_evidence": "a12b59b4867c14e32e121b511bc43f2d73b68311de5cfd4cd86f8e2eb288086d",
        "unsafe_prohibited_actions": "c85c3f3566ed75294e02380fa141cf90e4ff5502735c1a839502082053fbeb3b",
    },
    "tf-adapt-02": {
        "instructions": "005766fae417a857068f2f33c63f13ec9e161a635b25a852cf1bcb0b3f7cb7af",
        "technical_correctness": "403b8d7d6cfec969901e5083c6625258992cbd625fe67baefd8f4b06cc596ab1",
        "required_reasoning_evidence": "52696c8d1a71a2f20234a0aff0407008c1d1fe4ad191340e2472195db44b1a9a",
        "unsafe_prohibited_actions": "af737754691e1b37833ae6993eca0f928d611ce8d5375cc304c7194cd5389433",
    },
    "tf-held-01": {
        "instructions": "e1a69c4ca26d7bbf687dfbfdffb9a67fc9214a0b712cb3b7a929815cd336931a",
        "technical_correctness": "71c83622b24de763c70d2e8735662f9afc5de23e1e902eb85cc97a7db714fd71",
        "required_reasoning_evidence": "0c1d6c352868fe0f665e9c77e79080fb6a120ae1180925bcca2190413764255e",
        "unsafe_prohibited_actions": "a6348807fdfce4c608c58e572c684a86f4e562e764c93a72d32e270a02925b13",
    },
}


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


_FREE_TEXT_SECRET_PATTERNS = (
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(
        r"\b(?:password|passwd|token|api[_ -]?key|secret)\b\s*[:=]\s*"
        r"(?:['\"])?[^\s,;]{4,}",
        re.I,
    ),
)
_CREDENTIAL_KEY = re.compile(r"^(?:api[_-]?key|token|password|passwd|secret)$", re.I)


def _check_for_secrets(value: Any) -> None:
    """Recursively inspect mapping pairs, free text, and sequence members."""

    if isinstance(value, str):
        if any(pattern.search(value) for pattern in _FREE_TEXT_SECRET_PATTERNS):
            raise ValueError("fixture contains an obvious secret-like pattern")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str):
                _check_for_secrets(key)
                if (
                    _CREDENTIAL_KEY.fullmatch(key.strip())
                    and isinstance(item, str)
                    and len(item.strip()) >= 4
                ):
                    raise ValueError(
                        "fixture contains an obvious secret-like mapping pair"
                    )
            _check_for_secrets(item)
        return
    if _is_sequence(value):
        for item in value:
            _check_for_secrets(item)


def _normalized_content(value: str) -> str:
    return " ".join(value.casefold().split())


def _normalized_sha256(value: str) -> str:
    return hashlib.sha256(_normalized_content(value).encode("utf-8")).hexdigest()


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

    if bundle.claim_limit != TINY_CLAIM_LIMIT:
        raise ValueError(
            "claim_limit must exactly state mechanics-only and no efficacy claim"
        )

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
        raise ValueError("ontology must contain exactly its frozen fields")
    if ontology["config_hash"] != _config_hash(ontology):
        raise ValueError("ontology config hash mismatch")
    if _thaw(ontology) != TINY_ONTOLOGY_CONFIG:
        raise ValueError("ontology must exactly match the frozen tiny-fixture config")
    definitions = ontology["definitions"]

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
    if len(set(bundle.case_provenance_groups.values())) != len(bundle.cases):
        raise ValueError("case provenance group must be unique for all six cases")

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
    all_normalized_contents = []
    attrs_by_case: Dict[str, Dict[str, set]] = {}
    candidate_counts = []
    positive_budget_deltas = []
    labels_by_count_position: Dict[Tuple[int, int], set] = {}
    learning_attr_classes: Dict[str, set] = {}
    source_classes: Dict[str, set] = {}
    source_occurrences: Counter = Counter()
    metadata_tuple_classes: Dict[str, set] = {}
    rubric_ids = []
    rubric_instructions = []
    criterion_descriptions: Dict[str, list] = {
        "technical_correctness": [],
        "required_reasoning_evidence": [],
        "unsafe_prohibited_actions": [],
    }
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

        selector_text = " ".join(_strings(case.inputs.to_dict())).casefold()
        if re.search(r"\b(?:adapt(?:ation)?|held(?:[_ -]?out)?)\b", selector_text):
            raise ValueError("selector_inputs must not contain split-like markers")

        candidates = case.inputs.candidate_context
        candidate_counts.append(len(candidates))
        if len(candidates) not in {7, 8, 9}:
            raise ValueError("each tiny-fixture case must have 7, 8, or 9 candidates")
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
        if len(candidate_by_id) != len(candidates):
            raise ValueError("candidate context item IDs must be unique per case")
        group_sequences = tuple(groups.values())
        if any(len(values) != len(set(values)) for values in group_sequences):
            raise ValueError("sealed label sequences must be duplicate-free")
        flattened_labels = tuple(item for values in group_sequences for item in values)
        if len(flattened_labels) != len(set(flattened_labels)):
            raise ValueError("sealed label groups must be disjoint")
        if set(flattened_labels) != set(candidate_by_id):
            raise ValueError("sealed labels must collectively cover all candidates")
        nonnegative = set(groups["required"]) | set(groups["useful"])
        positive_total = sum(candidate_by_id[item].token_count for item in nonnegative)
        positive_budget_deltas.append(positive_total - case.inputs.token_budget)
        labels_by_id = {
            item_id: label for label, values in groups.items() for item_id in values
        }
        for position, item in enumerate(candidates):
            labels_by_count_position.setdefault((len(candidates), position), set()).add(
                labels_by_id[item.context_item_id]
            )
        label_runs = [labels_by_id[item.context_item_id] for item in candidates]
        if any(
            label_runs.count(label) > 1
            and max(index for index, value in enumerate(label_runs) if value == label)
            - min(index for index, value in enumerate(label_runs) if value == label)
            + 1
            == label_runs.count(label)
            for label in groups
        ):
            raise ValueError(
                "candidate label classes must not form contiguous role blocks"
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
            label = labels_by_id[item.context_item_id]
            for attribute in learning:
                learning_attr_classes.setdefault(attribute, set()).add(label)
            source_classes.setdefault(item.source, set()).add(label)
            source_occurrences[item.source] += 1
            metadata_key = json.dumps(
                _thaw(item.metadata), sort_keys=True, separators=(",", ":")
            )
            metadata_tuple_classes.setdefault(metadata_key, set()).add(label)
            for attribute in learning | controls | {fmt}:
                if attribute not in definitions:
                    raise ValueError("candidate attribute is absent from ontology")
            all_context_ids.append(item.context_item_id)
            normalized = _normalized_content(item.content)
            all_normalized_contents.append(normalized)
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
        criteria_ids = tuple(item.criterion_id for item in rubric.criteria)
        if len(criteria_ids) != 3 or set(criteria_ids) != expected_criteria:
            raise ValueError("rubric must contain three unique required criterion IDs")
        if any(len(item.description.split()) < 14 for item in rubric.criteria):
            raise ValueError(
                "rubric criterion descriptions must be substantive and case-specific"
            )
        anchors = TINY_RUBRIC_ANCHOR_TERMS.get(case.task_case_id)
        if anchors is None:
            raise ValueError("rubric anchor terms must be sealed for every case")
        rubric_texts = (rubric.instructions,) + tuple(
            item.description for item in rubric.criteria
        )
        if any(
            not all(anchor.casefold() in text.casefold() for anchor in anchors)
            for text in rubric_texts
        ):
            raise ValueError(
                "rubric text must contain its sealed concrete anchor terms"
            )
        rubric_fingerprints = {
            "instructions": _normalized_sha256(rubric.instructions),
            **{
                item.criterion_id: _normalized_sha256(item.description)
                for item in rubric.criteria
            },
        }
        if rubric_fingerprints != _TINY_RUBRIC_FINGERPRINTS.get(case.task_case_id):
            raise ValueError("rubric must match its frozen per-case rubric fingerprint")
        rubric_ids.append(rubric.rubric_id)
        rubric_instructions.append(_normalized_content(rubric.instructions))
        for item in rubric.criteria:
            criterion_descriptions[item.criterion_id].append(
                _normalized_content(item.description)
            )
        if not math.isclose(
            sum(float(item.weight) for item in rubric.criteria), 1.0, abs_tol=1e-9
        ):
            raise ValueError("rubric weights must sum to 1")

    if Counter(candidate_counts) != {7: 2, 8: 2, 9: 2}:
        raise ValueError("candidate counts must use 7, 8, and 9 exactly twice")
    if not any(delta < 0 for delta in positive_budget_deltas) or not any(
        delta > 0 for delta in positive_budget_deltas
    ):
        raise ValueError(
            "positive-set budget deltas must include shortfalls and surpluses"
        )
    if len(set(positive_budget_deltas)) < 4:
        raise ValueError("positive-set budget deltas must have varied magnitudes")
    if len({case.inputs.token_budget for case in bundle.cases}) != len(bundle.cases):
        raise ValueError("token budgets must be independent across cases")
    for count in (7, 8, 9):
        conflicts = sum(
            len(labels_by_count_position[(count, position)]) > 1
            for position in range(count)
        )
        if conflicts < 2:
            raise ValueError(
                "candidate count and ordinal position must conflict across paired cases"
            )
    if any(len(classes) < 2 for classes in learning_attr_classes.values()):
        raise ValueError(
            "every reusable learning attribute must span at least two utility classes"
        )
    if any(
        source_occurrences[source] > 1 and len(classes) < 2
        for source, classes in source_classes.items()
    ):
        raise ValueError(
            "every recurring source must span at least two utility classes"
        )
    if not any(len(classes) > 1 for classes in metadata_tuple_classes.values()):
        raise ValueError(
            "visible metadata tuple lookup must not perfectly predict labels"
        )
    if len(rubric_ids) != len(set(rubric_ids)):
        raise ValueError("rubric IDs must be unique across cases")
    if len(rubric_instructions) != len(set(rubric_instructions)):
        raise ValueError("rubric instructions must be nonidentical across cases")
    if any(
        len(values) != len(set(values)) for values in criterion_descriptions.values()
    ):
        raise ValueError("criterion descriptions must be nonidentical across cases")

    if len(set(all_context_ids)) != len(all_context_ids):
        raise ValueError("context item IDs must be unique across the bundle")
    if len(all_normalized_contents) != len(set(all_normalized_contents)):
        raise ValueError(
            "normalized duplicate content; candidate content must be globally unique"
        )
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
            "ambiguous_attributes",
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
        if len(useful_ids) != len(set(useful_ids)) or len(harmful_ids) != len(
            set(harmful_ids)
        ):
            raise ValueError(
                "feedback context item ID sequences must be duplicate-free"
            )
        sealed = case.sealed_evaluation
        expected_useful_sequence = (
            sealed.required_context_item_ids + sealed.useful_context_item_ids
        )
        expected_harmful_sequence = sealed.misleading_context_item_ids
        expected_useful = set(expected_useful_sequence)
        expected_harmful = set(expected_harmful_sequence)
        if (
            useful_ids != expected_useful_sequence
            or harmful_ids != expected_harmful_sequence
        ):
            raise ValueError(
                "feedback useful/harmful IDs must exactly match sealed labels"
            )
        affected_ids = event.affected_context_item_ids
        if len(affected_ids) != len(set(affected_ids)):
            raise ValueError("feedback affected ID sequence must be duplicate-free")
        if affected_ids != expected_useful_sequence + expected_harmful_sequence:
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
        expected_ambiguous_attrs = expected_useful_attrs & expected_harmful_attrs
        expected_useful_only = expected_useful_attrs - expected_harmful_attrs
        expected_harmful_only = expected_harmful_attrs - expected_useful_attrs
        useful_attrs = _string_tuple("useful_attributes", value["useful_attributes"])
        harmful_attrs = _string_tuple("harmful_attributes", value["harmful_attributes"])
        ambiguous_attrs = _string_tuple(
            "ambiguous_attributes", value["ambiguous_attributes"]
        )
        attribute_sequences = (useful_attrs, harmful_attrs, ambiguous_attrs)
        if any(len(items) != len(set(items)) for items in attribute_sequences):
            raise ValueError("feedback attribute sequences must be duplicate-free")
        if (
            useful_attrs != tuple(sorted(expected_useful_only))
            or harmful_attrs != tuple(sorted(expected_harmful_only))
            or ambiguous_attrs != tuple(sorted(expected_ambiguous_attrs))
        ):
            raise ValueError(
                "feedback attributes must exactly partition labeled learning attributes"
            )
        if (
            set(useful_attrs) & set(harmful_attrs)
            or set(useful_attrs) & set(ambiguous_attrs)
            or set(harmful_attrs) & set(ambiguous_attrs)
        ):
            raise ValueError("feedback attribute polarity categories must be disjoint")
        control_sequence = _string_tuple(
            "no_effect_attributes", value["no_effect_attributes"]
        )
        if len(control_sequence) != len(set(control_sequence)):
            raise ValueError("no-effect attribute sequence must be duplicate-free")
        controls = set(control_sequence)
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
                for item in expected_harmful
            )
        )
        if not controls or not controls.issubset(positive_controls & negative_controls):
            raise ValueError(
                "no-effect attributes must span positive and negative utility classes"
            )
        if control_sequence != ("presentation:teal-header",):
            raise ValueError("no_effect_attributes must exactly equal the teal control")
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
        if trap != "format:runbook":
            raise ValueError("shared_feature_trap must exactly equal format:runbook")

    for plan in bundle.family_plans:
        adaptation_attributes = set().union(
            *(
                set().union(*attrs_by_case[item].values())
                for item in plan.adaptation_order
            )
        )
        held_attributes = set().union(*attrs_by_case[plan.held_out_case_id].values())
        if not held_attributes.issubset(adaptation_attributes):
            raise ValueError(
                "heldout learning attributes must be represented in adaptation"
            )

    _check_for_secrets(bundle.to_dict())
