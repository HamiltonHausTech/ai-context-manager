"""Versioned, immutable records for adaptive context-selection experiments.

This stdlib-only module contains no persistence, provider, selection, scoring, or
learning implementation. Selector-visible task inputs and sealed evaluation evidence
are separate objects so held-out labels cannot be passed to a selector accidentally.

Schema v1 was an unreleased development format and is intentionally unsupported.
Schema v2 is the first candidate wire format; this module does not claim v1 migration
support and rejects v1 payloads before nested deserialization.

Held-out exact-text checks use ``" ".join(text.casefold().split())``: Unicode-aware
case folding plus collapse of every whitespace run to one ASCII space. A selector-
visible string is rejected when it contains the complete normalized gold answer.
This deliberately does not detect semantic paraphrases; corpus review and dataset
cross-split deduplication remain responsible for those.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime
import math
import re
from types import MappingProxyType
from typing import Any, Dict, Optional, Tuple, Type, TypeVar, cast

SCHEMA_VERSION = "2"
T = TypeVar("T", bound="RecordMixin")
_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$")


def _nonempty(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty")


def _finite_number(name: str, value: Any) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _range(name: str, value: Any, minimum: float, maximum: float) -> None:
    try:
        number = _finite_number(name, value)
    except ValueError:
        raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}")
    if not minimum <= number <= maximum:
        raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}")


def _normalized(name: str, value: Any) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{name} must be between 0 and 1")


def _positive(name: str, value: Any) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be positive")


def _nonnegative_integer(name: str, value: Any) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _nonnegative_finite(name: str, value: Any) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise ValueError(f"{name} must be nonnegative and finite")


def _timestamp(name: str, value: Any) -> None:
    if not isinstance(value, str) or not _TIMESTAMP_RE.fullmatch(value):
        raise ValueError(f"{name} must be canonical UTC RFC 3339 ending in Z")
    try:
        datetime.strptime(
            value, "%Y-%m-%dT%H:%M:%S.%fZ" if "." in value else "%Y-%m-%dT%H:%M:%SZ"
        )
    except ValueError:
        raise ValueError(f"{name} must be canonical UTC RFC 3339 ending in Z")


def _unique_nonempty(name: str, values: Tuple[str, ...]) -> None:
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{name} must not contain empty IDs")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must be unique")


def _paired(
    reference_name: str,
    reference: Optional[str],
    hash_name: str,
    artifact_hash: Optional[str],
) -> None:
    if (reference is None) != (artifact_hash is None):
        raise ValueError(f"{reference_name} and {hash_name} must be provided together")
    if reference is not None:
        _nonempty(reference_name, reference)
        _nonempty(hash_name, artifact_hash)


def _version(data: Mapping) -> str:
    if "schema_version" not in data:
        raise ValueError("schema_version is required")
    version = data["schema_version"]
    if version != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {version}")
    return cast(str, version)


def _freeze_json(value: Any, path: str) -> Any:
    """Validate, defensively copy, and recursively freeze one JSON value."""
    if value is None or type(value) in (bool, str):
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite")
        return value
    if isinstance(value, Mapping):
        for key in value:
            if type(key) is not str:
                raise ValueError(f"{path} object keys must be strings")
        return MappingProxyType(
            {key: _freeze_json(value[key], f"{path}.{key}") for key in sorted(value)}
        )
    if type(value) in (list, tuple):
        return tuple(
            _freeze_json(item, f"{path}[{index}]") for index, item in enumerate(value)
        )
    raise ValueError(f"{path} contains unsupported JSON value: {type(value).__name__}")


def _serialize(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _serialize(getattr(value, item.name)) for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {key: _serialize(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"record contains non-serializable value: {type(value).__name__}")


def _tuple_strings(data: Mapping, key: str) -> Tuple[str, ...]:
    return tuple(data.get(key, ()))


def _record_tuple(name: str, value: Any, record_type: Type[Any]) -> Tuple[Any, ...]:
    """Defensively convert a record sequence and reject bad members clearly."""
    try:
        records = tuple(value)
    except TypeError:
        raise ValueError(f"{name} must be a sequence of {record_type.__name__} records")
    if not all(isinstance(item, record_type) for item in records):
        raise ValueError(f"{name} must contain {record_type.__name__} records")
    return records


class RecordMixin:
    """Common deterministic JSON-compatible serialization for frozen records."""

    schema_version: str

    def _validate_version(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")

    def to_dict(self) -> Dict[str, Any]:
        return cast(Dict[str, Any], _serialize(self))

    @classmethod
    def _from_flat_dict(cls: Type[T], data: Mapping, **changes: Any) -> T:
        _version(data)
        values = dict(data)
        values.update(changes)
        return cls(**values)


@dataclass(frozen=True)
class TaskProfile(RecordMixin):
    task_family_id: str
    name: str
    description: str
    default_token_budget: int
    created_timestamp: str
    provenance: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self._validate_version()
        for name in ("task_family_id", "name", "description", "provenance"):
            _nonempty(name, getattr(self, name))
        _positive("default_token_budget", self.default_token_budget)
        _timestamp("created_timestamp", self.created_timestamp)

    @classmethod
    def from_dict(cls, data: Mapping) -> "TaskProfile":
        return cls._from_flat_dict(data)


@dataclass(frozen=True)
class ContextItem(RecordMixin):
    context_item_id: str
    content: str
    token_count: int
    source: str
    confidence: float
    created_timestamp: str
    provenance: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self._validate_version()
        for name in ("context_item_id", "content", "source", "provenance"):
            _nonempty(name, getattr(self, name))
        _positive("token_count", self.token_count)
        _normalized("confidence", self.confidence)
        _timestamp("created_timestamp", self.created_timestamp)
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be an object")
        object.__setattr__(self, "metadata", _freeze_json(self.metadata, "metadata"))

    @classmethod
    def from_dict(cls, data: Mapping) -> "ContextItem":
        return cls._from_flat_dict(data)


@dataclass(frozen=True)
class RubricCriterion(RecordMixin):
    criterion_id: str
    description: str
    weight: float
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self._validate_version()
        _nonempty("criterion_id", self.criterion_id)
        _nonempty("description", self.description)
        if (
            not isinstance(self.weight, (int, float))
            or isinstance(self.weight, bool)
            or not math.isfinite(float(self.weight))
            or self.weight <= 0
        ):
            raise ValueError("weight must be positive and finite")

    @classmethod
    def from_dict(cls, data: Mapping) -> "RubricCriterion":
        return cls._from_flat_dict(data)


@dataclass(frozen=True)
class ScoringRubric(RecordMixin):
    rubric_id: str
    instructions: str
    criteria: Tuple[RubricCriterion, ...]
    provenance: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self._validate_version()
        object.__setattr__(
            self,
            "criteria",
            _record_tuple("criteria", self.criteria, RubricCriterion),
        )
        _nonempty("rubric_id", self.rubric_id)
        _nonempty("instructions", self.instructions)
        _nonempty("provenance", self.provenance)
        if not self.criteria:
            raise ValueError("criteria must not be empty")
        ids = tuple(item.criterion_id for item in self.criteria)
        if len(set(ids)) != len(ids):
            raise ValueError("criterion IDs must be unique")

    @classmethod
    def from_dict(cls, data: Mapping) -> "ScoringRubric":
        _version(data)
        values = dict(data)
        values["criteria"] = tuple(
            RubricCriterion.from_dict(item) for item in data["criteria"]
        )
        return cls(**values)


@dataclass(frozen=True)
class TaskInputs(RecordMixin):
    """The complete and only payload that may be supplied to a selector."""

    profile: TaskProfile
    task_prompt: str
    candidate_context: Tuple[ContextItem, ...]
    token_budget: int
    visible_metadata: Mapping[str, Any]
    provenance: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self._validate_version()
        if not isinstance(self.profile, TaskProfile):
            raise ValueError("profile must be a TaskProfile record")
        object.__setattr__(
            self,
            "candidate_context",
            _record_tuple("candidate_context", self.candidate_context, ContextItem),
        )
        _nonempty("task_prompt", self.task_prompt)
        _nonempty("provenance", self.provenance)
        _positive("token_budget", self.token_budget)
        if not self.candidate_context:
            raise ValueError("candidate_context must not be empty")
        ids = tuple(item.context_item_id for item in self.candidate_context)
        if len(set(ids)) != len(ids):
            raise ValueError("candidate context IDs must be unique")
        if not isinstance(self.visible_metadata, Mapping):
            raise ValueError("visible_metadata must be an object")
        object.__setattr__(
            self,
            "visible_metadata",
            _freeze_json(self.visible_metadata, "visible_metadata"),
        )

    @classmethod
    def from_dict(cls, data: Mapping) -> "TaskInputs":
        _version(data)
        values = dict(data)
        values["profile"] = TaskProfile.from_dict(data["profile"])
        values["candidate_context"] = tuple(
            ContextItem.from_dict(item) for item in data["candidate_context"]
        )
        return cls(**values)


@dataclass(frozen=True)
class SealedEvaluation(RecordMixin):
    """Gold evidence available only after selection and task execution."""

    gold_answer: str
    scoring_rubric: ScoringRubric
    required_context_item_ids: Tuple[str, ...]
    useful_context_item_ids: Tuple[str, ...]
    misleading_context_item_ids: Tuple[str, ...]
    irrelevant_context_item_ids: Tuple[str, ...]
    provenance: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self._validate_version()
        if not isinstance(self.scoring_rubric, ScoringRubric):
            raise ValueError("scoring_rubric must be a ScoringRubric record")
        for name in (
            "required_context_item_ids",
            "useful_context_item_ids",
            "misleading_context_item_ids",
            "irrelevant_context_item_ids",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        _nonempty("gold_answer", self.gold_answer)
        _nonempty("provenance", self.provenance)
        groups = (
            self.required_context_item_ids,
            self.useful_context_item_ids,
            self.misleading_context_item_ids,
            self.irrelevant_context_item_ids,
        )
        names = (
            "required_context_item_ids",
            "useful_context_item_ids",
            "misleading_context_item_ids",
            "irrelevant_context_item_ids",
        )
        for name, values in zip(names, groups):
            _unique_nonempty(name, values)
        combined = [item for group in groups for item in group]
        if len(set(combined)) != len(combined):
            raise ValueError("sealed context ID sets must be pairwise disjoint")

    @classmethod
    def from_dict(cls, data: Mapping) -> "SealedEvaluation":
        _version(data)
        values = dict(data)
        values["scoring_rubric"] = ScoringRubric.from_dict(data["scoring_rubric"])
        for key in (
            "required_context_item_ids",
            "useful_context_item_ids",
            "misleading_context_item_ids",
            "irrelevant_context_item_ids",
        ):
            values[key] = _tuple_strings(data, key)
        return cls(**values)


_FORBIDDEN_HELD_OUT_METADATA_KEYS = frozenset(
    {
        "adaptation_feedback",
        "feedback",
        "gold_answer",
        "gold_answer_text",
        "expected_answer",
        "scoring_rubric",
        "required_context_item_ids",
        "useful_context_item_ids",
        "misleading_context_item_ids",
        "irrelevant_context_item_ids",
    }
)


def _normalize_visible_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _normalize_metadata_key(value: str) -> str:
    """Normalize casing and repeated punctuation/whitespace separators."""
    return "_".join(part for part in re.split(r"[\W_]+", value.casefold()) if part)


def _contains_sealed_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        if any(
            _normalize_metadata_key(key) in _FORBIDDEN_HELD_OUT_METADATA_KEYS
            for key in value
        ):
            return True
        return any(_contains_sealed_key(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_sealed_key(item) for item in value)
    return False


def _json_strings(value: Any) -> Tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        return tuple(
            text
            for key, item in value.items()
            for text in _json_strings(key) + _json_strings(item)
        )
    if isinstance(value, (list, tuple)):
        return tuple(text for item in value for text in _json_strings(item))
    return ()


@dataclass(frozen=True)
class TaskCase(RecordMixin):
    task_case_id: str
    split: str
    inputs: TaskInputs
    sealed_evaluation: SealedEvaluation
    dataset_version: str
    created_timestamp: str
    provenance: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self._validate_version()
        if not isinstance(self.inputs, TaskInputs):
            raise ValueError("inputs must be a TaskInputs record")
        if not isinstance(self.sealed_evaluation, SealedEvaluation):
            raise ValueError("sealed_evaluation must be a SealedEvaluation record")
        for name in ("task_case_id", "dataset_version", "provenance"):
            _nonempty(name, getattr(self, name))
        _timestamp("created_timestamp", self.created_timestamp)
        if self.split not in {"adaptation", "held_out"}:
            raise ValueError("split must be adaptation or held_out")
        candidate_ids = {item.context_item_id for item in self.inputs.candidate_context}
        sealed_ids = set(
            self.sealed_evaluation.required_context_item_ids
            + self.sealed_evaluation.useful_context_item_ids
            + self.sealed_evaluation.misleading_context_item_ids
            + self.sealed_evaluation.irrelevant_context_item_ids
        )
        if not sealed_ids.issubset(candidate_ids):
            raise ValueError("sealed context IDs must refer to candidates")
        if self.split == "held_out":
            selector_payload = self.inputs.to_dict()
            if _contains_sealed_key(selector_payload):
                raise ValueError(
                    "held_out selector-visible metadata contains sealed evaluation or adaptation feedback"
                )
            gold = _normalize_visible_text(self.sealed_evaluation.gold_answer)
            visible_strings = _json_strings(selector_payload)
            if any(gold in _normalize_visible_text(text) for text in visible_strings):
                raise ValueError(
                    "held_out selector-visible text contains the full normalized gold answer"
                )

    def selector_inputs(self) -> TaskInputs:
        return self.inputs

    @classmethod
    def from_dict(cls, data: Mapping) -> "TaskCase":
        _version(data)
        values = dict(data)
        values["inputs"] = TaskInputs.from_dict(data["inputs"])
        values["sealed_evaluation"] = SealedEvaluation.from_dict(
            data["sealed_evaluation"]
        )
        return cls(**values)


@dataclass(frozen=True)
class RunManifest(RecordMixin):
    run_id: str
    experiment_version: str
    protocol_version: str
    dataset_version: str
    dataset_hash: str
    selector_mode: str
    selector_version: str
    provider: str
    model_id: str
    prompt_template_hash: str
    config_hash: str
    code_revision: str
    temperature: float
    seed: Optional[int]
    seed_supported: bool
    tool_availability: Tuple[str, ...]
    started_timestamp: str
    provenance: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self._validate_version()
        object.__setattr__(self, "tool_availability", tuple(self.tool_availability))
        for name in (
            "run_id",
            "experiment_version",
            "protocol_version",
            "dataset_version",
            "dataset_hash",
            "selector_mode",
            "selector_version",
            "provider",
            "model_id",
            "prompt_template_hash",
            "config_hash",
            "code_revision",
            "provenance",
        ):
            _nonempty(name, getattr(self, name))
        _timestamp("started_timestamp", self.started_timestamp)
        _nonnegative_finite("temperature", self.temperature)
        if not isinstance(self.seed_supported, bool):
            raise ValueError("seed_supported must be boolean")
        if self.seed is not None and (
            not isinstance(self.seed, int) or isinstance(self.seed, bool)
        ):
            raise ValueError("seed must be an integer or null")
        if not self.seed_supported and self.seed is not None:
            raise ValueError("seed must be null when seed_supported is false")
        if self.seed_supported and self.seed is None:
            raise ValueError("seed is required when seed_supported is true")
        _unique_nonempty("tool_availability", self.tool_availability)

    @classmethod
    def from_dict(cls, data: Mapping) -> "RunManifest":
        _version(data)
        return cls._from_flat_dict(
            data, tool_availability=_tuple_strings(data, "tool_availability")
        )


@dataclass(frozen=True)
class SelectionDecision(RecordMixin):
    decision_id: str
    run_id: str
    task_case_id: str
    selected_context_item_ids: Tuple[str, ...]
    selected_token_counts: Tuple[int, ...]
    total_selected_tokens: int
    token_budget: int
    selector_score: Optional[float]
    selector_input_hash: str
    candidate_set_hash: str
    decision_latency_ms: float
    ranking_artifact_reference: Optional[str]
    ranking_artifact_hash: Optional[str]
    trace_artifact_reference: Optional[str]
    trace_artifact_hash: Optional[str]
    decided_timestamp: str
    provenance: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self._validate_version()
        object.__setattr__(
            self, "selected_context_item_ids", tuple(self.selected_context_item_ids)
        )
        object.__setattr__(
            self, "selected_token_counts", tuple(self.selected_token_counts)
        )
        for name in (
            "decision_id",
            "run_id",
            "task_case_id",
            "selector_input_hash",
            "candidate_set_hash",
            "provenance",
        ):
            _nonempty(name, getattr(self, name))
        _timestamp("decided_timestamp", self.decided_timestamp)
        _positive("token_budget", self.token_budget)
        _unique_nonempty("selected context IDs", self.selected_context_item_ids)
        if len(self.selected_context_item_ids) != len(self.selected_token_counts):
            raise ValueError("selected IDs and token counts must align")
        if any(
            not isinstance(count, int) or isinstance(count, bool) or count <= 0
            for count in self.selected_token_counts
        ):
            raise ValueError("selected token counts must be positive integers")
        _nonnegative_integer("total_selected_tokens", self.total_selected_tokens)
        if self.selected_context_item_ids and self.total_selected_tokens == 0:
            raise ValueError(
                "total_selected_tokens may be zero only when no context items are selected"
            )
        if self.total_selected_tokens != sum(self.selected_token_counts):
            raise ValueError("total_selected_tokens must equal selected token counts")
        if self.total_selected_tokens > self.token_budget:
            raise ValueError("total_selected_tokens must not exceed token_budget")
        if self.selector_score is not None:
            _normalized("selector_score", self.selector_score)
        _nonnegative_finite("decision_latency_ms", self.decision_latency_ms)
        _paired(
            "ranking_artifact_reference",
            self.ranking_artifact_reference,
            "ranking_artifact_hash",
            self.ranking_artifact_hash,
        )
        _paired(
            "trace_artifact_reference",
            self.trace_artifact_reference,
            "trace_artifact_hash",
            self.trace_artifact_hash,
        )

    @classmethod
    def from_dict(cls, data: Mapping) -> "SelectionDecision":
        _version(data)
        return cls._from_flat_dict(
            data,
            selected_context_item_ids=_tuple_strings(data, "selected_context_item_ids"),
            selected_token_counts=tuple(data.get("selected_token_counts", ())),
        )


@dataclass(frozen=True)
class CriterionScore(RecordMixin):
    """One bounded rubric-criterion result with reproducible normalization."""

    criterion_id: str
    raw_score: float
    max_score: float
    normalized_score: float
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self._validate_version()
        _nonempty("criterion_id", self.criterion_id)
        raw = _finite_number("raw_score", self.raw_score)
        maximum = _finite_number("max_score", self.max_score)
        _normalized("normalized_score", self.normalized_score)
        if maximum <= 0:
            raise ValueError("max_score must be positive")
        if not 0 <= raw <= maximum:
            raise ValueError("raw_score must be between 0 and max_score")
        if not math.isclose(
            float(self.normalized_score), raw / maximum, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError("normalized_score must equal raw_score / max_score")

    @classmethod
    def from_dict(cls, data: Mapping) -> "CriterionScore":
        return cls._from_flat_dict(data)


@dataclass(frozen=True)
class TaskOutcome(RecordMixin):
    """Completed execution-and-evaluation evidence for one task.

    Successful records require bounded aggregate and criterion scores plus paired
    evaluation and raw-provider-response artifacts. ``provider_response_hash`` covers
    the referenced raw provider response artifact, not ``response_text``. The evaluation
    artifact captures rubric inputs and detailed weighting. Aggregate arithmetic cannot
    be recomputed from criterion scores alone without that artifact, so the wire format
    preserves the aggregation method/version/artifact rather than assuming equal weights.
    Failed executions are unscored; paired artifacts remain optional evidence.
    """

    outcome_id: str
    run_id: str
    task_case_id: str
    selection_decision_id: str
    response_text: str
    execution_status: str
    raw_score: Optional[float]
    max_score: Optional[float]
    normalized_score: Optional[float]
    rubric_id: str
    scorer_id: str
    scorer_version: str
    scorer_hash: str
    aggregation_method: str
    aggregation_version: str
    criterion_scores: Tuple[CriterionScore, ...]
    evaluation_artifact_reference: Optional[str]
    evaluation_artifact_hash: Optional[str]
    model_input_tokens: int
    model_output_tokens: int
    execution_latency_ms: float
    provider_response_artifact_reference: Optional[str]
    provider_response_hash: Optional[str]
    error_category: Optional[str]
    completed_timestamp: str
    provenance: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self._validate_version()
        object.__setattr__(
            self,
            "criterion_scores",
            _record_tuple("criterion_scores", self.criterion_scores, CriterionScore),
        )
        for name in (
            "outcome_id",
            "run_id",
            "task_case_id",
            "selection_decision_id",
            "rubric_id",
            "scorer_id",
            "scorer_version",
            "scorer_hash",
            "aggregation_method",
            "aggregation_version",
            "provenance",
        ):
            _nonempty(name, getattr(self, name))
        _timestamp("completed_timestamp", self.completed_timestamp)
        if self.execution_status not in {"success", "failure"}:
            raise ValueError("execution_status must be success or failure")

        score_values = (self.raw_score, self.max_score, self.normalized_score)
        if self.execution_status == "success":
            _nonempty("response_text", self.response_text)
            if self.error_category is not None:
                raise ValueError("error_category must be null for success")
            if any(value is None for value in score_values) and not all(
                value is None for value in score_values
            ):
                raise ValueError(
                    "raw_score, max_score, and normalized_score must be provided together"
                )
            if all(value is None for value in score_values):
                raise ValueError("aggregate scores are required for success")
        else:
            if self.error_category is None:
                raise ValueError("error_category is required for failure")
            _nonempty("error_category", self.error_category)
            if any(value is not None for value in score_values):
                raise ValueError("aggregate scores must be null for failure")
            if self.criterion_scores:
                raise ValueError("criterion_scores must be empty for failure")

        if all(value is not None for value in score_values):
            raw = _finite_number("raw_score", self.raw_score)
            maximum = _finite_number("max_score", self.max_score)
            _normalized("normalized_score", self.normalized_score)
            if maximum <= 0:
                raise ValueError("max_score must be positive")
            if not 0 <= raw <= maximum:
                raise ValueError("raw_score must be between 0 and max_score")
            if not math.isclose(
                cast(float, self.normalized_score),
                raw / maximum,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                raise ValueError("normalized_score must equal raw_score / max_score")

        ids = tuple(item.criterion_id for item in self.criterion_scores)
        if len(set(ids)) != len(ids):
            raise ValueError("criterion IDs must be unique")
        _paired(
            "evaluation_artifact_reference",
            self.evaluation_artifact_reference,
            "evaluation_artifact_hash",
            self.evaluation_artifact_hash,
        )
        _paired(
            "provider_response_artifact_reference",
            self.provider_response_artifact_reference,
            "provider_response_hash",
            self.provider_response_hash,
        )
        if self.execution_status == "success":
            if not self.criterion_scores:
                raise ValueError("criterion_scores are required for success")
            if self.evaluation_artifact_reference is None:
                raise ValueError("evaluation artifact is required for success")
            if self.provider_response_artifact_reference is None:
                raise ValueError("provider response artifact is required for success")
        _nonnegative_integer("model_input_tokens", self.model_input_tokens)
        _nonnegative_integer("model_output_tokens", self.model_output_tokens)
        _nonnegative_finite("execution_latency_ms", self.execution_latency_ms)

    @classmethod
    def from_dict(cls, data: Mapping) -> "TaskOutcome":
        _version(data)
        values = dict(data)
        values["criterion_scores"] = tuple(
            CriterionScore.from_dict(item) for item in data.get("criterion_scores", ())
        )
        return cls(**values)


FEEDBACK_SOURCES = frozenset({"oracle", "simulated", "human", "judge"})
FEEDBACK_SIGNAL_TYPES = frozenset(
    {"context_utility", "task_score", "selection_quality", "correction", "preference"}
)


@dataclass(frozen=True)
class FeedbackEvent(RecordMixin):
    """Feedback with one unambiguous encoding.

    Numeric context_utility, preference, and correction values are signed in [-1, 1];
    task_score and selection_quality are normalized in [0, 1]. A non-null canonical
    JSON structured_value may be used instead of, never alongside, numeric_value.
    Correction signals additionally require a category and replacement/explanatory
    correction text; those fields are forbidden for every other signal.
    """

    event_id: str
    run_id: str
    task_case_id: str
    task_family_id: str
    signal_type: str
    numeric_value: Optional[float]
    structured_value: Any
    affected_context_item_ids: Tuple[str, ...]
    correction_category: Optional[str]
    correction_text: Optional[str]
    source: str
    occurred_timestamp: str
    provenance: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self._validate_version()
        object.__setattr__(
            self, "affected_context_item_ids", tuple(self.affected_context_item_ids)
        )
        for name in (
            "event_id",
            "run_id",
            "task_case_id",
            "task_family_id",
            "provenance",
        ):
            _nonempty(name, getattr(self, name))
        _timestamp("occurred_timestamp", self.occurred_timestamp)
        if self.signal_type not in FEEDBACK_SIGNAL_TYPES:
            raise ValueError(
                f"signal_type must be one of {sorted(FEEDBACK_SIGNAL_TYPES)}"
            )
        if self.source not in FEEDBACK_SOURCES:
            raise ValueError(f"source must be one of {sorted(FEEDBACK_SOURCES)}")
        if (self.numeric_value is None) == (self.structured_value is None):
            raise ValueError(
                "feedback requires exactly one of numeric_value or structured_value"
            )
        if self.numeric_value is not None:
            if self.signal_type in {"task_score", "selection_quality"}:
                _range("numeric_value", self.numeric_value, 0, 1)
            else:
                _range("numeric_value", self.numeric_value, -1, 1)
        else:
            object.__setattr__(
                self,
                "structured_value",
                _freeze_json(self.structured_value, "structured_value"),
            )
        if self.signal_type == "correction":
            for name in ("correction_category", "correction_text"):
                if getattr(self, name) is None:
                    raise ValueError(f"correction requires {name}")
                _nonempty(name, getattr(self, name))
        elif self.correction_category is not None or self.correction_text is not None:
            raise ValueError(
                "correction-only fields must be null for non-correction signals"
            )
        _unique_nonempty("affected_context_item_ids", self.affected_context_item_ids)

    @classmethod
    def from_dict(cls, data: Mapping) -> "FeedbackEvent":
        _version(data)
        return cls._from_flat_dict(
            data,
            affected_context_item_ids=_tuple_strings(data, "affected_context_item_ids"),
        )


@dataclass(frozen=True)
class UtilityEstimate(RecordMixin):
    """A signed utility estimate over context attributes, derived from feedback."""

    utility_estimate_id: str
    task_family_id: str
    context_attributes: Tuple[str, ...]
    estimated_utility: float
    confidence: float
    source_event_ids: Tuple[str, ...]
    estimator_version: str
    estimated_timestamp: str
    provenance: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self._validate_version()
        object.__setattr__(self, "context_attributes", tuple(self.context_attributes))
        object.__setattr__(self, "source_event_ids", tuple(self.source_event_ids))
        for name in (
            "utility_estimate_id",
            "task_family_id",
            "estimator_version",
            "provenance",
        ):
            _nonempty(name, getattr(self, name))
        _timestamp("estimated_timestamp", self.estimated_timestamp)
        _range("estimated_utility", self.estimated_utility, -1, 1)
        _normalized("confidence", self.confidence)
        _unique_nonempty("source_event_ids", self.source_event_ids)
        if not self.source_event_ids:
            raise ValueError("source_event_ids must not be empty")
        _unique_nonempty("context_attributes", self.context_attributes)
        if not self.context_attributes:
            raise ValueError("context_attributes must not be empty")

    @classmethod
    def from_dict(cls, data: Mapping) -> "UtilityEstimate":
        _version(data)
        return cls._from_flat_dict(
            data,
            context_attributes=_tuple_strings(data, "context_attributes"),
            source_event_ids=_tuple_strings(data, "source_event_ids"),
        )


@dataclass(frozen=True)
class ExperimentResult(RecordMixin):
    """Bounded index of artifacts produced by one run."""

    experiment_result_id: str
    run_id: str
    outcome_ids: Tuple[str, ...]
    selection_decision_ids: Tuple[str, ...]
    feedback_event_ids: Tuple[str, ...]
    utility_estimate_ids: Tuple[str, ...]
    completed_timestamp: str
    provenance: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self._validate_version()
        for name in (
            "outcome_ids",
            "selection_decision_ids",
            "feedback_event_ids",
            "utility_estimate_ids",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in ("experiment_result_id", "run_id", "provenance"):
            _nonempty(name, getattr(self, name))
        _timestamp("completed_timestamp", self.completed_timestamp)
        for name in (
            "outcome_ids",
            "selection_decision_ids",
            "feedback_event_ids",
            "utility_estimate_ids",
        ):
            _unique_nonempty(name, getattr(self, name))

    @classmethod
    def from_dict(cls, data: Mapping) -> "ExperimentResult":
        _version(data)
        changes = {
            key: _tuple_strings(data, key)
            for key in (
                "outcome_ids",
                "selection_decision_ids",
                "feedback_event_ids",
                "utility_estimate_ids",
            )
        }
        return cls._from_flat_dict(data, **changes)
