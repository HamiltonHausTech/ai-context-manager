"""Versioned records for adaptive context-selection experiments.

These records intentionally contain no persistence, provider, selection, or learning
logic.  Selector-visible task inputs and sealed evaluation evidence are separate
objects so held-out labels cannot be passed to a selector accidentally.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple, Type, TypeVar, cast

SCHEMA_VERSION = "1"

T = TypeVar("T", bound="RecordMixin")


def _nonempty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty")


def _normalized(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be between 0 and 1")
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def _positive(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be positive")


def _unique_nonempty(name: str, values: Tuple[str, ...]) -> None:
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{name} must not contain empty IDs")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must be unique")


def _version(data: Dict[str, Any]) -> str:
    if "schema_version" not in data:
        raise ValueError("schema_version is required")
    version = data["schema_version"]
    if version != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {version}")
    return version


def _json_compatible(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_compatible(item) for item in value]
    if isinstance(value, list):
        return [_json_compatible(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_compatible(item) for key, item in value.items()}
    return value


def _tuple_strings(data: Dict[str, Any], key: str) -> Tuple[str, ...]:
    return tuple(data.get(key, ()))


class RecordMixin:
    """Common deterministic serialization for immutable experiment records."""

    schema_version: str

    def _validate_version(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")

    def to_dict(self) -> Dict[str, Any]:
        return _json_compatible(asdict(cast(Any, self)))

    @classmethod
    def _from_flat_dict(cls: Type[T], data: Dict[str, Any], **changes: Any) -> T:
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
        for name in (
            "task_family_id",
            "name",
            "description",
            "created_timestamp",
            "provenance",
        ):
            _nonempty(name, getattr(self, name))
        _positive("default_token_budget", self.default_token_budget)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskProfile":
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
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self._validate_version()
        for name in (
            "context_item_id",
            "content",
            "source",
            "created_timestamp",
            "provenance",
        ):
            _nonempty(name, getattr(self, name))
        _positive("token_count", self.token_count)
        _normalized("confidence", self.confidence)
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be an object")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextItem":
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
            or self.weight <= 0
        ):
            raise ValueError("weight must be positive")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RubricCriterion":
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
        _nonempty("rubric_id", self.rubric_id)
        _nonempty("instructions", self.instructions)
        _nonempty("provenance", self.provenance)
        if not self.criteria:
            raise ValueError("criteria must not be empty")
        ids = tuple(item.criterion_id for item in self.criteria)
        if len(set(ids)) != len(ids):
            raise ValueError("criterion IDs must be unique")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScoringRubric":
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
    visible_metadata: Dict[str, Any]
    provenance: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self._validate_version()
        _nonempty("task_prompt", self.task_prompt)
        _nonempty("provenance", self.provenance)
        _positive("token_budget", self.token_budget)
        if not self.candidate_context:
            raise ValueError("candidate_context must not be empty")
        ids = tuple(item.context_item_id for item in self.candidate_context)
        if len(set(ids)) != len(ids):
            raise ValueError("candidate context IDs must be unique")
        if not isinstance(self.visible_metadata, dict):
            raise ValueError("visible_metadata must be an object")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskInputs":
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
        _nonempty("gold_answer", self.gold_answer)
        _nonempty("provenance", self.provenance)
        groups = (
            self.required_context_item_ids,
            self.useful_context_item_ids,
            self.misleading_context_item_ids,
            self.irrelevant_context_item_ids,
        )
        for name, values in zip(
            (
                "required_context_item_ids",
                "useful_context_item_ids",
                "misleading_context_item_ids",
                "irrelevant_context_item_ids",
            ),
            groups,
        ):
            _unique_nonempty(name, values)
        combined = [item for group in groups for item in group]
        if len(set(combined)) != len(combined):
            raise ValueError("sealed context ID sets must be pairwise disjoint")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SealedEvaluation":
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


_FORBIDDEN_HELD_OUT_METADATA_KEYS = {
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


def _metadata_contains_sealed_key(value: Any) -> bool:
    if isinstance(value, dict):
        if any(str(key).lower() in _FORBIDDEN_HELD_OUT_METADATA_KEYS for key in value):
            return True
        return any(_metadata_contains_sealed_key(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_metadata_contains_sealed_key(item) for item in value)
    return False


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
        for name in (
            "task_case_id",
            "dataset_version",
            "created_timestamp",
            "provenance",
        ):
            _nonempty(name, getattr(self, name))
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
        if self.split == "held_out" and _metadata_contains_sealed_key(
            self.inputs.to_dict()
        ):
            raise ValueError(
                "held_out selector-visible metadata contains sealed evaluation or adaptation feedback"
            )

    def selector_inputs(self) -> TaskInputs:
        return self.inputs

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskCase":
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
            "started_timestamp",
            "provenance",
        ):
            _nonempty(name, getattr(self, name))
        if (
            not isinstance(self.temperature, (int, float))
            or isinstance(self.temperature, bool)
            or self.temperature < 0
        ):
            raise ValueError("temperature must be nonnegative")
        if self.seed is not None and (
            not isinstance(self.seed, int) or isinstance(self.seed, bool)
        ):
            raise ValueError("seed must be an integer or null")
        if not self.seed_supported and self.seed is not None:
            raise ValueError("seed must be null when seed_supported is false")
        _unique_nonempty("tool_availability", self.tool_availability)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunManifest":
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
    decided_timestamp: str
    provenance: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self._validate_version()
        for name in (
            "decision_id",
            "run_id",
            "task_case_id",
            "decided_timestamp",
            "provenance",
        ):
            _nonempty(name, getattr(self, name))
        _positive("token_budget", self.token_budget)
        if any(
            not isinstance(item, str) or not item.strip()
            for item in self.selected_context_item_ids
        ):
            raise ValueError("selected context IDs must be nonempty")
        if len(set(self.selected_context_item_ids)) != len(
            self.selected_context_item_ids
        ):
            raise ValueError("selected context IDs must be unique")
        if len(self.selected_context_item_ids) != len(self.selected_token_counts):
            raise ValueError("selected IDs and token counts must align")
        if any(
            not isinstance(count, int) or isinstance(count, bool) or count <= 0
            for count in self.selected_token_counts
        ):
            raise ValueError("selected token counts must be positive integers")
        if self.total_selected_tokens != sum(self.selected_token_counts):
            raise ValueError("total_selected_tokens must equal selected token counts")
        if self.total_selected_tokens > self.token_budget:
            raise ValueError("total_selected_tokens must not exceed token_budget")
        if self.selector_score is not None:
            _normalized("selector_score", self.selector_score)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SelectionDecision":
        return cls._from_flat_dict(
            data,
            selected_context_item_ids=_tuple_strings(data, "selected_context_item_ids"),
            selected_token_counts=tuple(data.get("selected_token_counts", ())),
        )


@dataclass(frozen=True)
class TaskOutcome(RecordMixin):
    outcome_id: str
    run_id: str
    task_case_id: str
    selection_decision_id: str
    response_text: str
    normalized_score: float
    completed_timestamp: str
    provenance: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self._validate_version()
        for name in (
            "outcome_id",
            "run_id",
            "task_case_id",
            "selection_decision_id",
            "response_text",
            "completed_timestamp",
            "provenance",
        ):
            _nonempty(name, getattr(self, name))
        _normalized("normalized_score", self.normalized_score)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskOutcome":
        return cls._from_flat_dict(data)


FEEDBACK_SOURCES = frozenset({"oracle", "simulated", "human", "judge"})
FEEDBACK_SIGNAL_TYPES = frozenset(
    {"context_utility", "task_score", "selection_quality", "correction", "preference"}
)


@dataclass(frozen=True)
class FeedbackEvent(RecordMixin):
    event_id: str
    run_id: str
    task_case_id: str
    task_family_id: str
    signal_type: str
    numeric_value: Optional[float]
    structured_category: Optional[str]
    affected_context_item_ids: Tuple[str, ...]
    correction_category: Optional[str]
    correction_text: Optional[str]
    source: str
    occurred_timestamp: str
    provenance: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self._validate_version()
        for name in (
            "event_id",
            "run_id",
            "task_case_id",
            "task_family_id",
            "occurred_timestamp",
            "provenance",
        ):
            _nonempty(name, getattr(self, name))
        if self.signal_type not in FEEDBACK_SIGNAL_TYPES:
            raise ValueError(
                f"signal_type must be one of {sorted(FEEDBACK_SIGNAL_TYPES)}"
            )
        if self.source not in FEEDBACK_SOURCES:
            raise ValueError(f"source must be one of {sorted(FEEDBACK_SOURCES)}")
        if self.numeric_value is None and self.structured_category is None:
            raise ValueError("feedback requires numeric_value or structured_category")
        if self.numeric_value is not None:
            _normalized("numeric_value", self.numeric_value)
        for name in ("structured_category", "correction_category", "correction_text"):
            value = getattr(self, name)
            if value is not None:
                _nonempty(name, value)
        _unique_nonempty("affected_context_item_ids", self.affected_context_item_ids)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeedbackEvent":
        return cls._from_flat_dict(
            data,
            affected_context_item_ids=_tuple_strings(data, "affected_context_item_ids"),
        )


@dataclass(frozen=True)
class UtilityEstimate(RecordMixin):
    """An estimate over context attributes, derived from raw feedback events."""

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
        for name in (
            "utility_estimate_id",
            "task_family_id",
            "estimator_version",
            "estimated_timestamp",
            "provenance",
        ):
            _nonempty(name, getattr(self, name))
        _normalized("estimated_utility", self.estimated_utility)
        _normalized("confidence", self.confidence)
        _unique_nonempty("source_event_ids", self.source_event_ids)
        if not self.source_event_ids:
            raise ValueError("source_event_ids must not be empty")
        _unique_nonempty("context_attributes", self.context_attributes)
        if not self.context_attributes:
            raise ValueError("context_attributes must not be empty")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UtilityEstimate":
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
            "experiment_result_id",
            "run_id",
            "completed_timestamp",
            "provenance",
        ):
            _nonempty(name, getattr(self, name))
        for name in (
            "outcome_ids",
            "selection_decision_ids",
            "feedback_event_ids",
            "utility_estimate_ids",
        ):
            _unique_nonempty(name, getattr(self, name))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentResult":
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
