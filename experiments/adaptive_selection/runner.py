"""Deterministic, in-memory orchestration for ordered adaptive-selection experiments.

This module enforces temporal API discipline, condition-blind prompting/assessment, and
artifact-atomic execution.  It does not provide physical secrecy, persistence, retries,
reporting, statistics, a CLI, or a network provider.
"""

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Dict, NoReturn, Optional, Protocol, Tuple, Type, TypeVar, cast

from .dataset import DatasetBundle, canonical_bundle_sha256
from .learning import LearningPolicy, LearningSnapshot, learn_utilities
from .providers import (
    ManifestInputs,
    Provider,
    ProviderConfiguration,
    ProviderExecution,
    ProviderRequest,
    build_run_manifest,
    compare_manifests,
    validate_execution,
    validate_request_manifest,
)
from .schema import (
    ContextItem,
    CriterionScore,
    FeedbackEvent,
    RunManifest,
    ScoringRubric,
    SealedEvaluation,
    SelectionDecision,
    TaskCase,
    TaskInputs,
    TaskOutcome,
)
from .scoring import BlindedAssessment, ScoringResult, TaskScoringSpec, score_assessment
from .selectors import SelectionResult, SelectorDecision

RUNNER_RECORD_VERSION = "1"
ARTIFACT_VERSION = "ordered-experiment-v1"
SELECTOR_MODES = (
    "full_context",
    "similarity_top_k",
    "static_policy",
    "adaptive_policy",
)
ARM_CLASSIFICATIONS = frozenset(
    {"reference", "secondary", "primary_baseline", "candidate"}
)
_MAX_ARMS = 16
_MAX_REPETITIONS = 32
_MAX_FAMILIES = 64
_MAX_CASES = 10_000
_MAX_SLOTS = 100_000
_MAX_TRACE = 1_000_000
_MAX_TEXT = 16_384
_MAX_PROVENANCE = 4_096
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
_SHA_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_OPAQUE_FORBIDDEN = (
    "full_context",
    "similarity_top_k",
    "static_policy",
    "adaptive_policy",
    "adaptation",
    "evaluation",
    "reference",
    "baseline",
    "candidate",
    "phase",
    "run-",
)
T = TypeVar("T", bound="CanonicalRunnerRecord")


class RunnerError(RuntimeError):
    """Sanitized, fail-fast runner failure."""

    def __init__(self, category: str, stage: str) -> None:
        self.category = _short("category", category)
        self.stage = _short("stage", stage)
        super().__init__("ordered experiment failed: {} at {}".format(category, stage))


class RunnerValidationError(RunnerError, ValueError):
    """A plan, runtime, source, or derived artifact violated an invariant."""


def _fail(message: str) -> NoReturn:
    raise ValueError(message)


def _short(name: str, value: Any, limit: int = 256) -> str:
    if type(value) is not str or not value.strip() or len(value) > limit:
        _fail("{} must be a bounded nonempty string".format(name))
    return value


def _sha(name: str, value: Any) -> str:
    if type(value) is not str or not _SHA_RE.fullmatch(value):
        _fail("{} must be lowercase sha256".format(name))
    return value


def _int64(name: str, value: Any) -> int:
    if type(value) is not int or not _INT64_MIN <= value <= _INT64_MAX:
        _fail("{} must be a signed 64-bit integer".format(name))
    return value


def _utc_timestamp(name: str, value: Any) -> str:
    if type(value) is not str or not value.endswith("Z"):
        _fail("{} must be a canonical UTC timestamp".format(name))
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        _fail("{} must be a canonical UTC timestamp".format(name))
    if (
        parsed.tzinfo != timezone.utc
        or parsed.isoformat().replace("+00:00", "Z") != value
    ):
        _fail("{} must be a canonical UTC timestamp".format(name))
    return value


def _timestamp_value(value: str) -> datetime:
    _utc_timestamp("timestamp", value)
    return datetime.fromisoformat(value[:-1] + "+00:00")


def _index(name: str, value: Any) -> int:
    if type(value) is not int or value < 0:
        _fail("{} must be a nonnegative integer".format(name))
    return value


def _finite(name: str, value: Any, nonnegative: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        _fail("{} must be finite".format(name))
    result = float(value)
    if nonnegative and result < 0:
        _fail("{} must be nonnegative".format(name))
    return result


def _tuple(name: str, value: Any, limit: int, item_type: Type[Any]) -> Tuple[Any, ...]:
    if type(value) not in (list, tuple):
        _fail("{} must be an exact sequence".format(name))
    copied = tuple(value)
    if len(copied) > limit or any(type(item) is not item_type for item in copied):
        _fail("{} contains invalid or excessive records".format(name))
    return copied


def _strings(name: str, value: Any, limit: int) -> Tuple[str, ...]:
    copied = cast(Tuple[str, ...], _tuple(name, value, limit, str))
    for item in copied:
        _short(name, item)
    if len(set(copied)) != len(copied):
        _fail("{} must be unique".format(name))
    return copied


def _serialize(value: Any) -> Any:
    if type(value) is LearningPolicy:
        return _serialize(value.to_dict())
    if not isinstance(value, CanonicalRunnerRecord) and callable(
        getattr(value, "to_dict", None)
    ):
        return _serialize(value.to_dict())
    if type(value) is bytes:
        import base64

        return {
            "data": base64.b64encode(value).decode("ascii"),
            "encoding": "base64",
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _serialize(getattr(value, item.name)) for item in fields(value)
        }
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            _fail("mapping keys must be strings")
        return {key: _serialize(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_serialize(item) for item in value]
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            _fail("numbers must be finite")
        return value
    _fail("unsupported canonical value: {}".format(type(value).__name__))


def _canonical(value: Any) -> bytes:
    return json.dumps(
        _serialize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _domain_hash(domain: str, value: Any) -> str:
    payload = domain.encode("ascii") + b"\x00" + _canonical(value)
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _freeze_json(value: Any, depth: int = 0, counter: Optional[list] = None) -> Any:
    if counter is None:
        counter = [0]
    counter[0] += 1
    if depth > 32 or counter[0] > 100_000:
        _fail("canonical JSON exceeds resource bounds")
    if value is None or type(value) in (bool, int, str):
        if type(value) is int:
            _int64("JSON integer", value)
        if type(value) is str and len(value) > _MAX_TEXT:
            _fail("JSON string is too long")
        return value
    if type(value) is float:
        return _finite("JSON number", value)
    if type(value) is dict or isinstance(value, MappingProxyType):
        if any(type(key) is not str for key in value):
            _fail("JSON object keys must be strings")
        return MappingProxyType(
            {key: _freeze_json(value[key], depth + 1, counter) for key in sorted(value)}
        )
    if type(value) in (list, tuple):
        return tuple(_freeze_json(item, depth + 1, counter) for item in value)
    _fail("JSON must use exact built-in types")


def _exact(data: Any, expected: Sequence[str]) -> Dict[str, Any]:
    if type(data) is not dict or set(data) != set(expected):
        _fail("payload must contain exactly: {}".format(", ".join(expected)))
    return dict(data)


def _policy_from_dict(data: Any) -> LearningPolicy:
    values = _exact(
        data,
        (
            "accepted_signal_types",
            "credit_assignment_version",
            "estimator_version",
            "id_local_enabled",
            "minimum_evidence_count",
            "prior_mean",
            "prior_strength",
        ),
    )
    values["accepted_signal_types"] = frozenset(
        _strings("accepted_signal_types", values["accepted_signal_types"], 16)
    )
    return LearningPolicy(**values)


class CanonicalRunnerRecord:
    def to_dict(self) -> Dict[str, Any]:
        return cast(Dict[str, Any], _serialize(self))

    def canonical_bytes(self) -> bytes:
        return _canonical(self)


@dataclass(frozen=True)
class ArmSpec(CanonicalRunnerRecord):
    arm_id: str
    selector_mode: str
    selector_version: str
    selector_config_hash: str
    uses_feature_learning: bool
    classification: str
    arm_hash: str = field(init=False)

    def __post_init__(self) -> None:
        _short("arm_id", self.arm_id)
        _short("selector_version", self.selector_version)
        if self.selector_mode not in SELECTOR_MODES:
            _fail("selector_mode is invalid")
        _sha("selector_config_hash", self.selector_config_hash)
        if type(self.uses_feature_learning) is not bool:
            _fail("uses_feature_learning must be boolean")
        if self.uses_feature_learning != (self.selector_mode == "adaptive_policy"):
            _fail("only adaptive_policy uses feature learning")
        if self.classification not in ARM_CLASSIFICATIONS:
            _fail("classification is invalid")
        object.__setattr__(
            self,
            "arm_hash",
            _domain_hash(
                "ordered-arm-spec-v1",
                {
                    "arm_id": self.arm_id,
                    "classification": self.classification,
                    "selector_config_hash": self.selector_config_hash,
                    "selector_mode": self.selector_mode,
                    "selector_version": self.selector_version,
                    "uses_feature_learning": self.uses_feature_learning,
                },
            ),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArmSpec":
        values = _exact(data, tuple(item.name for item in fields(cls)))
        expected = values.pop("arm_hash")
        result = cls(**values)
        if result.arm_hash != expected:
            _fail("arm_hash does not match payload")
        return result


@dataclass(frozen=True)
class RepetitionSpec(CanonicalRunnerRecord):
    repetition_index: int
    provider_seed: int
    block_id: Optional[str]
    repetition_hash: str = field(init=False)

    def __post_init__(self) -> None:
        _index("repetition_index", self.repetition_index)
        _int64("provider_seed", self.provider_seed)
        if self.block_id is not None:
            _short("block_id", self.block_id)
        object.__setattr__(
            self,
            "repetition_hash",
            _domain_hash(
                "ordered-repetition-v1",
                {
                    "block_id": self.block_id,
                    "provider_seed": self.provider_seed,
                    "repetition_index": self.repetition_index,
                },
            ),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RepetitionSpec":
        values = _exact(data, tuple(item.name for item in fields(cls)))
        expected = values.pop("repetition_hash")
        result = cls(**values)
        if result.repetition_hash != expected:
            _fail("repetition_hash does not match payload")
        return result


@dataclass(frozen=True)
class PromptTemplateSpec(CanonicalRunnerRecord):
    system_prompt: str
    format_version: str
    template_hash: str = field(init=False)

    def __post_init__(self) -> None:
        _short("system_prompt", self.system_prompt, _MAX_TEXT)
        _short("format_version", self.format_version)
        object.__setattr__(
            self,
            "template_hash",
            _domain_hash(
                "adaptive-selection-prompt-template-v1",
                {"format": self.format_version, "system": self.system_prompt},
            ),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PromptTemplateSpec":
        values = _exact(data, tuple(item.name for item in fields(cls)))
        expected = values.pop("template_hash")
        result = cls(**values)
        if result.template_hash != expected:
            _fail("template_hash does not match prompt template")
        return result


@dataclass(frozen=True)
class ExperimentPlan(CanonicalRunnerRecord):
    runner_version: str
    experiment_version: str
    protocol_version: str
    dataset_version: str
    dataset_hash: str
    code_revision: str
    prompt_template: PromptTemplateSpec
    prompt_template_hash: str
    learning_policy: LearningPolicy
    schedule_seed: int
    family_order: Tuple[str, ...]
    arms: Tuple[ArmSpec, ...]
    repetitions: Tuple[RepetitionSpec, ...]
    provenance: str
    plan_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "runner_version",
            "experiment_version",
            "protocol_version",
            "dataset_version",
            "code_revision",
        ):
            _short(name, getattr(self, name))
        _sha("dataset_hash", self.dataset_hash)
        if type(self.prompt_template) is not PromptTemplateSpec:
            _fail("prompt_template must be exact PromptTemplateSpec")
        template = PromptTemplateSpec.from_dict(self.prompt_template.to_dict())
        _sha("prompt_template_hash", self.prompt_template_hash)
        if self.prompt_template_hash != template.template_hash:
            _fail("prompt_template_hash must match complete prompt template")
        if type(self.learning_policy) is not LearningPolicy:
            _fail("learning_policy must be exact LearningPolicy")
        _int64("schedule_seed", self.schedule_seed)
        families = _strings("family_order", self.family_order, _MAX_FAMILIES)
        if not families:
            _fail("family_order must not be empty")
        arms = cast(Tuple[ArmSpec, ...], _tuple("arms", self.arms, _MAX_ARMS, ArmSpec))
        reps = cast(
            Tuple[RepetitionSpec, ...],
            _tuple("repetitions", self.repetitions, _MAX_REPETITIONS, RepetitionSpec),
        )
        if not arms or not reps:
            _fail("arms and repetitions must not be empty")
        if tuple(item.selector_mode for item in arms) != SELECTOR_MODES:
            _fail("Task 9 requires the exact four ordered selector modes")
        if len({item.arm_id for item in arms}) != len(arms):
            _fail("arm IDs must be unique")
        if tuple(item.repetition_index for item in reps) != tuple(range(len(reps))):
            _fail("repetition indexes must be contiguous from zero")
        _short("provenance", self.provenance, _MAX_PROVENANCE)
        object.__setattr__(self, "family_order", families)
        object.__setattr__(self, "arms", arms)
        object.__setattr__(self, "repetitions", reps)
        object.__setattr__(self, "prompt_template", template)
        object.__setattr__(
            self,
            "plan_hash",
            _domain_hash(
                "ordered-experiment-plan-v1",
                {
                    "arms": arms,
                    "code_revision": self.code_revision,
                    "dataset_hash": self.dataset_hash,
                    "dataset_version": self.dataset_version,
                    "experiment_version": self.experiment_version,
                    "family_order": families,
                    "learning_policy": self.learning_policy.to_dict(),
                    "prompt_template": template,
                    "prompt_template_hash": self.prompt_template_hash,
                    "protocol_version": self.protocol_version,
                    "provenance": self.provenance,
                    "repetitions": reps,
                    "runner_version": self.runner_version,
                    "schedule_seed": self.schedule_seed,
                },
            ),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentPlan":
        values = _exact(data, tuple(item.name for item in fields(cls)))
        expected = values.pop("plan_hash")
        values["learning_policy"] = _policy_from_dict(values["learning_policy"])
        values["prompt_template"] = PromptTemplateSpec.from_dict(
            values["prompt_template"]
        )
        values["family_order"] = tuple(values["family_order"])
        values["arms"] = tuple(ArmSpec.from_dict(item) for item in values["arms"])
        values["repetitions"] = tuple(
            RepetitionSpec.from_dict(item) for item in values["repetitions"]
        )
        result = cls(**values)
        if result.plan_hash != expected:
            _fail("plan_hash does not match payload")
        return result


@dataclass(frozen=True)
class CaseExecutionMaterial(CanonicalRunnerRecord):
    task_case: TaskCase
    scoring_spec: TaskScoringSpec

    def __post_init__(self) -> None:
        if (
            type(self.task_case) is not TaskCase
            or type(self.scoring_spec) is not TaskScoringSpec
        ):
            _fail("case material records must be exact")
        canonical_case = TaskCase.from_dict(self.task_case.to_dict())
        canonical_spec = TaskScoringSpec.from_dict(self.scoring_spec.to_dict())
        rubric = canonical_case.sealed_evaluation.scoring_rubric
        if canonical_spec.rubric_id != rubric.rubric_id:
            _fail("scoring spec rubric_id must match case rubric")
        if canonical_spec.expected_criterion_ids != tuple(
            item.criterion_id for item in rubric.criteria
        ):
            _fail("scoring spec criteria must match case rubric")
        object.__setattr__(self, "task_case", canonical_case)
        object.__setattr__(self, "scoring_spec", canonical_spec)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CaseExecutionMaterial":
        values = _exact(data, ("task_case", "scoring_spec"))
        return cls(
            TaskCase.from_dict(values["task_case"]),
            TaskScoringSpec.from_dict(values["scoring_spec"]),
        )


@dataclass(frozen=True)
class BlindAssessmentRequest(CanonicalRunnerRecord):
    output_id: str
    task_prompt: str
    response_text: str
    sealed_evaluation: SealedEvaluation
    scoring_spec: TaskScoringSpec

    def __post_init__(self) -> None:
        _short("output_id", self.output_id)
        lowered = self.output_id.casefold()
        if any(token in lowered for token in _OPAQUE_FORBIDDEN):
            _fail("output_id is not condition-opaque")
        _short("task_prompt", self.task_prompt, _MAX_TEXT)
        if (
            type(self.response_text) is not str
            or len(self.response_text) > 10 * 1024 * 1024
        ):
            _fail("response_text must be bounded exact text")
        if type(self.sealed_evaluation) is not SealedEvaluation:
            _fail("sealed_evaluation must be exact")
        if type(self.scoring_spec) is not TaskScoringSpec:
            _fail("scoring_spec must be exact")
        sealed = SealedEvaluation.from_dict(self.sealed_evaluation.to_dict())
        spec = TaskScoringSpec.from_dict(self.scoring_spec.to_dict())
        if sealed.scoring_rubric.rubric_id != spec.rubric_id:
            _fail("assessment rubric/spec mismatch")
        object.__setattr__(self, "sealed_evaluation", sealed)
        object.__setattr__(self, "scoring_spec", spec)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BlindAssessmentRequest":
        values = _exact(data, tuple(item.name for item in fields(cls)))
        values["sealed_evaluation"] = SealedEvaluation.from_dict(
            values["sealed_evaluation"]
        )
        values["scoring_spec"] = TaskScoringSpec.from_dict(values["scoring_spec"])
        return cls(**values)


class SelectorFactory(Protocol):
    def __call__(self, feature_utilities: Mapping[str, float]) -> object: ...


@dataclass(frozen=True)
class ArmRuntime:
    spec: ArmSpec
    selector_factory: SelectorFactory

    def __post_init__(self) -> None:
        if type(self.spec) is not ArmSpec or not callable(self.selector_factory):
            _fail("ArmRuntime requires exact spec and callable factory")


@dataclass(frozen=True)
class RunnerClocks:
    utc_clock: Callable[[], Any]
    monotonic_clock: Callable[[], float]
    learning_clock: Callable[[], str]
    blinding_id: Callable[[str], str]

    def __post_init__(self) -> None:
        if not all(
            callable(value)
            for value in (
                self.utc_clock,
                self.monotonic_clock,
                self.learning_clock,
                self.blinding_id,
            )
        ):
            _fail("all RunnerClocks fields must be callable")


class PromptRenderer(Protocol):
    @property
    def template_hash(self) -> str: ...

    def render(
        self, inputs: TaskInputs, selected_items: Tuple[ContextItem, ...]
    ) -> ProviderRequest: ...


@dataclass(frozen=True)
class CanonicalPromptRenderer:
    system_prompt: str
    format_version: str = "adaptive-selection-prompt-v1"
    template_hash: str = field(init=False)
    template_spec: PromptTemplateSpec = field(init=False)

    def __post_init__(self) -> None:
        _short("system_prompt", self.system_prompt, _MAX_TEXT)
        _short("format_version", self.format_version)
        template = PromptTemplateSpec(self.system_prompt, self.format_version)
        object.__setattr__(self, "template_spec", template)
        object.__setattr__(self, "template_hash", template.template_hash)

    def render(
        self, inputs: TaskInputs, selected_items: Tuple[ContextItem, ...]
    ) -> ProviderRequest:
        if type(inputs) is not TaskInputs:
            _fail("renderer inputs must be exact TaskInputs")
        selected = _tuple("selected_items", selected_items, 10_000, ContextItem)
        candidate_by_id = {
            item.context_item_id: item for item in inputs.candidate_context
        }
        if any(
            item.context_item_id not in candidate_by_id
            or candidate_by_id[item.context_item_id].to_dict() != item.to_dict()
            for item in selected
        ):
            _fail("selected items must be exact input candidates")
        prompt = {
            "context": [
                {"content": item.content, "ordinal": ordinal}
                for ordinal, item in enumerate(selected)
            ],
            "format": self.format_version,
            "system": self.system_prompt,
            "task": inputs.task_prompt,
        }
        return ProviderRequest(_canonical(prompt).decode("utf-8"), self.template_hash)


class OutcomeAssessor(Protocol):
    def assess(self, request: BlindAssessmentRequest) -> BlindedAssessment: ...


@dataclass(frozen=True)
class EvaluationBatch(CanonicalRunnerRecord):
    materials: Tuple[CaseExecutionMaterial, ...]

    def __post_init__(self) -> None:
        items = cast(
            Tuple[CaseExecutionMaterial, ...],
            _tuple("materials", self.materials, _MAX_CASES, CaseExecutionMaterial),
        )
        if any(item.task_case.split != "held_out" for item in items):
            _fail("evaluation batch may contain only held_out cases")
        if len({item.task_case.task_case_id for item in items}) != len(items):
            _fail("evaluation case IDs must be unique")
        object.__setattr__(self, "materials", items)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvaluationBatch":
        values = _exact(data, ("materials",))
        return cls(
            tuple(CaseExecutionMaterial.from_dict(item) for item in values["materials"])
        )


_RECEIPT_TOKEN = object()
_GATE_TOKEN = object()


@dataclass(frozen=True, init=False)
class OutcomeAppendedReceipt:
    source_identity: int
    slot_id: str
    outcome_id: str
    family_id: str
    case_id: str
    ordinal: int

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("OutcomeAppendedReceipt is runner-minted")

    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError("OutcomeAppendedReceipt cannot be subclassed")

    @classmethod
    def _mint(
        cls,
        token: object,
        source: object,
        slot_id: str,
        outcome_id: str,
        family_id: str,
        case_id: str,
        ordinal: int,
    ) -> "OutcomeAppendedReceipt":
        if token is not _RECEIPT_TOKEN:
            raise TypeError("receipt mint is internal")
        result = object.__new__(cls)
        for name, value in (
            ("source_identity", id(source)),
            ("slot_id", _short("slot_id", slot_id)),
            ("outcome_id", _short("outcome_id", outcome_id)),
            ("family_id", _short("family_id", family_id)),
            ("case_id", _short("case_id", case_id)),
            ("ordinal", _index("ordinal", ordinal)),
        ):
            object.__setattr__(result, name, value)
        return result


@dataclass(frozen=True, init=False)
class EvaluationGate:
    source_identity: int
    expected_slots: int
    completed_slots: int
    plan_hash: str

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("EvaluationGate is runner-minted")

    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError("EvaluationGate cannot be subclassed")

    @classmethod
    def _mint(
        cls,
        token: object,
        source: object,
        expected: int,
        completed: int,
        plan_hash: str,
    ) -> "EvaluationGate":
        if token is not _GATE_TOKEN:
            raise TypeError("gate mint is internal")
        result = object.__new__(cls)
        for name, value in (
            ("source_identity", id(source)),
            ("expected_slots", _index("expected_slots", expected)),
            ("completed_slots", _index("completed_slots", completed)),
            ("plan_hash", _sha("plan_hash", plan_hash)),
        ):
            object.__setattr__(result, name, value)
        return result


class OrderedDatasetSource(Protocol):
    @property
    def dataset_version(self) -> str: ...

    @property
    def dataset_hash(self) -> str: ...

    @property
    def family_order(self) -> Tuple[str, ...]: ...

    def adaptation_case_ids(self, family_id: str) -> Tuple[str, ...]: ...

    def load_adaptation_case(
        self, family_id: str, ordinal: int
    ) -> CaseExecutionMaterial: ...

    def reveal_feedback(self, receipt: OutcomeAppendedReceipt) -> FeedbackEvent: ...

    def open_evaluation(self, gate: EvaluationGate) -> EvaluationBatch: ...


class Stage0OrderedDatasetSource:
    """Temporal capability adapter over a canonical Stage 0 bundle.

    It proves call ordering only; Python introspection can still access private objects.
    """

    def __init__(
        self,
        bundle: DatasetBundle,
        scoring_specs_by_case_id: Mapping[str, TaskScoringSpec],
    ) -> None:
        if (
            type(bundle) is not DatasetBundle
            or type(scoring_specs_by_case_id) is not dict
        ):
            _fail("Stage0 source requires exact bundle and exact spec dict")
        self._bundle = DatasetBundle.from_dict(bundle.to_dict())
        self._dataset_hash = "sha256:" + canonical_bundle_sha256(self._bundle)
        supplied = dict(scoring_specs_by_case_id)
        case_ids = {item.task_case_id for item in self._bundle.cases}
        if set(supplied) != case_ids:
            _fail("scoring specs must correspond exactly to dataset cases")
        self._specs = {
            case_id: TaskScoringSpec.from_dict(supplied[case_id].to_dict())
            for case_id in supplied
        }
        self._cases = {item.task_case_id: item for item in self._bundle.cases}
        self._plans = {item.task_family_id: item for item in self._bundle.family_plans}
        self._events = {
            item.task_case_id: item for item in self._bundle.adaptation_feedback
        }
        self._revealed_slots = set()
        self._opened = False

    @property
    def dataset_version(self) -> str:
        return self._bundle.dataset_version

    @property
    def dataset_hash(self) -> str:
        return self._dataset_hash

    @property
    def family_order(self) -> Tuple[str, ...]:
        return self._bundle.family_order

    def adaptation_case_ids(self, family_id: str) -> Tuple[str, ...]:
        _short("family_id", family_id)
        if family_id not in self._plans:
            _fail("unknown family")
        return self._plans[family_id].adaptation_order

    def load_adaptation_case(
        self, family_id: str, ordinal: int
    ) -> CaseExecutionMaterial:
        case_ids = self.adaptation_case_ids(family_id)
        _index("ordinal", ordinal)
        if ordinal >= len(case_ids):
            _fail("adaptation ordinal is out of range")
        case = self._cases[case_ids[ordinal]]
        if (
            case.split != "adaptation"
            or case.inputs.profile.task_family_id != family_id
        ):
            _fail("adaptation case order is inconsistent")
        return CaseExecutionMaterial(case, self._specs[case.task_case_id])

    def reveal_feedback(self, receipt: OutcomeAppendedReceipt) -> FeedbackEvent:
        if type(receipt) is not OutcomeAppendedReceipt or receipt.source_identity != id(
            self
        ):
            _fail("valid source-bound outcome receipt is required")
        expected_ids = self.adaptation_case_ids(receipt.family_id)
        if not receipt.outcome_id:
            _fail("receipt must bind an appended outcome")
        if (
            receipt.ordinal >= len(expected_ids)
            or expected_ids[receipt.ordinal] != receipt.case_id
        ):
            _fail("receipt is for the wrong adaptation case")
        if receipt.slot_id in self._revealed_slots:
            _fail("feedback was already revealed for this slot")
        event = self._events.get(receipt.case_id)
        if event is None:
            _fail("locked feedback is missing")
        payload = event.structured_value
        if (
            event.source != "oracle"
            or not isinstance(payload, Mapping)
            or payload.get("locked") is not True
            or payload.get("selector_independent") is not True
        ):
            _fail("feedback must be locked selector-independent oracle evidence")
        self._revealed_slots.add(receipt.slot_id)
        return FeedbackEvent.from_dict(event.to_dict())

    def open_evaluation(self, gate: EvaluationGate) -> EvaluationBatch:
        if type(gate) is not EvaluationGate or gate.source_identity != id(self):
            _fail("valid source-bound evaluation gate is required")
        if self._opened:
            _fail("evaluation may open only once")
        if gate.expected_slots != gate.completed_slots:
            _fail("evaluation gate is incomplete")
        if len(self._revealed_slots) != gate.expected_slots:
            _fail("every adaptation slot must reveal feedback before evaluation")
        self._opened = True
        materials = []
        for family in self.family_order:
            case_id = self._plans[family].held_out_case_id
            case = self._cases[case_id]
            if case.split != "held_out" or case.inputs.profile.task_family_id != family:
                _fail("held-out family order is inconsistent")
            materials.append(CaseExecutionMaterial(case, self._specs[case_id]))
        return EvaluationBatch(tuple(materials))


@dataclass(frozen=True)
class SelectionDecisionEvidence(CanonicalRunnerRecord):
    context_item_id: str
    included: bool
    reason: str
    detail: str
    score: float
    token_count: int
    score_factors: Mapping[str, Any]

    def __post_init__(self) -> None:
        _short("context_item_id", self.context_item_id)
        if type(self.included) is not bool:
            _fail("included must be boolean")
        _short("reason", self.reason)
        _short("detail", self.detail)
        object.__setattr__(self, "score", _finite("score", self.score))
        if type(self.token_count) is not int or self.token_count <= 0:
            _fail("token_count must be positive")
        object.__setattr__(self, "score_factors", _freeze_json(self.score_factors))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SelectionDecisionEvidence":
        return cls(**_exact(data, tuple(item.name for item in fields(cls))))


@dataclass(frozen=True)
class SelectionResultEvidence(CanonicalRunnerRecord):
    selector_mode: str
    selected_items: Tuple[ContextItem, ...]
    decisions: Tuple[SelectionDecisionEvidence, ...]
    eligible_context_item_ids: Tuple[str, ...]
    used_tokens: int
    token_budget: int

    def __post_init__(self) -> None:
        if self.selector_mode not in SELECTOR_MODES:
            _fail("selection mode is invalid")
        selected = cast(
            Tuple[ContextItem, ...],
            _tuple("selected_items", self.selected_items, _MAX_CASES, ContextItem),
        )
        decisions = cast(
            Tuple[SelectionDecisionEvidence, ...],
            _tuple("decisions", self.decisions, _MAX_CASES, SelectionDecisionEvidence),
        )
        eligible = _strings(
            "eligible_context_item_ids", self.eligible_context_item_ids, _MAX_CASES
        )
        _index("used_tokens", self.used_tokens)
        if type(self.token_budget) is not int or self.token_budget <= 0:
            _fail("token_budget must be positive")
        if self.used_tokens != sum(item.token_count for item in selected):
            _fail("used_tokens must equal selected item tokens")
        if self.used_tokens > self.token_budget:
            _fail("selection exceeds token budget")
        object.__setattr__(self, "selected_items", selected)
        object.__setattr__(self, "decisions", decisions)
        object.__setattr__(self, "eligible_context_item_ids", eligible)

    @classmethod
    def from_selection(cls, result: SelectionResult) -> "SelectionResultEvidence":
        if type(result) is not SelectionResult:
            _fail("selector must return exact SelectionResult")
        return cls(
            result.selector_mode,
            tuple(
                ContextItem.from_dict(item.to_dict()) for item in result.selected_items
            ),
            tuple(
                SelectionDecisionEvidence(
                    item.context_item_id,
                    item.included,
                    item.reason,
                    item.detail,
                    item.score,
                    item.token_count,
                    dict(item.score_factors),
                )
                for item in result.decisions
            ),
            tuple(result.eligible_context_item_ids),
            result.used_tokens,
            result.token_budget,
        )

    def to_selection(self) -> SelectionResult:
        return SelectionResult(
            self.selector_mode,
            self.selected_items,
            tuple(
                SelectorDecision(
                    item.context_item_id,
                    item.included,
                    item.reason,
                    item.detail,
                    item.score,
                    item.token_count,
                    item.score_factors,
                )
                for item in self.decisions
            ),
            self.eligible_context_item_ids,
            self.used_tokens,
            self.token_budget,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SelectionResultEvidence":
        values = _exact(data, tuple(item.name for item in fields(cls)))
        values["selected_items"] = tuple(
            ContextItem.from_dict(item) for item in values["selected_items"]
        )
        values["decisions"] = tuple(
            SelectionDecisionEvidence.from_dict(item) for item in values["decisions"]
        )
        values["eligible_context_item_ids"] = tuple(values["eligible_context_item_ids"])
        return cls(**values)


@dataclass(frozen=True)
class LearningStateEvidence(CanonicalRunnerRecord):
    task_family_id: str
    feedback_event_ids: Tuple[str, ...]
    snapshot_payload: Mapping[str, Any]
    state_hash: str = field(init=False)

    def __post_init__(self) -> None:
        _short("task_family_id", self.task_family_id)
        events = _strings("feedback_event_ids", self.feedback_event_ids, _MAX_CASES)
        payload = _freeze_json(self.snapshot_payload)
        object.__setattr__(self, "feedback_event_ids", events)
        object.__setattr__(self, "snapshot_payload", payload)
        object.__setattr__(
            self,
            "state_hash",
            _domain_hash(
                "ordered-learning-state-v1",
                {
                    "feedback_event_ids": events,
                    "snapshot_payload": payload,
                    "task_family_id": self.task_family_id,
                },
            ),
        )

    @classmethod
    def from_snapshot(
        cls, family: str, prefix: Tuple[str, ...], snapshot: LearningSnapshot
    ) -> "LearningStateEvidence":
        if type(snapshot) is not LearningSnapshot:
            _fail("learning snapshot must be sealed exact LearningSnapshot")
        return cls(family, prefix, snapshot.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LearningStateEvidence":
        values = _exact(data, tuple(item.name for item in fields(cls)))
        expected = values.pop("state_hash")
        values["feedback_event_ids"] = tuple(values["feedback_event_ids"])
        result = cls(**values)
        if result.state_hash != expected:
            _fail("state_hash does not match payload")
        return result


@dataclass(frozen=True)
class PhaseTraceEvent(CanonicalRunnerRecord):
    sequence: int
    state: str
    action: str
    family_id: Optional[str]
    case_id: Optional[str]
    arm_id: Optional[str]
    repetition_index: Optional[int]
    feedback_prefix_event_ids: Tuple[str, ...]
    learning_state_hash: Optional[str]
    timestamp: str

    def __post_init__(self) -> None:
        _index("sequence", self.sequence)
        _short("state", self.state)
        _short("action", self.action)
        for name in ("family_id", "case_id", "arm_id"):
            value = getattr(self, name)
            if value is not None:
                _short(name, value)
        if self.repetition_index is not None:
            _index("repetition_index", self.repetition_index)
        object.__setattr__(
            self,
            "feedback_prefix_event_ids",
            _strings(
                "feedback_prefix_event_ids", self.feedback_prefix_event_ids, _MAX_CASES
            ),
        )
        if self.learning_state_hash is not None:
            _sha("learning_state_hash", self.learning_state_hash)
        _utc_timestamp("timestamp", self.timestamp)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PhaseTraceEvent":
        values = _exact(data, tuple(item.name for item in fields(cls)))
        values["feedback_prefix_event_ids"] = tuple(values["feedback_prefix_event_ids"])
        return cls(**values)


@dataclass
class _TraceBuffer:
    events: list
    clock: Callable[[], str]

    def append(self, event: PhaseTraceEvent) -> None:
        self.events.append(event)

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self):
        return iter(self.events)


@dataclass(frozen=True)
class OrderedTaskRecord(CanonicalRunnerRecord):
    sequence: int
    phase: str
    family_id: str
    task_case_id: str
    ordinal: int
    arm_id: str
    repetition_index: int
    case_material: CaseExecutionMaterial
    selector_input_hash: str
    candidate_set_hash: str
    feedback_prefix_before: Tuple[str, ...]
    learning_state_before: LearningStateEvidence
    selection_result: SelectionResultEvidence
    selection_decision: SelectionDecision
    policy_decision_hash: str
    provider_request: ProviderRequest
    provider_execution: ProviderExecution
    scoring_result: ScoringResult
    task_outcome: TaskOutcome
    revealed_feedback_events: Tuple[FeedbackEvent, ...]
    feedback_prefix_after: Tuple[str, ...]
    learning_state_after: LearningStateEvidence
    task_record_hash: str = field(init=False)

    def __post_init__(self) -> None:
        _index("sequence", self.sequence)
        if self.phase not in {"adaptation", "evaluation"}:
            _fail("phase is invalid")
        for name in ("family_id", "task_case_id", "arm_id"):
            _short(name, getattr(self, name))
        _index("ordinal", self.ordinal)
        _index("repetition_index", self.repetition_index)
        if type(self.case_material) is not CaseExecutionMaterial:
            _fail("case_material must be exact")
        material = CaseExecutionMaterial.from_dict(self.case_material.to_dict())
        case = material.task_case
        inputs = case.inputs
        expected_phase = "adaptation" if case.split == "adaptation" else "evaluation"
        if (
            self.phase != expected_phase
            or self.family_id != inputs.profile.task_family_id
            or self.task_case_id != case.task_case_id
            or (self.phase == "evaluation" and self.ordinal != 0)
        ):
            _fail("task identity does not match embedded case material")
        selector_input_hash, candidate_set_hash = _visible_hashes(inputs)
        if self.selector_input_hash != selector_input_hash:
            _fail("selector_input_hash does not match embedded inputs")
        if self.candidate_set_hash != candidate_set_hash:
            _fail("candidate_set_hash does not match embedded candidates")
        _sha("policy_decision_hash", self.policy_decision_hash)
        before = _strings(
            "feedback_prefix_before", self.feedback_prefix_before, _MAX_CASES
        )
        events = cast(
            Tuple[FeedbackEvent, ...],
            _tuple(
                "revealed_feedback_events",
                self.revealed_feedback_events,
                1,
                FeedbackEvent,
            ),
        )
        events = tuple(FeedbackEvent.from_dict(item.to_dict()) for item in events)
        revealed = tuple(item.event_id for item in events)
        after = _strings(
            "feedback_prefix_after", self.feedback_prefix_after, _MAX_CASES
        )
        if (
            type(self.learning_state_before) is not LearningStateEvidence
            or type(self.learning_state_after) is not LearningStateEvidence
        ):
            _fail("learning evidence records must be exact")
        if (
            self.learning_state_before.task_family_id != self.family_id
            or self.learning_state_after.task_family_id != self.family_id
            or self.learning_state_before.feedback_event_ids != before
            or self.learning_state_after.feedback_event_ids != after
        ):
            _fail("learning evidence does not match task family/prefix")
        if type(self.selection_result) is not SelectionResultEvidence:
            _fail("selection_result must be exact")
        _validate_selection(
            self.selection_result, inputs, self.selection_result.selector_mode
        )
        if type(self.selection_decision) is not SelectionDecision:
            _fail("selection_decision must be exact")
        decision = SelectionDecision.from_dict(self.selection_decision.to_dict())
        selected_ids = tuple(
            item.context_item_id for item in self.selection_result.selected_items
        )
        selected_tokens = tuple(
            item.token_count for item in self.selection_result.selected_items
        )
        trace_hash = _domain_hash("selection-trace-v1", self.selection_result.to_dict())
        if (
            decision.task_case_id != self.task_case_id
            or decision.selected_context_item_ids != selected_ids
            or decision.selected_token_counts != selected_tokens
            or decision.total_selected_tokens != self.selection_result.used_tokens
            or decision.token_budget != self.selection_result.token_budget
            or decision.selector_score is not None
            or decision.selector_input_hash != selector_input_hash
            or decision.candidate_set_hash != candidate_set_hash
            or decision.ranking_artifact_hash != trace_hash
            or decision.ranking_artifact_reference != "inline:selection:" + trace_hash
            or decision.trace_artifact_hash != trace_hash
            or decision.trace_artifact_reference
            != "inline:selection-trace:" + trace_hash
        ):
            _fail("selection decision does not exactly bind selection evidence")
        expected_decision_id = _domain_hash(
            "selection-decision-v1",
            {
                "policy_decision_hash": self.policy_decision_hash,
                "run_id": decision.run_id,
                "task_case_id": self.task_case_id,
            },
        )
        if decision.decision_id != expected_decision_id:
            _fail("selection decision ID is not derived from policy/run/case")
        if (
            type(self.provider_request) is not ProviderRequest
            or type(self.provider_execution) is not ProviderExecution
        ):
            _fail("provider records must be exact")
        request = ProviderRequest.from_dict(self.provider_request.to_dict())
        try:
            prompt_payload = json.loads(request.prompt_text)
        except (TypeError, ValueError):
            prompt_payload = None
        if (
            isinstance(prompt_payload, dict)
            and prompt_payload.get("format") == "adaptive-selection-prompt-v1"
        ):
            expected_context = [
                {"content": item.content, "ordinal": ordinal}
                for ordinal, item in enumerate(self.selection_result.selected_items)
            ]
            if (
                prompt_payload.get("task") != inputs.task_prompt
                or prompt_payload.get("context") != expected_context
                or set(prompt_payload) != {"context", "format", "system", "task"}
                or _canonical(prompt_payload).decode("utf-8") != request.prompt_text
            ):
                _fail("canonical prompt does not exactly bind task/selected items")
        execution = ProviderExecution.from_dict(self.provider_execution.to_dict())
        if execution.request.to_dict() != request.to_dict():
            _fail("provider execution does not embed the exact request")
        if (
            type(self.scoring_result) is not ScoringResult
            or type(self.task_outcome) is not TaskOutcome
        ):
            _fail("scoring/outcome records must be exact")
        scoring = ScoringResult.from_dict(self.scoring_result.to_dict())
        if (
            scoring.rubric.to_dict() != case.sealed_evaluation.scoring_rubric.to_dict()
            or scoring.spec.to_dict() != material.scoring_spec.to_dict()
        ):
            _fail("scoring result does not bind embedded rubric/spec")
        outcome = TaskOutcome.from_dict(self.task_outcome.to_dict())
        expected_outcome = _outcome(
            decision.run_id,
            case,
            decision,
            execution,
            scoring,
            outcome.provenance,
        )
        if outcome.to_dict() != expected_outcome.to_dict():
            _fail("task outcome is not exactly derived from execution/scoring")
        if self.phase == "evaluation":
            if (
                events
                or before != after
                or self.learning_state_before != self.learning_state_after
            ):
                _fail("evaluation cannot reveal feedback or change learning state")
        else:
            if len(events) != 1:
                _fail("adaptation must reveal exactly its current feedback event")
            event = events[0]
            payload = event.structured_value
            if (
                event.task_case_id != self.task_case_id
                or event.task_family_id != self.family_id
                or event.source != "oracle"
                or not isinstance(payload, Mapping)
                or payload.get("locked") is not True
                or payload.get("selector_independent") is not True
            ):
                _fail(
                    "revealed feedback must be current locked selector-independent evidence"
                )
        if after != before + revealed:
            _fail("feedback prefixes must append exactly revealed events")
        object.__setattr__(self, "case_material", material)
        object.__setattr__(self, "feedback_prefix_before", before)
        object.__setattr__(self, "revealed_feedback_events", events)
        object.__setattr__(self, "feedback_prefix_after", after)
        payload = {
            item.name: getattr(self, item.name) for item in fields(self) if item.init
        }
        object.__setattr__(
            self,
            "task_record_hash",
            _domain_hash("ordered-task-record-v1", payload),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OrderedTaskRecord":
        values = _exact(data, tuple(item.name for item in fields(cls)))
        expected = values.pop("task_record_hash")
        values["case_material"] = CaseExecutionMaterial.from_dict(
            values["case_material"]
        )
        for name in ("feedback_prefix_before", "feedback_prefix_after"):
            values[name] = tuple(values[name])
        values["revealed_feedback_events"] = tuple(
            FeedbackEvent.from_dict(item) for item in values["revealed_feedback_events"]
        )
        values["learning_state_before"] = LearningStateEvidence.from_dict(
            values["learning_state_before"]
        )
        values["learning_state_after"] = LearningStateEvidence.from_dict(
            values["learning_state_after"]
        )
        values["selection_result"] = SelectionResultEvidence.from_dict(
            values["selection_result"]
        )
        values["selection_decision"] = SelectionDecision.from_dict(
            values["selection_decision"]
        )
        values["provider_request"] = ProviderRequest.from_dict(
            values["provider_request"]
        )
        values["provider_execution"] = ProviderExecution.from_dict(
            values["provider_execution"]
        )
        values["scoring_result"] = ScoringResult.from_dict(values["scoring_result"])
        values["task_outcome"] = TaskOutcome.from_dict(values["task_outcome"])
        result = cls(**values)
        if result.task_record_hash != expected:
            _fail("task_record_hash does not match payload")
        return result


@dataclass(frozen=True)
class ArmRunRecord(CanonicalRunnerRecord):
    run_id: str
    arm_id: str
    repetition_index: int
    provider_seed: int
    manifest: RunManifest
    task_records: Tuple[OrderedTaskRecord, ...]
    final_learning_states: Tuple[LearningStateEvidence, ...]
    completed_timestamp: str
    run_hash: str = field(init=False)

    def __post_init__(self) -> None:
        _short("run_id", self.run_id)
        _short("arm_id", self.arm_id)
        _index("repetition_index", self.repetition_index)
        _int64("provider_seed", self.provider_seed)
        if type(self.manifest) is not RunManifest:
            _fail("manifest must be exact RunManifest")
        tasks = cast(
            Tuple[OrderedTaskRecord, ...],
            _tuple("task_records", self.task_records, _MAX_CASES, OrderedTaskRecord),
        )
        states = cast(
            Tuple[LearningStateEvidence, ...],
            _tuple(
                "final_learning_states",
                self.final_learning_states,
                _MAX_FAMILIES,
                LearningStateEvidence,
            ),
        )
        _utc_timestamp("completed_timestamp", self.completed_timestamp)
        if self.manifest.run_id != self.run_id:
            _fail("manifest run_id mismatch")
        if any(
            item.arm_id != self.arm_id
            or item.repetition_index != self.repetition_index
            or item.selection_decision.run_id != self.run_id
            or item.task_outcome.run_id != self.run_id
            for item in tasks
        ):
            _fail("task records belong to a different run")
        if len({item.task_case_id for item in tasks}) != len(tasks):
            _fail("duplicate outcome for run/case")
        if tuple(item.sequence for item in tasks) != tuple(
            sorted(item.sequence for item in tasks)
        ):
            _fail("run task records must preserve global sequence order")
        if any(
            item.provider_execution.provider != self.manifest.provider
            or item.provider_execution.model_id != self.manifest.model_id
            or item.provider_execution.config_hash != self.manifest.config_hash
            or item.provider_request.prompt_template_hash
            != self.manifest.prompt_template_hash
            for item in tasks
        ):
            _fail("task provider/config/template does not match manifest")
        completed = _timestamp_value(self.completed_timestamp)
        if completed < _timestamp_value(self.manifest.started_timestamp) or any(
            completed < _timestamp_value(item.task_outcome.completed_timestamp)
            for item in tasks
        ):
            _fail("run completion precedes manifest or task completion")
        if len({item.task_family_id for item in states}) != len(states):
            _fail("final learning-state families must be unique")
        for state in states:
            family_tasks = [
                item for item in tasks if item.family_id == state.task_family_id
            ]
            if not family_tasks or family_tasks[-1].learning_state_after != state:
                _fail("final learning state must equal the run's last family state")
        object.__setattr__(self, "task_records", tasks)
        object.__setattr__(self, "final_learning_states", states)
        payload = {
            item.name: getattr(self, item.name) for item in fields(self) if item.init
        }
        object.__setattr__(
            self, "run_hash", _domain_hash("ordered-arm-run-v1", payload)
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArmRunRecord":
        values = _exact(data, tuple(item.name for item in fields(cls)))
        expected = values.pop("run_hash")
        values["manifest"] = RunManifest.from_dict(values["manifest"])
        values["task_records"] = tuple(
            OrderedTaskRecord.from_dict(item) for item in values["task_records"]
        )
        values["final_learning_states"] = tuple(
            LearningStateEvidence.from_dict(item)
            for item in values["final_learning_states"]
        )
        result = cls(**values)
        if result.run_hash != expected:
            _fail("run_hash does not match payload")
        return result


def _snapshot_timestamp(evidence: LearningStateEvidence) -> Optional[str]:
    payload = evidence.snapshot_payload
    estimates = tuple(payload.get("feature_estimates", ())) + tuple(
        payload.get("id_local_estimates", ())
    )
    timestamps = {
        item.get("estimated_timestamp")
        for item in estimates
        if isinstance(item, Mapping)
    }
    if not estimates:
        if timestamps:
            _fail("empty learning snapshot has timestamps")
        return None
    if len(timestamps) != 1 or None in timestamps:
        _fail("learning snapshot must contain one estimate timestamp")
    return cast(str, next(iter(timestamps)))


def _replay_learning_state(
    family: str,
    events: Tuple[FeedbackEvent, ...],
    inputs: Mapping[str, TaskInputs],
    policy: LearningPolicy,
    expected: LearningStateEvidence,
) -> None:
    timestamp = _snapshot_timestamp(expected)
    calls = [0]

    def replay_clock() -> str:
        calls[0] += 1
        if timestamp is None or calls[0] > 1:
            _fail("learning replay consumed an unexpected timestamp")
        return cast(str, timestamp)

    snapshot = learn_utilities(events, inputs, policy, replay_clock)
    actual = LearningStateEvidence.from_snapshot(
        family, tuple(item.event_id for item in events), snapshot
    )
    expected_calls = 0 if timestamp is None else 1
    if calls[0] != expected_calls or actual.to_dict() != expected.to_dict():
        _fail("learning state is not derivable from exact feedback/input prefix")


def _expected_trace(
    tasks: Tuple[OrderedTaskRecord, ...], timestamps: Tuple[str, ...]
) -> Tuple[PhaseTraceEvent, ...]:
    timestamp_iter = iter(timestamps)

    def expected_clock() -> str:
        try:
            return next(timestamp_iter)
        except StopIteration:
            _fail("phase trace contains too few timestamps")

    trace = _TraceBuffer([], expected_clock)
    _trace(trace, "ADAPTATION", "adaptation_started")
    adaptation = tuple(item for item in tasks if item.phase == "adaptation")
    evaluation = tuple(item for item in tasks if item.phase == "evaluation")
    for task in adaptation:
        common = (
            task.family_id,
            task.task_case_id,
            task.arm_id,
            task.repetition_index,
        )
        before = task.feedback_prefix_before
        before_hash = task.learning_state_before.state_hash
        for action in (
            "case_loaded",
            "selected",
            "rendered",
            "executed",
            "assessed",
            "outcome_appended",
        ):
            _trace(trace, "ADAPTATION", action, *common, before, before_hash)
        _trace(
            trace,
            "ADAPTATION",
            "feedback_revealed",
            *common,
            task.feedback_prefix_after,
            before_hash,
        )
        _trace(
            trace,
            "ADAPTATION",
            "learner_updated",
            *common,
            task.feedback_prefix_after,
            task.learning_state_after.state_hash,
        )
    _trace(trace, "ADAPTATION_COMPLETE", "adaptation_completed")
    _trace(trace, "EVALUATION_OPEN", "evaluation_opened")
    _trace(trace, "EVALUATION", "evaluation_started")
    for task in evaluation:
        common = (
            task.family_id,
            task.task_case_id,
            task.arm_id,
            task.repetition_index,
        )
        for action in (
            "case_loaded",
            "selected",
            "rendered",
            "executed",
            "assessed",
            "evaluation_outcome_appended",
        ):
            _trace(
                trace,
                "EVALUATION",
                action,
                *common,
                task.feedback_prefix_before,
                task.learning_state_before.state_hash,
            )
    _trace(trace, "EVALUATION", "evaluation_completed")
    _trace(trace, "COMPLETE", "experiment_completed")
    result = tuple(trace)
    try:
        next(timestamp_iter)
    except StopIteration:
        return result
    _fail("phase trace contains excess timestamps")


def _validate_artifact_semantics(
    plan: ExperimentPlan,
    runs: Tuple[ArmRunRecord, ...],
    trace: Tuple[PhaseTraceEvent, ...],
) -> None:
    arm_by_id = {item.arm_id: item for item in plan.arms}
    rep_by_index = {item.repetition_index: item for item in plan.repetitions}
    all_tasks = tuple(
        sorted(
            (task for run in runs for task in run.task_records),
            key=lambda x: x.sequence,
        )
    )
    if tuple(item.sequence for item in all_tasks) != tuple(range(len(all_tasks))):
        _fail("global task sequence union must be unique and contiguous")
    if not all_tasks:
        _fail("artifact must contain task records")

    expected_shape = tuple(
        (item.phase, item.family_id, item.task_case_id, item.ordinal)
        for item in runs[0].task_records
    )
    adaptation_shape = tuple(item for item in expected_shape if item[0] == "adaptation")
    evaluation_shape = tuple(item for item in expected_shape if item[0] == "evaluation")
    if expected_shape != adaptation_shape + evaluation_shape:
        _fail("every run must execute adaptation before evaluation")
    for family in plan.family_order:
        family_adaptation = tuple(
            item for item in adaptation_shape if item[1] == family
        )
        family_evaluation = tuple(
            item for item in evaluation_shape if item[1] == family
        )
        if (
            not family_adaptation
            or tuple(item[3] for item in family_adaptation)
            != tuple(range(len(family_adaptation)))
            or len(family_evaluation) != 1
            or family_evaluation[0][3] != 0
        ):
            _fail("run phase coverage/ordinals do not match family order")
    if (
        tuple(dict.fromkeys(item[1] for item in adaptation_shape)) != plan.family_order
        or tuple(item[1] for item in evaluation_shape) != plan.family_order
    ):
        _fail("run family phase order does not match plan")

    locked_feedback_by_case: Dict[str, Dict[str, Any]] = {}
    visible_fairness: Dict[str, Tuple[Any, ...]] = {}
    runtime_fairness: Dict[Tuple[str, int], Tuple[Any, ...]] = {}
    for run in runs:
        arm = arm_by_id[run.arm_id]
        rep = rep_by_index[run.repetition_index]
        expected_run_id = _domain_hash(
            "ordered-run-id-v1",
            {
                "arm_hash": arm.arm_hash,
                "plan_hash": plan.plan_hash,
                "provider_seed": rep.provider_seed,
                "repetition_index": rep.repetition_index,
            },
        )
        manifest = run.manifest
        if (
            run.run_id != expected_run_id
            or run.provider_seed != rep.provider_seed
            or manifest.run_id != expected_run_id
            or manifest.experiment_version != plan.experiment_version
            or manifest.protocol_version != plan.protocol_version
            or manifest.dataset_version != plan.dataset_version
            or manifest.dataset_hash != plan.dataset_hash
            or manifest.selector_mode != arm.selector_mode
            or manifest.selector_version != arm.selector_version
            or manifest.code_revision != plan.code_revision
            or manifest.prompt_template_hash != plan.prompt_template_hash
            or manifest.provenance != plan.provenance
        ):
            _fail("run/manifest identity does not match plan")
        shape = tuple(
            (item.phase, item.family_id, item.task_case_id, item.ordinal)
            for item in run.task_records
        )
        if shape != expected_shape:
            _fail("every run must execute exactly the same ordered cases")
        if (
            tuple(item.task_family_id for item in run.final_learning_states)
            != plan.family_order
        ):
            _fail("final learning states must follow plan family order")

        events_by_family: Dict[str, list] = {family: [] for family in plan.family_order}
        inputs_by_family: Dict[str, Dict[str, TaskInputs]] = {
            family: {} for family in plan.family_order
        }
        for task in run.task_records:
            if task.selection_result.selector_mode != arm.selector_mode:
                _fail("task selector mode does not match arm")
            prompt_renderer = CanonicalPromptRenderer(
                plan.prompt_template.system_prompt,
                plan.prompt_template.format_version,
            )
            expected_request = prompt_renderer.render(
                task.case_material.task_case.inputs,
                task.selection_result.selected_items,
            )
            if task.provider_request.to_dict() != expected_request.to_dict():
                _fail("provider request is not derived from the frozen prompt template")
            expected_policy_hash = _domain_hash(
                "policy-decision-v1",
                {
                    "feedback_prefix_event_ids": task.feedback_prefix_before,
                    "learning_state_hash": task.learning_state_before.state_hash,
                    "selection_result": task.selection_result,
                    "selector_config_hash": arm.selector_config_hash,
                    "selector_input_hash": task.selector_input_hash,
                    "selector_mode": arm.selector_mode,
                    "selector_version": arm.selector_version,
                },
            )
            if task.policy_decision_hash != expected_policy_hash:
                _fail(
                    "policy decision hash is not derivable from arm/input/state/selection"
                )
            if (
                task.selection_decision.provenance != plan.provenance
                or task.task_outcome.provenance != plan.provenance
            ):
                _fail("task evidence provenance does not match plan")
            family_events = tuple(events_by_family[task.family_id])
            family_inputs = inputs_by_family[task.family_id]
            _replay_learning_state(
                task.family_id,
                family_events,
                family_inputs,
                plan.learning_policy,
                task.learning_state_before,
            )
            if task.phase == "adaptation":
                event = task.revealed_feedback_events[0]
                locked = locked_feedback_by_case.setdefault(
                    task.task_case_id, event.to_dict()
                )
                if locked != event.to_dict():
                    _fail("locked feedback must be identical across arms/repetitions")
                events_by_family[task.family_id].append(event)
                family_inputs[task.task_case_id] = task.case_material.task_case.inputs
            _replay_learning_state(
                task.family_id,
                tuple(events_by_family[task.family_id]),
                family_inputs,
                plan.learning_policy,
                task.learning_state_after,
            )
            visible_value = (
                task.selector_input_hash,
                task.candidate_set_hash,
                task.selection_result.token_budget,
                task.scoring_result.spec_hash,
            )
            if (
                task.task_case_id in visible_fairness
                and visible_fairness[task.task_case_id] != visible_value
            ):
                _fail("same-case visible inputs differ across arms or repetitions")
            visible_fairness[task.task_case_id] = visible_value
            runtime_key = (task.task_case_id, task.repetition_index)
            runtime_value = (
                visible_value,
                task.provider_execution.configuration.to_dict(),
            )
            if (
                runtime_key in runtime_fairness
                and runtime_fairness[runtime_key] != runtime_value
            ):
                _fail("same-case/repetition fairness invariant failed across arms")
            runtime_fairness[runtime_key] = runtime_value

    for rep in plan.repetitions:
        same_rep = [
            item.manifest
            for item in runs
            if item.repetition_index == rep.repetition_index
        ]
        for other in same_rep[1:]:
            comparison = compare_manifests(same_rep[0], other)
            if not comparison.valid_for_primary_comparison:
                _fail("within-repetition manifests are not primary-compatible")
    if any(
        _timestamp_value(later.timestamp) < _timestamp_value(earlier.timestamp)
        for earlier, later in zip(trace, trace[1:])
    ):
        _fail("phase trace timestamps must be nondecreasing")
    if _expected_trace(all_tasks, tuple(item.timestamp for item in trace)) != trace:
        _fail(
            "phase trace is missing, reordered, or inconsistent with task transitions"
        )


_ARTIFACT_TOKEN = object()


@dataclass(frozen=True, init=False)
class OrderedExperimentArtifact(CanonicalRunnerRecord):
    artifact_version: str
    plan: ExperimentPlan
    plan_hash: str
    arm_runs: Tuple[ArmRunRecord, ...]
    phase_trace: Tuple[PhaseTraceEvent, ...]
    completed_timestamp: str
    artifact_hash: str

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("OrderedExperimentArtifact is runner-derived")

    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError("OrderedExperimentArtifact cannot be subclassed")

    @classmethod
    def _derive(
        cls,
        token: object,
        plan: ExperimentPlan,
        arm_runs: Tuple[ArmRunRecord, ...],
        phase_trace: Tuple[PhaseTraceEvent, ...],
        completed_timestamp: str,
    ) -> "OrderedExperimentArtifact":
        if token is not _ARTIFACT_TOKEN:
            raise TypeError("artifact derivation is internal")
        if type(plan) is not ExperimentPlan:
            _fail("artifact plan must be exact")
        runs = cast(
            Tuple[ArmRunRecord, ...],
            _tuple("arm_runs", arm_runs, _MAX_ARMS * _MAX_REPETITIONS, ArmRunRecord),
        )
        trace = cast(
            Tuple[PhaseTraceEvent, ...],
            _tuple("phase_trace", phase_trace, _MAX_TRACE, PhaseTraceEvent),
        )
        expected_pairs = tuple(
            (arm.arm_id, rep.repetition_index)
            for arm in plan.arms
            for rep in plan.repetitions
        )
        if (
            tuple((item.arm_id, item.repetition_index) for item in runs)
            != expected_pairs
        ):
            _fail("arm run order must match plan")
        if tuple(item.sequence for item in trace) != tuple(range(len(trace))):
            _fail("trace sequences must be contiguous")
        _validate_artifact_semantics(plan, runs, trace)
        completed_timestamp = _utc_timestamp("completed_timestamp", completed_timestamp)
        if any(item.completed_timestamp != completed_timestamp for item in runs):
            _fail("run and artifact completion timestamps must match")
        if any(
            _timestamp_value(item.timestamp) > _timestamp_value(completed_timestamp)
            for item in trace
        ):
            _fail("artifact completion must not precede phase trace events")
        result = object.__new__(cls)
        values = {
            "artifact_version": ARTIFACT_VERSION,
            "plan": plan,
            "plan_hash": plan.plan_hash,
            "arm_runs": runs,
            "phase_trace": trace,
            "completed_timestamp": completed_timestamp,
        }
        for name, value in values.items():
            object.__setattr__(result, name, value)
        object.__setattr__(
            result,
            "artifact_hash",
            _domain_hash("ordered-experiment-artifact-v1", values),
        )
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OrderedExperimentArtifact":
        values = _exact(data, tuple(item.name for item in fields(cls)))
        expected = values.pop("artifact_hash")
        if values.pop("artifact_version") != ARTIFACT_VERSION:
            _fail("unsupported artifact_version")
        supplied_plan_hash = values.pop("plan_hash")
        plan = ExperimentPlan.from_dict(values.pop("plan"))
        if supplied_plan_hash != plan.plan_hash:
            _fail("artifact plan_hash mismatch")
        result = cls._derive(
            _ARTIFACT_TOKEN,
            plan,
            tuple(ArmRunRecord.from_dict(item) for item in values.pop("arm_runs")),
            tuple(
                PhaseTraceEvent.from_dict(item) for item in values.pop("phase_trace")
            ),
            values.pop("completed_timestamp"),
        )
        if values or result.artifact_hash != expected:
            _fail("artifact_hash does not match payload")
        return result


@dataclass
class _RunWork:
    runtime: ArmRuntime
    repetition: RepetitionSpec
    run_id: str
    provider: Provider
    manifest: RunManifest
    tasks: list
    prefixes: Dict[str, list]
    inputs: Dict[str, Dict[str, TaskInputs]]
    snapshots: Dict[str, LearningSnapshot]
    evidence: Dict[str, LearningStateEvidence]
    events: Dict[str, list]
    appended_outcome_ids: set
    selector_instances: list


def _call_seam(category: str, stage: str, callback: Callable[[], Any]) -> Any:
    try:
        return callback()
    except RunnerError:
        raise
    except Exception as exc:
        raise RunnerError(category, stage) from exc


def _trace(
    trace: _TraceBuffer,
    state: str,
    action: str,
    family: Optional[str] = None,
    case: Optional[str] = None,
    arm: Optional[str] = None,
    repetition: Optional[int] = None,
    prefix: Tuple[str, ...] = (),
    state_hash: Optional[str] = None,
) -> None:
    trace.append(
        PhaseTraceEvent(
            len(trace),
            state,
            action,
            family,
            case,
            arm,
            repetition,
            prefix,
            state_hash,
            trace.clock(),
        )
    )
    if len(trace) > _MAX_TRACE:
        _fail("phase trace exceeds resource bound")


def _schedule(
    plan: ExperimentPlan, family: str, case: str
) -> Tuple[Tuple[str, int], ...]:
    slots = tuple(
        (arm.arm_id, rep.repetition_index)
        for rep in plan.repetitions
        for arm in plan.arms
    )
    return tuple(
        sorted(
            slots,
            key=lambda slot: _domain_hash(
                "ordered-local-schedule-v1",
                {
                    "arm_id": slot[0],
                    "case_id": case,
                    "family_id": family,
                    "repetition_index": slot[1],
                    "schedule_seed": plan.schedule_seed,
                },
            ),
        )
    )


def _visible_hashes(inputs: TaskInputs) -> Tuple[str, str]:
    canonical_inputs = TaskInputs.from_dict(inputs.to_dict())
    return (
        _domain_hash("selector-input-v1", canonical_inputs.to_dict()),
        _domain_hash(
            "ordered-candidate-pool-v1",
            [item.to_dict() for item in canonical_inputs.candidate_context],
        ),
    )


def _validate_selection(
    result: SelectionResultEvidence, inputs: TaskInputs, expected_mode: str
) -> None:
    if (
        result.selector_mode != expected_mode
        or result.token_budget != inputs.token_budget
    ):
        _fail("selector mode or budget mismatch")
    candidate_ids = tuple(item.context_item_id for item in inputs.candidate_context)
    if tuple(item.context_item_id for item in result.decisions) != candidate_ids:
        _fail("selector trace must contain every candidate exactly once in order")
    selected_ids = tuple(item.context_item_id for item in result.selected_items)
    included_ids = tuple(
        item.context_item_id for item in result.decisions if item.included
    )
    if set(selected_ids) != set(included_ids) or len(set(selected_ids)) != len(
        selected_ids
    ):
        _fail("selected items and trace inclusion must correspond exactly")
    by_id = {item.context_item_id: item for item in inputs.candidate_context}
    if any(
        by_id[item.context_item_id].to_dict() != item.to_dict()
        for item in result.selected_items
    ):
        _fail("selected items must be exact candidates")
    if any(
        item.token_count != by_id[item.context_item_id].token_count
        for item in result.decisions
    ):
        _fail("trace token counts must be authoritative")


def _outcome(
    run_id: str,
    case: TaskCase,
    decision: SelectionDecision,
    execution: ProviderExecution,
    scoring: ScoringResult,
    provenance: str,
) -> TaskOutcome:
    scored = scoring.status == "scored" and bool(execution.response_text)
    scoring_hash = _domain_hash("complete-scoring-result-v1", scoring.to_dict())
    outcome_id = _domain_hash(
        "task-outcome-v1", {"run_id": run_id, "task_case_id": case.task_case_id}
    )
    common = dict(
        outcome_id=outcome_id,
        run_id=run_id,
        task_case_id=case.task_case_id,
        selection_decision_id=decision.decision_id,
        response_text=execution.response_text,
        rubric_id=scoring.rubric_id,
        scorer_id="deterministic-blinded-scorer",
        scorer_version=scoring.engine_version,
        scorer_hash=scoring.scorer_hash,
        aggregation_method=scoring.normalization_version,
        aggregation_version=scoring.rule_version,
        model_input_tokens=execution.input_tokens,
        model_output_tokens=execution.output_tokens,
        execution_latency_ms=execution.latency_ms,
        provider_response_artifact_reference="inline:provider-response:"
        + execution.raw_response_hash,
        provider_response_hash=execution.raw_response_hash,
        completed_timestamp=execution.completed_timestamp,
        provenance=provenance,
    )
    if scored:
        return TaskOutcome(
            execution_status="success",
            raw_score=scoring.raw_score,
            max_score=scoring.max_score,
            normalized_score=scoring.normalized_score,
            criterion_scores=scoring.criterion_scores,
            evaluation_artifact_reference="inline:scoring:" + scoring_hash,
            evaluation_artifact_hash=scoring_hash,
            error_category=None,
            **common,
        )
    category = (
        "empty_provider_response"
        if not execution.response_text
        else "needs_adjudication"
    )
    return TaskOutcome(
        execution_status="failure",
        raw_score=None,
        max_score=None,
        normalized_score=None,
        criterion_scores=(),
        evaluation_artifact_reference="inline:scoring:" + scoring_hash,
        evaluation_artifact_hash=scoring_hash,
        error_category=category,
        **common,
    )


def _execute_slot(
    *,
    plan: ExperimentPlan,
    dataset: OrderedDatasetSource,
    work: _RunWork,
    material: CaseExecutionMaterial,
    phase: str,
    ordinal: int,
    sequence: int,
    renderer: PromptRenderer,
    assessor: OutcomeAssessor,
    clocks: RunnerClocks,
    trace: _TraceBuffer,
) -> Tuple[OrderedTaskRecord, Optional[FeedbackEvent]]:
    case = TaskCase.from_dict(material.task_case.to_dict())
    spec = TaskScoringSpec.from_dict(material.scoring_spec.to_dict())
    family = case.inputs.profile.task_family_id
    prefix_before = tuple(work.prefixes[family])
    state_before = work.evidence[family]
    utilities = (
        dict(work.snapshots[family].feature_utilities_for(family))
        if work.runtime.spec.uses_feature_learning
        else {}
    )
    selector = _call_seam(
        "selector_factory_failure",
        phase,
        lambda: work.runtime.selector_factory(MappingProxyType(utilities)),
    )
    if selector is None or not callable(getattr(selector, "select", None)):
        _fail("selector factory must return a fresh selector with select(inputs)")
    if any(selector is prior for prior in work.selector_instances):
        _fail("selector factory must return a fresh selector for every slot")
    work.selector_instances.append(selector)
    inputs = TaskInputs.from_dict(case.inputs.to_dict())
    selector_input_hash, candidate_set_hash = _visible_hashes(inputs)
    _trace(
        trace,
        phase.upper(),
        "case_loaded",
        family,
        case.task_case_id,
        work.runtime.spec.arm_id,
        work.repetition.repetition_index,
        prefix_before,
        state_before.state_hash,
    )
    started = _finite("selection monotonic start", clocks.monotonic_clock())
    selected_raw = _call_seam(
        "selector_failure", phase, lambda: selector.select(inputs)
    )
    completed = _finite("selection monotonic completion", clocks.monotonic_clock())
    if completed < started:
        _fail("selection monotonic clock moved backward")
    selected = SelectionResultEvidence.from_selection(selected_raw)
    _validate_selection(selected, inputs, work.runtime.spec.selector_mode)
    policy_hash = _domain_hash(
        "policy-decision-v1",
        {
            "feedback_prefix_event_ids": prefix_before,
            "learning_state_hash": state_before.state_hash,
            "selection_result": selected,
            "selector_config_hash": work.runtime.spec.selector_config_hash,
            "selector_input_hash": selector_input_hash,
            "selector_mode": work.runtime.spec.selector_mode,
            "selector_version": work.runtime.spec.selector_version,
        },
    )
    decision_id = _domain_hash(
        "selection-decision-v1",
        {
            "policy_decision_hash": policy_hash,
            "run_id": work.run_id,
            "task_case_id": case.task_case_id,
        },
    )
    trace_hash = _domain_hash("selection-trace-v1", selected.to_dict())
    decision = SelectionDecision(
        decision_id=decision_id,
        run_id=work.run_id,
        task_case_id=case.task_case_id,
        selected_context_item_ids=tuple(
            item.context_item_id for item in selected.selected_items
        ),
        selected_token_counts=tuple(
            item.token_count for item in selected.selected_items
        ),
        total_selected_tokens=selected.used_tokens,
        token_budget=selected.token_budget,
        selector_score=None,
        selector_input_hash=selector_input_hash,
        candidate_set_hash=candidate_set_hash,
        decision_latency_ms=(completed - started) * 1000.0,
        ranking_artifact_reference="inline:selection:" + trace_hash,
        ranking_artifact_hash=trace_hash,
        trace_artifact_reference="inline:selection-trace:" + trace_hash,
        trace_artifact_hash=trace_hash,
        decided_timestamp=clocks.utc_clock(),
        provenance=plan.provenance,
    )
    _trace(
        trace,
        phase.upper(),
        "selected",
        family,
        case.task_case_id,
        work.runtime.spec.arm_id,
        work.repetition.repetition_index,
        prefix_before,
        state_before.state_hash,
    )
    request = _call_seam(
        "renderer_failure",
        phase,
        lambda: renderer.render(inputs, selected.selected_items),
    )
    if (
        type(request) is not ProviderRequest
        or request.prompt_template_hash != plan.prompt_template_hash
    ):
        _fail("renderer returned an incompatible ProviderRequest")
    validate_request_manifest(work.manifest, work.provider, request)
    _trace(
        trace,
        phase.upper(),
        "rendered",
        family,
        case.task_case_id,
        work.runtime.spec.arm_id,
        work.repetition.repetition_index,
        prefix_before,
        state_before.state_hash,
    )
    execution = _call_seam(
        "provider_failure", phase, lambda: work.provider.execute(request)
    )
    validate_execution(work.manifest, work.provider, request, execution)
    _trace(
        trace,
        phase.upper(),
        "executed",
        family,
        case.task_case_id,
        work.runtime.spec.arm_id,
        work.repetition.repetition_index,
        prefix_before,
        state_before.state_hash,
    )
    blind_material = _domain_hash(
        "blind-output-material-v1",
        {
            "request_hash": request.request_hash,
            "response_hash": execution.raw_response_hash,
            "task_prompt": inputs.task_prompt,
        },
    )
    output_id = clocks.blinding_id(blind_material)
    blind_request = BlindAssessmentRequest(
        output_id,
        inputs.task_prompt,
        execution.response_text,
        case.sealed_evaluation,
        spec,
    )
    if any(arm.arm_id.casefold() in output_id.casefold() for arm in plan.arms):
        _fail("blinding output ID contains an arm ID")
    assessment = _call_seam(
        "assessment_failure", phase, lambda: assessor.assess(blind_request)
    )
    if type(assessment) is not BlindedAssessment:
        _fail("assessor must return exact BlindedAssessment")
    assessment = BlindedAssessment.from_dict(assessment.to_dict())
    if (
        assessment.output_id != output_id
        or assessment.rubric_id != case.sealed_evaluation.scoring_rubric.rubric_id
        or assessment.spec_id != spec.spec_id
        or assessment.spec_version != spec.spec_version
    ):
        _fail("assessment identity mismatch")
    scoring = _call_seam(
        "scoring_failure",
        phase,
        lambda: score_assessment(
            case.sealed_evaluation.scoring_rubric, spec, assessment
        ),
    )
    _trace(
        trace,
        phase.upper(),
        "assessed",
        family,
        case.task_case_id,
        work.runtime.spec.arm_id,
        work.repetition.repetition_index,
        prefix_before,
        state_before.state_hash,
    )
    outcome = _outcome(work.run_id, case, decision, execution, scoring, plan.provenance)
    if outcome.outcome_id in work.appended_outcome_ids:
        _fail("outcome was already appended")
    work.appended_outcome_ids.add(outcome.outcome_id)
    _trace(
        trace,
        phase.upper(),
        "outcome_appended" if phase == "adaptation" else "evaluation_outcome_appended",
        family,
        case.task_case_id,
        work.runtime.spec.arm_id,
        work.repetition.repetition_index,
        prefix_before,
        state_before.state_hash,
    )

    revealed_events: Tuple[FeedbackEvent, ...] = ()
    feedback = None
    state_after = state_before
    prefix_after = prefix_before
    if phase == "adaptation":
        slot_id = _domain_hash(
            "adaptation-slot-v1",
            {"case_id": case.task_case_id, "run_id": work.run_id, "sequence": sequence},
        )
        receipt = OutcomeAppendedReceipt._mint(
            _RECEIPT_TOKEN,
            dataset,
            slot_id,
            outcome.outcome_id,
            family,
            case.task_case_id,
            ordinal,
        )
        feedback = _call_seam(
            "feedback_reveal_failure",
            phase,
            lambda: dataset.reveal_feedback(receipt),
        )
        if type(feedback) is not FeedbackEvent:
            _fail("dataset must reveal exact FeedbackEvent")
        feedback = FeedbackEvent.from_dict(feedback.to_dict())
        if (
            feedback.task_case_id != case.task_case_id
            or feedback.task_family_id != family
        ):
            _fail("revealed feedback does not match current case")
        if feedback.event_id in prefix_before:
            _fail("feedback event IDs must be unique within a prefix")
        work.prefixes[family].append(feedback.event_id)
        work.inputs[family][case.task_case_id] = inputs
        prefix_after = tuple(work.prefixes[family])
        revealed_events = (feedback,)
        _trace(
            trace,
            "ADAPTATION",
            "feedback_revealed",
            family,
            case.task_case_id,
            work.runtime.spec.arm_id,
            work.repetition.repetition_index,
            prefix_after,
            state_before.state_hash,
        )
        work.events[family].append(feedback)
        snapshot = _call_seam(
            "learning_failure",
            phase,
            lambda: learn_utilities(
                tuple(work.events[family]),
                work.inputs[family],
                plan.learning_policy,
                clocks.learning_clock,
            ),
        )
        work.snapshots[family] = snapshot
        state_after = LearningStateEvidence.from_snapshot(
            family, prefix_after, snapshot
        )
        work.evidence[family] = state_after
        _trace(
            trace,
            "ADAPTATION",
            "learner_updated",
            family,
            case.task_case_id,
            work.runtime.spec.arm_id,
            work.repetition.repetition_index,
            prefix_after,
            state_after.state_hash,
        )

    record = OrderedTaskRecord(
        sequence,
        phase,
        family,
        case.task_case_id,
        ordinal,
        work.runtime.spec.arm_id,
        work.repetition.repetition_index,
        material,
        selector_input_hash,
        candidate_set_hash,
        prefix_before,
        state_before,
        selected,
        decision,
        policy_hash,
        request,
        execution,
        scoring,
        outcome,
        revealed_events,
        prefix_after,
        state_after,
    )
    return record, feedback


def run_ordered_experiment(
    plan: ExperimentPlan,
    dataset: OrderedDatasetSource,
    arms: Sequence[ArmRuntime],
    provider_factory: Callable[[RepetitionSpec], Provider],
    renderer: PromptRenderer,
    assessor: OutcomeAssessor,
    clocks: RunnerClocks,
) -> OrderedExperimentArtifact:
    """Execute a complete deterministic ordered experiment or raise a sanitized error."""

    stage = "preflight"
    try:
        if type(plan) is not ExperimentPlan:
            _fail("plan must be exact ExperimentPlan")
        plan = ExperimentPlan.from_dict(plan.to_dict())
        if type(arms) not in (list, tuple):
            _fail("arms must be an exact sequence")
        runtimes = tuple(arms)
        if len(runtimes) != len(plan.arms) or any(
            type(item) is not ArmRuntime for item in runtimes
        ):
            _fail("runtime arms must correspond exactly to plan arms")
        if tuple(item.spec for item in runtimes) != plan.arms:
            _fail("runtime arm order/specs must match plan")
        if (
            not callable(provider_factory)
            or type(renderer) is not CanonicalPromptRenderer
            or not callable(getattr(assessor, "assess", None))
            or type(clocks) is not RunnerClocks
        ):
            _fail("runtime seams are invalid")
        if (
            renderer.template_hash != plan.prompt_template_hash
            or renderer.template_spec != plan.prompt_template
        ):
            _fail("renderer template hash must match plan")
        if (
            dataset.dataset_version != plan.dataset_version
            or dataset.dataset_hash != plan.dataset_hash
            or tuple(dataset.family_order) != plan.family_order
        ):
            _fail("dataset identity/order must match plan")
        adaptation_ids = {
            family: tuple(dataset.adaptation_case_ids(family))
            for family in plan.family_order
        }
        if any(not values for values in adaptation_ids.values()):
            _fail("every family requires adaptation cases")
        total_adaptation_cases = sum(len(values) for values in adaptation_ids.values())
        if total_adaptation_cases > _MAX_CASES:
            _fail("case count exceeds preflight bound")
        expected_slots = total_adaptation_cases * len(plan.arms) * len(plan.repetitions)
        if expected_slots > _MAX_SLOTS:
            _fail("slot count exceeds preflight bound")
        estimated_trace = (
            expected_slots * 9
            + _MAX_CASES * len(plan.arms) * len(plan.repetitions) * 7
            + 16
        )
        if estimated_trace > _MAX_TRACE:
            _fail("trace count exceeds preflight bound")

        stage = "runtime_initialization"
        trace = _TraceBuffer([], clocks.utc_clock)
        _trace(trace, "ADAPTATION", "adaptation_started")
        work_by_pair: Dict[Tuple[str, int], _RunWork] = {}
        empty_snapshots: Dict[str, LearningSnapshot] = {}
        for family in plan.family_order:
            empty_snapshots[family] = learn_utilities(
                [], {}, plan.learning_policy, clocks.learning_clock
            )
        binding_request = ProviderRequest(
            _canonical(
                {
                    "format": "ordered-manifest-binding-v1",
                    "template_hash": renderer.template_hash,
                }
            ).decode("utf-8"),
            renderer.template_hash,
        )
        manifests_by_rep: Dict[int, list] = {}
        for runtime in runtimes:
            for repetition in plan.repetitions:
                provider = provider_factory(
                    RepetitionSpec.from_dict(repetition.to_dict())
                )
                config = getattr(provider, "configuration", None)
                if type(config) is not ProviderConfiguration:
                    _fail("provider must expose exact ProviderConfiguration")
                if not config.seed_supported or config.seed != repetition.provider_seed:
                    _fail("provider seed must match repetition provider_seed")
                run_id = _domain_hash(
                    "ordered-run-id-v1",
                    {
                        "arm_hash": runtime.spec.arm_hash,
                        "plan_hash": plan.plan_hash,
                        "provider_seed": repetition.provider_seed,
                        "repetition_index": repetition.repetition_index,
                    },
                )
                manifest = build_run_manifest(
                    ManifestInputs(
                        run_id,
                        plan.experiment_version,
                        plan.protocol_version,
                        plan.dataset_version,
                        plan.dataset_hash,
                        runtime.spec.selector_mode,
                        runtime.spec.selector_version,
                        plan.code_revision,
                        plan.provenance,
                    ),
                    provider,
                    binding_request,
                    clocks.utc_clock,
                )
                prefixes = {family: [] for family in plan.family_order}
                snapshots = {
                    family: empty_snapshots[family] for family in plan.family_order
                }
                evidence = {
                    family: LearningStateEvidence.from_snapshot(
                        family, (), snapshots[family]
                    )
                    for family in plan.family_order
                }
                work = _RunWork(
                    runtime,
                    repetition,
                    run_id,
                    provider,
                    manifest,
                    [],
                    prefixes,
                    {family: {} for family in plan.family_order},
                    snapshots,
                    evidence,
                    {family: [] for family in plan.family_order},
                    set(),
                    [],
                )
                work_by_pair[(runtime.spec.arm_id, repetition.repetition_index)] = work
                manifests_by_rep.setdefault(repetition.repetition_index, []).append(
                    manifest
                )
        for manifests in manifests_by_rep.values():
            for other in manifests[1:]:
                compare_manifests(manifests[0], other)

        stage = "adaptation"
        sequence = 0
        common_hashes: Dict[Tuple[str, str, int], Tuple[str, str, int, str]] = {}
        for family in plan.family_order:
            for ordinal, expected_case_id in enumerate(adaptation_ids[family]):
                material = dataset.load_adaptation_case(family, ordinal)
                material = CaseExecutionMaterial.from_dict(material.to_dict())
                if (
                    material.task_case.task_case_id != expected_case_id
                    or material.task_case.split != "adaptation"
                    or material.task_case.inputs.profile.task_family_id != family
                ):
                    _fail("loaded adaptation material violates declared order")
                for arm_id, repetition_index in _schedule(
                    plan, family, expected_case_id
                ):
                    work = work_by_pair[(arm_id, repetition_index)]
                    record, _feedback = _execute_slot(
                        plan=plan,
                        dataset=dataset,
                        work=work,
                        material=material,
                        phase="adaptation",
                        ordinal=ordinal,
                        sequence=sequence,
                        renderer=renderer,
                        assessor=assessor,
                        clocks=clocks,
                        trace=trace,
                    )
                    fairness = (
                        record.selector_input_hash,
                        record.candidate_set_hash,
                        record.selection_decision.token_budget,
                        record.scoring_result.spec_hash,
                    )
                    key = (family, expected_case_id, repetition_index)
                    if key in common_hashes and common_hashes[key] != fairness:
                        _fail("fair-arm visible input/budget/scoring invariant failed")
                    common_hashes[key] = fairness
                    work.tasks.append(record)
                    sequence += 1
        if sequence != expected_slots:
            _fail("adaptation slot count mismatch")
        _trace(trace, "ADAPTATION_COMPLETE", "adaptation_completed")

        stage = "evaluation_open"
        gate = EvaluationGate._mint(
            _GATE_TOKEN, dataset, expected_slots, sequence, plan.plan_hash
        )
        evaluation = dataset.open_evaluation(gate)
        if type(evaluation) is not EvaluationBatch:
            _fail("dataset must return exact EvaluationBatch")
        evaluation = EvaluationBatch.from_dict(evaluation.to_dict())
        _trace(trace, "EVALUATION_OPEN", "evaluation_opened")
        _trace(trace, "EVALUATION", "evaluation_started")
        by_family = {
            item.task_case.inputs.profile.task_family_id: item
            for item in evaluation.materials
        }
        if tuple(by_family) != plan.family_order:
            _fail("evaluation batch family order must match plan")

        stage = "evaluation"
        for family in plan.family_order:
            material = by_family[family]
            case_id = material.task_case.task_case_id
            for arm_id, repetition_index in _schedule(plan, family, case_id):
                work = work_by_pair[(arm_id, repetition_index)]
                before = work.evidence[family]
                record, feedback = _execute_slot(
                    plan=plan,
                    dataset=dataset,
                    work=work,
                    material=material,
                    phase="evaluation",
                    ordinal=0,
                    sequence=sequence,
                    renderer=renderer,
                    assessor=assessor,
                    clocks=clocks,
                    trace=trace,
                )
                if feedback is not None or record.learning_state_after != before:
                    _fail("evaluation must be learning-state read-only")
                work.tasks.append(record)
                sequence += 1
        _trace(trace, "EVALUATION", "evaluation_completed")

        stage = "complete"
        _trace(trace, "COMPLETE", "experiment_completed")
        completed_timestamp = clocks.utc_clock()
        arm_runs = []
        for runtime in runtimes:
            for repetition in plan.repetitions:
                work = work_by_pair[(runtime.spec.arm_id, repetition.repetition_index)]
                arm_runs.append(
                    ArmRunRecord(
                        work.run_id,
                        runtime.spec.arm_id,
                        repetition.repetition_index,
                        repetition.provider_seed,
                        work.manifest,
                        tuple(work.tasks),
                        tuple(work.evidence[family] for family in plan.family_order),
                        completed_timestamp,
                    )
                )
        artifact = OrderedExperimentArtifact._derive(
            _ARTIFACT_TOKEN, plan, tuple(arm_runs), tuple(trace), completed_timestamp
        )
        return OrderedExperimentArtifact.from_dict(artifact.to_dict())
    except RunnerError:
        raise
    except Exception as exc:
        category = (
            "validation_failure"
            if isinstance(exc, (TypeError, ValueError))
            else "execution_failure"
        )
        error_type = (
            RunnerValidationError if category == "validation_failure" else RunnerError
        )
        raise error_type(category, stage) from exc


__all__ = [
    "ARTIFACT_VERSION",
    "RUNNER_RECORD_VERSION",
    "ArmRunRecord",
    "ArmRuntime",
    "ArmSpec",
    "BlindAssessmentRequest",
    "CanonicalPromptRenderer",
    "CaseExecutionMaterial",
    "EvaluationBatch",
    "EvaluationGate",
    "ExperimentPlan",
    "LearningStateEvidence",
    "OrderedDatasetSource",
    "OrderedExperimentArtifact",
    "OrderedTaskRecord",
    "OutcomeAppendedReceipt",
    "PhaseTraceEvent",
    "PromptTemplateSpec",
    "PromptRenderer",
    "RepetitionSpec",
    "RunnerClocks",
    "RunnerError",
    "RunnerValidationError",
    "SelectionDecisionEvidence",
    "SelectionResultEvidence",
    "Stage0OrderedDatasetSource",
    "run_ordered_experiment",
]
