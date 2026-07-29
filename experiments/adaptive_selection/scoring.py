"""Deterministic aggregation of frozen, blinded semantic assessments.

This module intentionally accepts assessment evidence only.  It is not a prose judge and
has no access to selector conditions, retrieval measurements, context labels, or runtime
telemetry.  Semantic judgments must be made upstream and frozen as ``BlindedAssessment``.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from decimal import Context, Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
import hashlib
import json
import re
from types import MappingProxyType
from typing import Any, Dict, Optional, Tuple, Type, TypeVar, cast

from .schema import CriterionScore, ScoringRubric

T = TypeVar("T", bound="CanonicalRecord")
_DECIMAL_RE = re.compile(r"^(?:0|[1-9][0-9]*)(?:\.[0-9]*[1-9])?$")
_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_FORBIDDEN_EVIDENCE_KEYS = frozenset(
    {
        "condition",
        "condition_label",
        "context_labels",
        "filename",
        "latency",
        "ndcg",
        "retrieval_precision",
        "run_order",
        "selected_context_item_ids",
        "selected_ids",
        "selection_decision",
        "selector_mode",
        "tokens",
        "trace",
        "traces",
    }
)


def _nonempty(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{} must be nonempty".format(name))


def _identifier(name: str, value: Any) -> None:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise ValueError("{} must be a stable nonempty ID".format(name))


def _timestamp(name: str, value: Any) -> None:
    if not isinstance(value, str) or not _TIMESTAMP_RE.fullmatch(value):
        raise ValueError("{} must be canonical UTC RFC 3339 ending in Z".format(name))
    from datetime import datetime

    try:
        datetime.strptime(
            value, "%Y-%m-%dT%H:%M:%S.%fZ" if "." in value else "%Y-%m-%dT%H:%M:%SZ"
        )
    except ValueError:
        raise ValueError("{} must be canonical UTC RFC 3339 ending in Z".format(name))


def _decimal(name: str, value: Any, positive: bool = False) -> Decimal:
    if not isinstance(value, str) or not _DECIMAL_RE.fullmatch(value):
        qualifier = "positive " if positive else "nonnegative "
        raise ValueError(
            "{} must be a canonical {}decimal string".format(name, qualifier)
        )
    try:
        parsed = Decimal(value)
    except InvalidOperation:
        raise ValueError("{} must be a canonical decimal string".format(name))
    if not parsed.is_finite() or parsed < 0 or (positive and parsed <= 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError("{} must be {}".format(name, qualifier))
    return parsed


def _cap(value: Optional[str]) -> None:
    if value is None:
        return
    parsed = _decimal("quality_cap", value)
    if parsed > 1:
        raise ValueError("quality_cap must be between 0 and 1")


def _freeze_json(value: Any, path: str) -> Any:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        import math

        if not math.isfinite(value):
            raise ValueError("{} must be finite".format(path))
        return value
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ValueError("{} object keys must be strings".format(path))
        result = {}
        for key in sorted(value):
            result[key] = _freeze_json(value[key], "{}.{}".format(path, key))
        return MappingProxyType(result)
    if type(value) in (tuple, list):
        return tuple(
            _freeze_json(item, "{}[{}]".format(path, index))
            for index, item in enumerate(value)
        )
    raise ValueError(
        "{} contains unsupported JSON value: {}".format(path, type(value).__name__)
    )


def _freeze_evidence(value: Any) -> Any:
    frozen = _freeze_json(value, "supporting_evidence")

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                normalized = "_".join(
                    part for part in re.split(r"[^a-z0-9]+", key.casefold()) if part
                )
                if normalized in _FORBIDDEN_EVIDENCE_KEYS:
                    raise ValueError(
                        "supporting_evidence must not contain selector, condition, or retrieval fields"
                    )
                visit(child)
        elif isinstance(item, tuple):
            for child in item:
                visit(child)

    visit(frozen)
    return frozen


def _serialize(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _serialize(getattr(value, item.name)) for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {key: _serialize(value[key]) for key in sorted(value)}
    if type(value) in (tuple, list):
        return [_serialize(item) for item in value]
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise TypeError("non-serializable value: {}".format(type(value).__name__))


def _mapping(data: Any) -> Mapping:
    if not isinstance(data, Mapping):
        raise ValueError("payload must be an object")
    return data


def _sequence(data: Mapping, key: str) -> Tuple[Any, ...]:
    if key not in data:
        raise ValueError("{} is required".format(key))
    value = data[key]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("{} must be a sequence".format(key))
    return tuple(value)


def _hash(payload: Mapping) -> str:
    encoded = json.dumps(
        _serialize(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _decimal_text(value: Decimal) -> str:
    if value == 0:
        return "0"
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


class CanonicalRecord:
    def to_dict(self) -> Dict[str, Any]:
        return cast(Dict[str, Any], _serialize(self))

    @classmethod
    def _flat(cls: Type[T], data: Mapping, **changes: Any) -> T:
        values = dict(_mapping(data))
        values.update(changes)
        try:
            return cls(**values)
        except TypeError as error:
            raise ValueError("invalid {} payload: {}".format(cls.__name__, error))


@dataclass(frozen=True)
class RequiredStepSpec(CanonicalRecord):
    step_id: str
    criterion_id: str
    positive_points: str
    critical_omission: bool
    quality_cap: Optional[str] = None

    def __post_init__(self) -> None:
        _identifier("step_id", self.step_id)
        _identifier("criterion_id", self.criterion_id)
        _decimal("positive_points", self.positive_points, positive=True)
        if type(self.critical_omission) is not bool:
            raise ValueError("critical_omission must be a boolean")
        _cap(self.quality_cap)

    @classmethod
    def from_dict(cls, data: Mapping) -> "RequiredStepSpec":
        return cls._flat(data)


@dataclass(frozen=True)
class NegativeFindingSpec(CanonicalRecord):
    finding_id: str
    kind: str
    criterion_id: str
    deduction: str
    severity: str
    quality_cap: Optional[str] = None

    def __post_init__(self) -> None:
        _identifier("finding_id", self.finding_id)
        _identifier("criterion_id", self.criterion_id)
        if self.kind not in {"false_claim", "prohibited_action"}:
            raise ValueError("kind must be false_claim or prohibited_action")
        _decimal("deduction", self.deduction)
        if self.severity not in {"minor", "major", "severe", "critical"}:
            raise ValueError("severity must be minor, major, severe, or critical")
        _cap(self.quality_cap)

    @classmethod
    def from_dict(cls, data: Mapping) -> "NegativeFindingSpec":
        return cls._flat(data)


@dataclass(frozen=True)
class TaskScoringSpec(CanonicalRecord):
    spec_id: str
    spec_version: str
    rubric_id: str
    expected_criterion_ids: Tuple[str, ...]
    required_steps: Tuple[RequiredStepSpec, ...]
    negative_findings: Tuple[NegativeFindingSpec, ...]
    scorer_use: str
    engine_version: str
    normalization_version: str
    rule_version: str
    decimal_precision: int
    decimal_version: str
    provenance: str

    def __post_init__(self) -> None:
        for name in ("spec_id", "spec_version", "rubric_id"):
            _identifier(name, getattr(self, name))
        for name in (
            "engine_version",
            "normalization_version",
            "rule_version",
            "decimal_version",
            "provenance",
        ):
            _nonempty(name, getattr(self, name))
        object.__setattr__(
            self, "expected_criterion_ids", tuple(self.expected_criterion_ids)
        )
        object.__setattr__(self, "required_steps", tuple(self.required_steps))
        object.__setattr__(self, "negative_findings", tuple(self.negative_findings))
        if not self.expected_criterion_ids or len(
            set(self.expected_criterion_ids)
        ) != len(self.expected_criterion_ids):
            raise ValueError("expected criterion IDs must be nonempty and unique")
        for criterion_id in self.expected_criterion_ids:
            _identifier("expected criterion ID", criterion_id)
        if not all(isinstance(item, RequiredStepSpec) for item in self.required_steps):
            raise ValueError("required_steps must contain RequiredStepSpec records")
        if not all(
            isinstance(item, NegativeFindingSpec) for item in self.negative_findings
        ):
            raise ValueError(
                "negative_findings must contain NegativeFindingSpec records"
            )
        step_ids = tuple(item.step_id for item in self.required_steps)
        finding_ids = tuple(item.finding_id for item in self.negative_findings)
        if len(set(step_ids)) != len(step_ids):
            raise ValueError("step IDs must be unique")
        if len(set(finding_ids)) != len(finding_ids):
            raise ValueError("finding IDs must be unique")
        expected = set(self.expected_criterion_ids)
        if any(item.criterion_id not in expected for item in self.required_steps):
            raise ValueError("required step criterion_id must be expected")
        if any(item.criterion_id not in expected for item in self.negative_findings):
            raise ValueError("finding criterion_id must be expected")
        if any(
            not any(step.criterion_id == criterion_id for step in self.required_steps)
            for criterion_id in self.expected_criterion_ids
        ):
            raise ValueError("every criterion must have a positive required step")
        if self.scorer_use not in {
            "fixture_only",
            "human_annotated",
            "validated_automatic",
        }:
            raise ValueError("scorer_use is invalid")
        if (
            not isinstance(self.decimal_precision, int)
            or isinstance(self.decimal_precision, bool)
            or self.decimal_precision <= 0
        ):
            raise ValueError("decimal_precision must be a positive integer")

    @classmethod
    def from_dict(cls, data: Mapping) -> "TaskScoringSpec":
        data = _mapping(data)
        return cls._flat(
            data,
            expected_criterion_ids=_sequence(data, "expected_criterion_ids"),
            required_steps=tuple(
                RequiredStepSpec.from_dict(item)
                for item in _sequence(data, "required_steps")
            ),
            negative_findings=tuple(
                NegativeFindingSpec.from_dict(item)
                for item in _sequence(data, "negative_findings")
            ),
        )


@dataclass(frozen=True)
class StepAssessment(CanonicalRecord):
    step_id: str
    status: str
    supporting_evidence: Mapping[str, Any]

    def __post_init__(self) -> None:
        _identifier("step_id", self.step_id)
        if self.status not in {"met", "not_met", "contradicted", "unresolved"}:
            raise ValueError("step status is invalid")
        if not isinstance(self.supporting_evidence, Mapping):
            raise ValueError("supporting_evidence must be an object")
        frozen = _freeze_evidence(self.supporting_evidence)
        if self.status in {"met", "contradicted"} and not frozen:
            raise ValueError(
                "supporting_evidence must be nonempty for met/contradicted"
            )
        object.__setattr__(self, "supporting_evidence", frozen)

    @classmethod
    def from_dict(cls, data: Mapping) -> "StepAssessment":
        return cls._flat(data)


@dataclass(frozen=True)
class FindingAssessment(CanonicalRecord):
    finding_id: str
    status: str
    supporting_evidence: Mapping[str, Any]

    def __post_init__(self) -> None:
        _identifier("finding_id", self.finding_id)
        if self.status not in {"present", "absent", "unresolved"}:
            raise ValueError("finding status is invalid")
        if not isinstance(self.supporting_evidence, Mapping):
            raise ValueError("supporting_evidence must be an object")
        frozen = _freeze_evidence(self.supporting_evidence)
        if self.status == "present" and not frozen:
            raise ValueError("supporting_evidence must be nonempty for present")
        object.__setattr__(self, "supporting_evidence", frozen)

    @classmethod
    def from_dict(cls, data: Mapping) -> "FindingAssessment":
        return cls._flat(data)


@dataclass(frozen=True)
class CorrectionAssessment(CanonicalRecord):
    correction_id: str
    description: str
    supporting_evidence: Mapping[str, Any]

    def __post_init__(self) -> None:
        _identifier("correction_id", self.correction_id)
        _nonempty("description", self.description)
        if not isinstance(self.supporting_evidence, Mapping):
            raise ValueError("supporting_evidence must be an object")
        frozen = _freeze_evidence(self.supporting_evidence)
        if not frozen:
            raise ValueError("supporting_evidence must be nonempty")
        object.__setattr__(self, "supporting_evidence", frozen)

    @classmethod
    def from_dict(cls, data: Mapping) -> "CorrectionAssessment":
        return cls._flat(data)


@dataclass(frozen=True)
class BlindedAssessment(CanonicalRecord):
    output_id: str
    rubric_id: str
    spec_id: str
    spec_version: str
    step_assessments: Tuple[StepAssessment, ...]
    finding_assessments: Tuple[FindingAssessment, ...]
    corrections: Tuple[CorrectionAssessment, ...]
    rater_id: str
    rater_version: str
    assessment_timestamp: str
    provenance: str

    def __post_init__(self) -> None:
        for name in ("output_id", "rubric_id", "spec_id", "spec_version", "rater_id"):
            _identifier(name, getattr(self, name))
        for name in ("rater_version", "provenance"):
            _nonempty(name, getattr(self, name))
        _timestamp("assessment_timestamp", self.assessment_timestamp)
        object.__setattr__(self, "step_assessments", tuple(self.step_assessments))
        object.__setattr__(self, "finding_assessments", tuple(self.finding_assessments))
        object.__setattr__(self, "corrections", tuple(self.corrections))
        if not all(isinstance(item, StepAssessment) for item in self.step_assessments):
            raise ValueError("step_assessments must contain StepAssessment records")
        if not all(
            isinstance(item, FindingAssessment) for item in self.finding_assessments
        ):
            raise ValueError(
                "finding_assessments must contain FindingAssessment records"
            )
        if not all(isinstance(item, CorrectionAssessment) for item in self.corrections):
            raise ValueError("corrections must contain CorrectionAssessment records")
        correction_ids = tuple(item.correction_id for item in self.corrections)
        if len(set(correction_ids)) != len(correction_ids):
            raise ValueError("correction IDs must be unique")

    @classmethod
    def from_dict(cls, data: Mapping) -> "BlindedAssessment":
        data = _mapping(data)
        return cls._flat(
            data,
            step_assessments=tuple(
                StepAssessment.from_dict(item)
                for item in _sequence(data, "step_assessments")
            ),
            finding_assessments=tuple(
                FindingAssessment.from_dict(item)
                for item in _sequence(data, "finding_assessments")
            ),
            corrections=tuple(
                CorrectionAssessment.from_dict(item)
                for item in _sequence(data, "corrections")
            ),
        )


@dataclass(frozen=True)
class RuleEffect(CanonicalRecord):
    rule_id: str
    criterion_id: str
    effect: str
    value: str
    reason: str

    def __post_init__(self) -> None:
        _identifier("rule_id", self.rule_id)
        _identifier("criterion_id", self.criterion_id)
        if self.effect not in {"deduction", "quality_cap"}:
            raise ValueError("effect must be deduction or quality_cap")
        parsed = _decimal("value", self.value)
        if self.effect == "quality_cap" and parsed > 1:
            raise ValueError("quality cap effect must be between 0 and 1")
        _nonempty("reason", self.reason)

    @classmethod
    def from_dict(cls, data: Mapping) -> "RuleEffect":
        return cls._flat(data)


@dataclass(frozen=True)
class CriterionArithmetic(CanonicalRecord):
    criterion_id: str
    met_step_ids: Tuple[str, ...]
    no_credit_step_ids: Tuple[str, ...]
    present_finding_ids: Tuple[str, ...]
    max_points: str
    met_points: str
    deduction_points: str
    raw_points: str
    normalized: str

    def __post_init__(self) -> None:
        _identifier("criterion_id", self.criterion_id)
        for name in ("met_step_ids", "no_credit_step_ids", "present_finding_ids"):
            value = tuple(getattr(self, name))
            if len(set(value)) != len(value):
                raise ValueError("{} must be unique".format(name))
            for rule_id in value:
                _identifier(name, rule_id)
            object.__setattr__(self, name, value)
        maximum = _decimal("max_points", self.max_points, positive=True)
        met = _decimal("met_points", self.met_points)
        deduction = _decimal("deduction_points", self.deduction_points)
        raw = _decimal("raw_points", self.raw_points)
        normalized = _decimal("normalized", self.normalized)
        if raw != max(Decimal("0"), min(maximum, met - deduction)):
            raise ValueError(
                "raw_points must equal clamped met_points minus deductions"
            )
        if normalized > 1:
            raise ValueError("normalized must be between 0 and 1")

    @classmethod
    def from_dict(cls, data: Mapping) -> "CriterionArithmetic":
        data = _mapping(data)
        return cls._flat(
            data,
            met_step_ids=_sequence(data, "met_step_ids"),
            no_credit_step_ids=_sequence(data, "no_credit_step_ids"),
            present_finding_ids=_sequence(data, "present_finding_ids"),
        )


@dataclass(frozen=True)
class ScoringResult(CanonicalRecord):
    status: str
    rubric_id: str
    spec_id: str
    spec_version: str
    spec_hash: str
    assessment_hash: str
    scorer_hash: str
    criterion_scores: Tuple[CriterionScore, ...]
    criterion_arithmetic: Tuple[CriterionArithmetic, ...]
    raw_score: Optional[float]
    max_score: Optional[float]
    normalized_score: Optional[float]
    base_quality: Optional[float]
    triggered_deductions: Tuple[RuleEffect, ...]
    triggered_caps: Tuple[RuleEffect, ...]
    correction_count: int
    severe_or_critical_event_ids: Tuple[str, ...]
    critical_omission_ids: Tuple[str, ...]
    unresolved_rule_ids: Tuple[str, ...]
    engine_version: str
    normalization_version: str
    rule_version: str
    decimal_precision: int
    decimal_version: str
    provenance: str

    def __post_init__(self) -> None:
        for name in (
            "criterion_scores",
            "criterion_arithmetic",
            "triggered_deductions",
            "triggered_caps",
            "severe_or_critical_event_ids",
            "critical_omission_ids",
            "unresolved_rule_ids",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in ("rubric_id", "spec_id", "spec_version"):
            _identifier(name, getattr(self, name))
        for name in (
            "engine_version",
            "normalization_version",
            "rule_version",
            "decimal_version",
            "provenance",
        ):
            _nonempty(name, getattr(self, name))
        for name in ("spec_hash", "assessment_hash", "scorer_hash"):
            value = getattr(self, name)
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError("{} must be a lowercase SHA-256 digest".format(name))
        if (
            not isinstance(self.correction_count, int)
            or isinstance(self.correction_count, bool)
            or self.correction_count < 0
        ):
            raise ValueError("correction_count must be a nonnegative integer")
        if (
            not isinstance(self.decimal_precision, int)
            or isinstance(self.decimal_precision, bool)
            or self.decimal_precision <= 0
        ):
            raise ValueError("decimal_precision must be a positive integer")
        if not all(isinstance(item, CriterionScore) for item in self.criterion_scores):
            raise ValueError("criterion_scores must contain CriterionScore records")
        if not all(
            isinstance(item, CriterionArithmetic) for item in self.criterion_arithmetic
        ):
            raise ValueError(
                "criterion_arithmetic must contain CriterionArithmetic records"
            )
        if not all(
            isinstance(item, RuleEffect)
            for item in self.triggered_deductions + self.triggered_caps
        ):
            raise ValueError("triggered effects must contain RuleEffect records")
        for name in (
            "severe_or_critical_event_ids",
            "critical_omission_ids",
            "unresolved_rule_ids",
        ):
            values = getattr(self, name)
            if len(set(values)) != len(values):
                raise ValueError("{} must be unique".format(name))
            for rule_id in values:
                _identifier(name, rule_id)
        if self.status not in {"scored", "needs_adjudication"}:
            raise ValueError("status is invalid")
        if self.status == "needs_adjudication":
            if any(
                value is not None
                for value in (
                    self.raw_score,
                    self.max_score,
                    self.normalized_score,
                    self.base_quality,
                )
            ):
                raise ValueError("adjudication results must not contain scores")
            if self.criterion_scores or self.criterion_arithmetic:
                raise ValueError(
                    "adjudication results must not contain criterion scores"
                )
            if self.triggered_deductions or self.triggered_caps:
                raise ValueError(
                    "adjudication results must not interpret deductions or caps"
                )
            if self.severe_or_critical_event_ids or self.critical_omission_ids:
                raise ValueError("adjudication results must not interpret event rules")
            if not self.unresolved_rule_ids:
                raise ValueError("adjudication results require unresolved_rule_ids")
        else:
            import math

            numeric = (
                self.raw_score,
                self.max_score,
                self.normalized_score,
                self.base_quality,
            )
            if any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                for value in numeric
            ):
                raise ValueError("scored results require finite numeric scores")
            maximum = cast(float, self.max_score)
            raw = cast(float, self.raw_score)
            normalized = cast(float, self.normalized_score)
            base = cast(float, self.base_quality)
            if (
                maximum <= 0
                or not 0 <= raw <= maximum
                or not 0 <= normalized <= base <= 1
            ):
                raise ValueError("scored result values are out of range")
            if self.unresolved_rule_ids:
                raise ValueError("scored results must not contain unresolved rules")
            if len(self.criterion_scores) != len(self.criterion_arithmetic) or tuple(
                item.criterion_id for item in self.criterion_scores
            ) != tuple(item.criterion_id for item in self.criterion_arithmetic):
                raise ValueError("criterion score and arithmetic IDs/order must match")
            context = Context(prec=self.decimal_precision, rounding=ROUND_HALF_EVEN)
            with localcontext(context):
                for item in self.criterion_arithmetic:
                    if _decimal("normalized", item.normalized) != _decimal(
                        "raw_points", item.raw_points
                    ) / _decimal("max_points", item.max_points, positive=True):
                        raise ValueError(
                            "criterion normalized arithmetic is inconsistent"
                        )

    @classmethod
    def from_dict(cls, data: Mapping) -> "ScoringResult":
        data = _mapping(data)
        return cls._flat(
            data,
            criterion_scores=tuple(
                CriterionScore.from_dict(item)
                for item in _sequence(data, "criterion_scores")
            ),
            criterion_arithmetic=tuple(
                CriterionArithmetic.from_dict(item)
                for item in _sequence(data, "criterion_arithmetic")
            ),
            triggered_deductions=tuple(
                RuleEffect.from_dict(item)
                for item in _sequence(data, "triggered_deductions")
            ),
            triggered_caps=tuple(
                RuleEffect.from_dict(item) for item in _sequence(data, "triggered_caps")
            ),
            severe_or_critical_event_ids=_sequence(
                data, "severe_or_critical_event_ids"
            ),
            critical_omission_ids=_sequence(data, "critical_omission_ids"),
            unresolved_rule_ids=_sequence(data, "unresolved_rule_ids"),
        )


def _canonical_inputs(
    rubric: ScoringRubric, spec: TaskScoringSpec, assessment: BlindedAssessment
) -> Tuple[ScoringRubric, TaskScoringSpec, BlindedAssessment]:
    if not isinstance(rubric, ScoringRubric):
        raise ValueError("rubric must be a ScoringRubric")
    if not isinstance(spec, TaskScoringSpec):
        raise ValueError("spec must be a TaskScoringSpec")
    if not isinstance(assessment, BlindedAssessment):
        raise ValueError("assessment must be a BlindedAssessment")
    return (
        ScoringRubric.from_dict(rubric.to_dict()),
        TaskScoringSpec.from_dict(spec.to_dict()),
        BlindedAssessment.from_dict(assessment.to_dict()),
    )


def score_assessment(
    rubric: ScoringRubric, spec: TaskScoringSpec, assessment: BlindedAssessment
) -> ScoringResult:
    """Aggregate declared rules; never infer semantics from arbitrary response prose."""

    rubric, spec, assessment = _canonical_inputs(rubric, spec, assessment)
    rubric_ids = tuple(item.criterion_id for item in rubric.criteria)
    if rubric.rubric_id != spec.rubric_id or rubric_ids != spec.expected_criterion_ids:
        raise ValueError("rubric/spec criterion IDs/order must correspond exactly")
    if assessment.rubric_id != rubric.rubric_id:
        raise ValueError("assessment rubric_id must match rubric")
    if (assessment.spec_id, assessment.spec_version) != (
        spec.spec_id,
        spec.spec_version,
    ):
        raise ValueError("assessment spec identity must match spec")
    for criterion_id in rubric_ids:
        if not any(step.criterion_id == criterion_id for step in spec.required_steps):
            raise ValueError("every criterion must have a positive required step")

    step_ids = tuple(item.step_id for item in spec.required_steps)
    assessed_step_ids = tuple(item.step_id for item in assessment.step_assessments)
    if len(set(assessed_step_ids)) != len(assessed_step_ids) or set(
        assessed_step_ids
    ) != set(step_ids):
        raise ValueError("step assessment IDs must exactly match declared step IDs")
    finding_ids = tuple(item.finding_id for item in spec.negative_findings)
    assessed_finding_ids = tuple(
        item.finding_id for item in assessment.finding_assessments
    )
    if len(set(assessed_finding_ids)) != len(assessed_finding_ids) or set(
        assessed_finding_ids
    ) != set(finding_ids):
        raise ValueError(
            "finding assessment IDs must exactly match declared finding IDs"
        )

    step_by_id = {item.step_id: item for item in assessment.step_assessments}
    finding_by_id = {item.finding_id: item for item in assessment.finding_assessments}
    canonical_assessment = assessment.to_dict()
    canonical_assessment["step_assessments"] = [
        step_by_id[rule_id].to_dict() for rule_id in step_ids
    ]
    canonical_assessment["finding_assessments"] = [
        finding_by_id[rule_id].to_dict() for rule_id in finding_ids
    ]
    canonical_assessment["corrections"] = [
        item.to_dict()
        for item in sorted(assessment.corrections, key=lambda item: item.correction_id)
    ]
    spec_hash = _hash(spec.to_dict())
    assessment_hash = _hash(canonical_assessment)
    scorer_hash = _hash(
        {
            "engine_version": spec.engine_version,
            "spec_hash": spec_hash,
            "normalization_version": spec.normalization_version,
            "rule_version": spec.rule_version,
            "decimal_precision": spec.decimal_precision,
            "decimal_version": spec.decimal_version,
        }
    )
    common = dict(
        rubric_id=rubric.rubric_id,
        spec_id=spec.spec_id,
        spec_version=spec.spec_version,
        spec_hash=spec_hash,
        assessment_hash=assessment_hash,
        scorer_hash=scorer_hash,
        correction_count=len(assessment.corrections),
        engine_version=spec.engine_version,
        normalization_version=spec.normalization_version,
        rule_version=spec.rule_version,
        decimal_precision=spec.decimal_precision,
        decimal_version=spec.decimal_version,
        provenance=spec.provenance,
    )
    unresolved = tuple(
        [rule_id for rule_id in step_ids if step_by_id[rule_id].status == "unresolved"]
        + [
            rule_id
            for rule_id in finding_ids
            if finding_by_id[rule_id].status == "unresolved"
        ]
    )
    if unresolved:
        return ScoringResult(
            status="needs_adjudication",
            criterion_scores=(),
            criterion_arithmetic=(),
            raw_score=None,
            max_score=None,
            normalized_score=None,
            base_quality=None,
            triggered_deductions=(),
            triggered_caps=(),
            severe_or_critical_event_ids=(),
            critical_omission_ids=(),
            unresolved_rule_ids=unresolved,
            **common,
        )

    context = Context(prec=spec.decimal_precision, rounding=ROUND_HALF_EVEN)
    criterion_scores = []
    arithmetic = []
    deductions = []
    caps = []
    severe = []
    critical_omissions = []
    with localcontext(context):
        weighted = Decimal("0")
        total_weight = Decimal("0")
        for criterion in rubric.criteria:
            steps = tuple(
                item
                for item in spec.required_steps
                if item.criterion_id == criterion.criterion_id
            )
            findings = tuple(
                item
                for item in spec.negative_findings
                if item.criterion_id == criterion.criterion_id
            )
            maximum = sum(
                (_decimal("positive_points", item.positive_points) for item in steps),
                Decimal("0"),
            )
            met = sum(
                (
                    _decimal("positive_points", item.positive_points)
                    for item in steps
                    if step_by_id[item.step_id].status == "met"
                ),
                Decimal("0"),
            )
            deduction = sum(
                (
                    _decimal("deduction", item.deduction)
                    for item in findings
                    if finding_by_id[item.finding_id].status == "present"
                ),
                Decimal("0"),
            )
            raw = max(Decimal("0"), min(maximum, met - deduction))
            normalized = raw / maximum
            criterion_scores.append(
                CriterionScore(
                    criterion.criterion_id,
                    float(raw),
                    float(maximum),
                    float(normalized),
                )
            )
            arithmetic.append(
                CriterionArithmetic(
                    criterion.criterion_id,
                    tuple(
                        item.step_id
                        for item in steps
                        if step_by_id[item.step_id].status == "met"
                    ),
                    tuple(
                        item.step_id
                        for item in steps
                        if step_by_id[item.step_id].status != "met"
                    ),
                    tuple(
                        item.finding_id
                        for item in findings
                        if finding_by_id[item.finding_id].status == "present"
                    ),
                    _decimal_text(maximum),
                    _decimal_text(met),
                    _decimal_text(deduction),
                    _decimal_text(raw),
                    _decimal_text(normalized),
                )
            )
            weight = Decimal(str(criterion.weight))
            weighted += weight * normalized
            total_weight += weight

            for item in steps:
                status = step_by_id[item.step_id].status
                if item.critical_omission and status != "met":
                    critical_omissions.append(item.step_id)
                if item.quality_cap is not None and (
                    status == "contradicted"
                    or (item.critical_omission and status == "not_met")
                ):
                    caps.append(
                        RuleEffect(
                            item.step_id,
                            item.criterion_id,
                            "quality_cap",
                            item.quality_cap,
                            status,
                        )
                    )
            for item in findings:
                if finding_by_id[item.finding_id].status != "present":
                    continue
                deductions.append(
                    RuleEffect(
                        item.finding_id,
                        item.criterion_id,
                        "deduction",
                        item.deduction,
                        item.kind,
                    )
                )
                if item.quality_cap is not None:
                    caps.append(
                        RuleEffect(
                            item.finding_id,
                            item.criterion_id,
                            "quality_cap",
                            item.quality_cap,
                            item.kind,
                        )
                    )
                if item.severity in {"severe", "critical"}:
                    severe.append(item.finding_id)

        base = weighted / total_weight
        final = min([base] + [_decimal("quality_cap", item.value) for item in caps])
        raw_score = final * total_weight

    return ScoringResult(
        status="scored",
        criterion_scores=tuple(criterion_scores),
        criterion_arithmetic=tuple(arithmetic),
        raw_score=float(raw_score),
        max_score=float(total_weight),
        normalized_score=float(final),
        base_quality=float(base),
        triggered_deductions=tuple(deductions),
        triggered_caps=tuple(caps),
        severe_or_critical_event_ids=tuple(severe),
        critical_omission_ids=tuple(critical_omissions),
        unresolved_rule_ids=(),
        **common,
    )


__all__ = [
    "RequiredStepSpec",
    "NegativeFindingSpec",
    "TaskScoringSpec",
    "StepAssessment",
    "FindingAssessment",
    "CorrectionAssessment",
    "BlindedAssessment",
    "RuleEffect",
    "CriterionArithmetic",
    "ScoringResult",
    "score_assessment",
]
