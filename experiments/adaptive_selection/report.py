"""Deterministic paired reporting for sealed ordered experiment artifacts.

The module is deliberately a pure derivation boundary.  It reads no clock, environment,
database, provider, or fixture and makes no efficacy or significance claim.
"""

import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field, fields, is_dataclass
from decimal import ROUND_HALF_EVEN, Context, Decimal, InvalidOperation, localcontext
from fractions import Fraction
from types import MappingProxyType
from typing import Any, Dict, Mapping, NoReturn, Optional, Sequence, Tuple, cast

from .runner import OrderedExperimentArtifact, OrderedTaskRecord

REPORT_VERSION = "paired-report-v1"
INTERVAL_METHOD_VERSION = "paired-repetition-cluster-percentile-v1"
_AGGREGATION_VERSION = "family-balanced-repetition-v1"
_DECIMAL_VERSION = "decimal-half-even-v1"
_MAX_TEXT = 16_384
_MAX_DRAWS = 100_000
_MAX_RATES = 256
_MAX_RECORDS = 2_000_000
_MAX_BOOTSTRAP_OPERATIONS = 10_000_000
_DECIMAL_RE = re.compile(r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]*[1-9])?$")
_PRIMARY_ESTIMAND = "heldout-evaluation-adaptive-minus-primary-baseline-v1"


def _fail(message: str) -> NoReturn:
    raise ValueError(message)


def _text(name: str, value: Any, maximum: int = 256) -> str:
    if type(value) is not str or not value.strip() or len(value) > maximum:
        _fail("{} must be a nonempty bounded string".format(name))
    return cast(str, value)


def _integer(name: str, value: Any, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail("{} must be an integer >= {}".format(name, minimum))
    return cast(int, value)


def _decimal(name: str, value: Any, minimum: Optional[Decimal] = None) -> Decimal:
    if type(value) is not str or len(value) > 128 or not _DECIMAL_RE.fullmatch(value):
        _fail("{} must be canonical decimal text".format(name))
    try:
        result = Decimal(value)
    except InvalidOperation:
        _fail("{} must be canonical decimal text".format(name))
    if not result.is_finite() or (minimum is not None and result < minimum):
        _fail("{} is outside its allowed decimal range".format(name))
    if _decimal_text(result) != value:
        _fail("{} must use canonical decimal rendering".format(name))
    return result


def _decimal_text(value: Decimal) -> str:
    if not value.is_finite():
        _fail("non-finite decimal is forbidden")
    if value == 0:
        return "0"
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return "0" if rendered == "-0" else rendered


def _canonicalize(value: Any) -> Any:
    if type(value) is OrderedExperimentArtifact:
        return _canonicalize(value.to_dict())
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _canonicalize(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, MappingProxyType) or type(value) is dict:
        return {key: _canonicalize(value[key]) for key in sorted(value)}
    if type(value) in (tuple, list):
        return [_canonicalize(item) for item in value]
    if type(value) is frozenset:
        return [_canonicalize(item) for item in sorted(value)]
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        # Floats occur only inside the embedded, already-canonical Task 9 artifact.
        # Task 10 derived arithmetic is represented by integer fractions/decimal text.
        return value
    _fail("unsupported canonical value: {}".format(type(value).__name__))


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _canonicalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _domain_hash(domain: str, value: Any) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            domain.encode("utf-8") + b"\0" + _canonical_bytes(value)
        ).hexdigest()
    )


def _exact(data: Any, names: Sequence[str], label: str) -> Dict[str, Any]:
    if type(data) is not dict:
        _fail("{} payload must be an exact dict".format(label))
    expected = set(names)
    if set(data) != expected:
        _fail("{} payload fields are not exact".format(label))
    return dict(data)


class _Record:
    def to_dict(self) -> Dict[str, Any]:
        return cast(Dict[str, Any], _canonicalize(self))

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self)


@dataclass(frozen=True)
class PriceRate(_Record):
    provider: str
    model_id: str
    provider_revision: Optional[str]
    token_accounting_version: str
    input_per_million: str
    output_per_million: str
    rate_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self) is not PriceRate:
            _fail("PriceRate subclasses are not accepted")
        for name in (
            "provider",
            "model_id",
            "token_accounting_version",
        ):
            _text(name, getattr(self, name))
        if self.provider_revision is not None:
            _text("provider_revision", self.provider_revision)
        _decimal("input_per_million", self.input_per_million, Decimal("0"))
        _decimal("output_per_million", self.output_per_million, Decimal("0"))
        object.__setattr__(
            self, "rate_hash", _domain_hash("adaptive-price-rate-v2", self._payload())
        )

    def _payload(self) -> Dict[str, Any]:
        return {
            item.name: getattr(self, item.name) for item in fields(self) if item.init
        }

    @classmethod
    def from_dict(cls, data: Any) -> "PriceRate":
        values = _exact(data, tuple(item.name for item in fields(cls)), "PriceRate")
        expected = values.pop("rate_hash")
        result = cls(**values)
        if result.rate_hash != expected:
            _fail("rate_hash does not match PriceRate")
        return result


@dataclass(frozen=True)
class PricingSpec(_Record):
    currency: str
    effective_version: str
    rates: Tuple[PriceRate, ...]
    pricing_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self) is not PricingSpec:
            _fail("PricingSpec subclasses are not accepted")
        _text("currency", self.currency, 16)
        _text("effective_version", self.effective_version)
        if type(self.rates) not in (tuple, list):
            _fail("rates must be a tuple/list")
        if not 1 <= len(self.rates) <= _MAX_RATES:
            _fail("rates must contain bounded exact PriceRate records")
        rates = tuple(self.rates)
        if any(type(item) is not PriceRate for item in rates):
            _fail("rates must contain bounded exact PriceRate records")
        rates = tuple(PriceRate.from_dict(item.to_dict()) for item in rates)
        keys = tuple(
            (
                item.provider,
                item.model_id,
                item.provider_revision,
                item.token_accounting_version,
            )
            for item in rates
        )
        sorted_keys = tuple(
            sorted(
                keys,
                key=lambda item: (
                    item[0],
                    item[1],
                    item[2] is not None,
                    item[2] or "",
                    item[3],
                ),
            )
        )
        if len(set(keys)) != len(keys) or keys != sorted_keys:
            _fail("pricing rates must be unique and canonically sorted by identity")
        object.__setattr__(self, "rates", rates)
        object.__setattr__(
            self,
            "pricing_hash",
            _domain_hash("adaptive-pricing-spec-v2", self._payload()),
        )

    def _payload(self) -> Dict[str, Any]:
        return {
            item.name: getattr(self, item.name) for item in fields(self) if item.init
        }

    @classmethod
    def from_dict(cls, data: Any) -> "PricingSpec":
        values = _exact(data, tuple(item.name for item in fields(cls)), "PricingSpec")
        expected = values.pop("pricing_hash")
        raw_rates = values["rates"]
        if (
            type(raw_rates) not in (list, tuple)
            or not 1 <= len(raw_rates) <= _MAX_RATES
        ):
            _fail("rates must be a bounded exact list/tuple")
        values["rates"] = tuple(PriceRate.from_dict(item) for item in raw_rates)
        result = cls(**values)
        if result.pricing_hash != expected:
            _fail("pricing_hash does not match PricingSpec")
        return result


@dataclass(frozen=True)
class IntervalSpec(_Record):
    method_version: str = INTERVAL_METHOD_VERSION
    confidence_level: str = "0.95"
    draw_count: int = 10_000
    seed: int = 0
    interval_spec_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self) is not IntervalSpec:
            _fail("IntervalSpec subclasses are not accepted")
        if self.method_version != INTERVAL_METHOD_VERSION:
            _fail("unsupported interval method")
        confidence = _decimal("confidence_level", self.confidence_level)
        if not Decimal("0") < confidence < Decimal("1"):
            _fail("confidence_level must be between zero and one")
        if type(self.draw_count) is not int or not 1 <= self.draw_count <= _MAX_DRAWS:
            _fail("draw_count is outside its bound")
        if (
            type(self.seed) is not int
            or isinstance(self.seed, bool)
            or not -(2**63) <= self.seed < 2**63
        ):
            _fail("seed must be a signed 64-bit integer")
        object.__setattr__(
            self,
            "interval_spec_hash",
            _domain_hash("adaptive-interval-spec-v1", self._payload()),
        )

    def _payload(self) -> Dict[str, Any]:
        return {
            item.name: getattr(self, item.name) for item in fields(self) if item.init
        }

    @classmethod
    def from_dict(cls, data: Any) -> "IntervalSpec":
        values = _exact(data, tuple(item.name for item in fields(cls)), "IntervalSpec")
        expected = values.pop("interval_spec_hash")
        result = cls(**values)
        if result.interval_spec_hash != expected:
            _fail("interval_spec_hash does not match payload")
        return result


@dataclass(frozen=True)
class ReportingSpec(_Record):
    report_version: str = REPORT_VERSION
    primary_estimand: str = _PRIMARY_ESTIMAND
    aggregation_version: str = _AGGREGATION_VERSION
    unscored_quality_value: str = "0"
    pass_threshold: Optional[str] = None
    interval: IntervalSpec = field(default_factory=IntervalSpec)
    decimal_precision: int = 28
    decimal_version: str = _DECIMAL_VERSION
    pricing: Optional[PricingSpec] = None
    claim_scope: str = "Controlled paired evidence only; no efficacy claim."
    reporting_spec_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self) is not ReportingSpec:
            _fail("ReportingSpec subclasses are not accepted")
        if (
            self.report_version != REPORT_VERSION
            or self.primary_estimand != _PRIMARY_ESTIMAND
            or self.aggregation_version != _AGGREGATION_VERSION
        ):
            _fail("unsupported report, estimand, or aggregation version")
        if self.unscored_quality_value != "0":
            _fail("unscored quality is prospectively fixed to exact ITT zero")
        if self.pass_threshold is not None:
            threshold = _decimal("pass_threshold", self.pass_threshold)
            if not Decimal("0") <= threshold <= Decimal("1"):
                _fail("pass_threshold must be between zero and one")
        if type(self.interval) is not IntervalSpec:
            _fail("interval must be exact IntervalSpec")
        interval = IntervalSpec.from_dict(self.interval.to_dict())
        if (
            type(self.decimal_precision) is not int
            or isinstance(self.decimal_precision, bool)
            or not 16 <= self.decimal_precision <= 64
        ):
            _fail("decimal_precision must be between 16 and 64")
        if self.decimal_version != _DECIMAL_VERSION:
            _fail("unsupported decimal rendering version")
        pricing = self.pricing
        if pricing is not None:
            if type(pricing) is not PricingSpec:
                _fail("pricing must be exact PricingSpec")
            pricing = PricingSpec.from_dict(pricing.to_dict())
        _text("claim_scope", self.claim_scope, _MAX_TEXT)
        object.__setattr__(self, "interval", interval)
        object.__setattr__(self, "pricing", pricing)
        object.__setattr__(
            self,
            "reporting_spec_hash",
            _domain_hash("adaptive-reporting-spec-v1", self._payload()),
        )

    def _payload(self) -> Dict[str, Any]:
        return {
            item.name: getattr(self, item.name) for item in fields(self) if item.init
        }

    @classmethod
    def from_dict(cls, data: Any) -> "ReportingSpec":
        values = _exact(data, tuple(item.name for item in fields(cls)), "ReportingSpec")
        expected = values.pop("reporting_spec_hash")
        values["interval"] = IntervalSpec.from_dict(values["interval"])
        if values["pricing"] is not None:
            values["pricing"] = PricingSpec.from_dict(values["pricing"])
        result = cls(**values)
        if result.reporting_spec_hash != expected:
            _fail("reporting_spec_hash does not match payload")
        return result


@dataclass(frozen=True, init=False)
class MetricValue(_Record):
    available: bool
    reason: Optional[str]
    numerator: Optional[int]
    denominator: Optional[int]
    decimal: Optional[str]
    record_hash: str

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("MetricValue is report-derived")

    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError("MetricValue cannot be subclassed")

    @classmethod
    def _make(
        cls,
        available: bool,
        reason: Optional[str],
        value: Optional[Fraction],
        precision: int,
        numerator: Optional[int] = None,
        denominator: Optional[int] = None,
    ) -> "MetricValue":
        result = object.__new__(cls)
        if available:
            if value is None:
                _fail("available metric requires an exact value")
            numerator, denominator = value.numerator, value.denominator
            decimal = _render_fraction(value, precision)
            reason = None
        else:
            if type(reason) is not str or not reason:
                _fail("unavailable metric requires a reason")
            decimal = None
            if numerator is not None:
                _integer("numerator", numerator)
            if denominator is not None:
                _integer("denominator", denominator)
        for name, item in (
            ("available", available),
            ("reason", reason),
            ("numerator", numerator),
            ("denominator", denominator),
            ("decimal", decimal),
        ):
            object.__setattr__(result, name, item)
        object.__setattr__(
            result,
            "record_hash",
            _domain_hash(
                "adaptive-metric-value-v1",
                {
                    "available": available,
                    "reason": reason,
                    "numerator": numerator,
                    "denominator": denominator,
                    "decimal": decimal,
                },
            ),
        )
        return result


@dataclass(frozen=True, init=False)
class PairEffect(_Record):
    phase: str
    family_id: str
    task_case_id: str
    repetition_index: int
    metric: str
    favorable_direction: str
    orientation: str
    baseline_run_id: str
    adaptive_run_id: str
    baseline_task_hash: str
    adaptive_task_hash: str
    baseline_value: MetricValue
    adaptive_value: MetricValue
    effect: MetricValue
    relative_improvement: MetricValue
    record_hash: str

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("PairEffect is report-derived")

    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError("PairEffect cannot be subclassed")


@dataclass(frozen=True, init=False)
class TrajectoryEffect(_Record):
    phase: str
    family_id: Optional[str]
    repetition_index: int
    metric: str
    favorable_direction: str
    nested_case_count: int
    effect: MetricValue
    record_hash: str

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("TrajectoryEffect is report-derived")

    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError("TrajectoryEffect cannot be subclassed")


@dataclass(frozen=True, init=False)
class MetricSummary(_Record):
    summary_kind: str
    phase: str
    family_id: Optional[str]
    arm_id: Optional[str]
    metric: str
    favorable_direction: str
    observation_count: int
    available_count: int
    mean: MetricValue
    record_hash: str

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("MetricSummary is report-derived")

    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError("MetricSummary cannot be subclassed")


@dataclass(frozen=True, init=False)
class IntervalEstimate(_Record):
    phase: str
    family_id: Optional[str]
    metric: str
    favorable_direction: str
    interval_available: bool
    reason: Optional[str]
    lower: Optional[MetricValue]
    upper: Optional[MetricValue]
    confidence_level: str
    draw_count: int
    repetition_count: int
    method_version: str
    interval_spec_hash: str
    evidence_label: str
    record_hash: str

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("IntervalEstimate is report-derived")

    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError("IntervalEstimate cannot be subclassed")


@dataclass(frozen=True, init=False)
class LearningEstimateEvidence(_Record):
    estimate_kind: str
    estimate_id: str
    context_attributes: Tuple[str, ...]
    context_item_id: Optional[str]
    estimated_utility: MetricValue
    confidence: MetricValue
    estimator_version: str
    provenance: str
    estimated_timestamp: str
    source_event_ids: Tuple[str, ...]
    record_hash: str

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("LearningEstimateEvidence is report-derived")

    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError("LearningEstimateEvidence cannot be subclassed")


@dataclass(frozen=True, init=False)
class LearningEvidenceSummary(_Record):
    arm_id: str
    repetition_index: int
    family_id: str
    state_hash: str
    feedback_count: int
    feature_estimate_count: int
    active_feature_count: int
    id_local_estimate_count: int
    active_id_local_count: int
    feature_estimate_ids: Tuple[str, ...]
    id_local_estimate_ids: Tuple[str, ...]
    source_event_ids: Tuple[str, ...]
    estimates: Tuple[LearningEstimateEvidence, ...]
    record_hash: str

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("LearningEvidenceSummary is report-derived")

    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError("LearningEvidenceSummary cannot be subclassed")


def _derived(cls: Any, **values: Any) -> Any:
    expected = {item.name for item in fields(cls)} - {"record_hash"}
    if set(values) != expected:
        raise TypeError("internal {} fields are incomplete".format(cls.__name__))
    result = object.__new__(cls)
    for name, value in values.items():
        object.__setattr__(result, name, value)
    object.__setattr__(
        result,
        "record_hash",
        _domain_hash(
            "adaptive-{}-v1".format(
                re.sub(r"(?<!^)(?=[A-Z])", "-", cls.__name__).lower()
            ),
            values,
        ),
    )
    return result


def _render_fraction(value: Fraction, precision: int) -> str:
    with localcontext(Context(prec=precision, rounding=ROUND_HALF_EVEN)):
        rendered = Decimal(value.numerator) / Decimal(value.denominator)
    return _decimal_text(rendered)


def _available(value: Fraction, spec: ReportingSpec) -> MetricValue:
    return MetricValue._make(True, None, value, spec.decimal_precision)


def _unavailable(
    reason: str,
    spec: ReportingSpec,
    numerator: Optional[int] = None,
    denominator: Optional[int] = None,
) -> MetricValue:
    return MetricValue._make(
        False, reason, None, spec.decimal_precision, numerator, denominator
    )


def _fraction_number(name: str, value: Any) -> Fraction:
    # Freeze source floats through Python's shortest round-trip decimal rendering.
    # Task 9 already binds those source floats into its canonical artifact bytes.
    if (
        type(value) not in (int, float)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        _fail("{} must be a finite non-boolean number".format(name))
    return Fraction(Decimal(str(value)))


_METRICS = (
    ("task_quality", "higher"),
    ("execution_success", "higher"),
    ("scored_outcome", "higher"),
    ("pass", "higher"),
    ("critical_scoring_failure", "lower"),
    ("severe_scoring_failure", "lower"),
    ("correction_count", "lower"),
    ("correction_required", "lower"),
    ("selected_context_tokens", "lower"),
    ("context_precision", "higher"),
    ("context_recall", "higher"),
    ("required_context_recall", "higher"),
    ("misleading_selected_count", "lower"),
    ("misleading_selected_rate", "lower"),
    ("irrelevant_selected_count", "lower"),
    ("irrelevant_selected_rate", "lower"),
    ("provider_input_tokens", "lower"),
    ("provider_output_tokens", "lower"),
    ("provider_latency_ms", "lower"),
    ("selector_latency_ms", "lower"),
    ("estimated_cost", "lower"),
)
_DIRECTION = dict(_METRICS)


def _ratio(numerator: int, denominator: int, spec: ReportingSpec) -> MetricValue:
    if denominator == 0:
        return _unavailable("undefined_zero_denominator", spec, numerator, 0)
    return _available(Fraction(numerator, denominator), spec)


def _rate_for(task: OrderedTaskRecord, pricing: PricingSpec) -> PriceRate:
    execution = task.provider_execution
    key = (
        execution.provider,
        execution.model_id,
        execution.provider_revision,
        execution.token_accounting_version,
    )
    matches = [
        rate
        for rate in pricing.rates
        if (
            rate.provider,
            rate.model_id,
            rate.provider_revision,
            rate.token_accounting_version,
        )
        == key
    ]
    if len(matches) != 1:
        _fail("pricing specification does not completely cover execution identity")
    return matches[0]


def _task_metrics(
    task: OrderedTaskRecord, spec: ReportingSpec
) -> Mapping[str, MetricValue]:
    case = task.case_material.task_case
    candidate_ids = tuple(
        item.context_item_id for item in case.inputs.candidate_context
    )
    candidate_set = set(candidate_ids)
    sealed = case.sealed_evaluation
    groups = (
        tuple(sealed.required_context_item_ids),
        tuple(sealed.useful_context_item_ids),
        tuple(sealed.misleading_context_item_ids),
        tuple(sealed.irrelevant_context_item_ids),
    )
    flat = tuple(item for group in groups for item in group)
    if len(set(flat)) != len(flat) or set(flat) != candidate_set:
        _fail(
            "sealed context labels must be pairwise disjoint, candidate-local, and exhaustive"
        )
    selected = tuple(task.selection_decision.selected_context_item_ids)
    if len(set(selected)) != len(selected) or not set(selected).issubset(candidate_set):
        _fail("selected context IDs must be unique candidate IDs")
    selected_tokens = tuple(task.selection_decision.selected_token_counts)
    expected_tokens = tuple(
        next(
            item.token_count
            for item in case.inputs.candidate_context
            if item.context_item_id == item_id
        )
        for item_id in selected
    )
    if (
        selected_tokens != expected_tokens
        or sum(selected_tokens) != task.selection_decision.total_selected_tokens
    ):
        _fail("selected token counts do not rederive from candidate evidence")

    required, useful, misleading, irrelevant = (set(group) for group in groups)
    relevant = required | useful
    scoring = task.scoring_result
    execution_success = task.task_outcome.execution_status == "success"
    scored_outcome = scoring.status == "scored"
    scored_for_quality = execution_success and scored_outcome
    if scored_for_quality:
        if scoring.normalized_score_decimal is None:
            _fail("scored result lacks normalized_score_decimal")
        quality_decimal = _decimal(
            "normalized_score_decimal", scoring.normalized_score_decimal
        )
        if not Decimal("0") <= quality_decimal <= Decimal("1"):
            _fail("normalized score is out of range")
        quality = Fraction(quality_decimal)
    else:
        quality = Fraction(0)

    finding_status = {
        item.finding_id: item.status for item in scoring.assessment.finding_assessments
    }
    step_status = {
        item.step_id: item.status for item in scoring.assessment.step_assessments
    }
    critical_omission = any(
        item.critical_omission and step_status.get(item.step_id) != "met"
        for item in scoring.spec.required_steps
    )
    critical_finding = any(
        item.severity == "critical" and finding_status.get(item.finding_id) == "present"
        for item in scoring.spec.negative_findings
    )
    severe_finding = any(
        item.severity == "severe" and finding_status.get(item.finding_id) == "present"
        for item in scoring.spec.negative_findings
    )
    critical = critical_omission or critical_finding
    correction_count = scoring.correction_count
    _integer("correction_count", correction_count)
    selected_count = len(selected)
    values: Dict[str, MetricValue] = {
        "task_quality": _available(quality, spec),
        "execution_success": _available(Fraction(int(execution_success)), spec),
        "scored_outcome": _available(Fraction(int(scored_outcome)), spec),
        "critical_scoring_failure": _available(Fraction(int(critical)), spec),
        "severe_scoring_failure": _available(Fraction(int(severe_finding)), spec),
        "correction_count": _available(Fraction(correction_count), spec),
        "correction_required": _available(Fraction(int(correction_count > 0)), spec),
        "selected_context_tokens": _available(
            Fraction(task.selection_decision.total_selected_tokens), spec
        ),
        "context_precision": _ratio(
            len(set(selected) & relevant), selected_count, spec
        ),
        "context_recall": _ratio(len(set(selected) & relevant), len(relevant), spec),
        "required_context_recall": _ratio(
            len(set(selected) & required), len(required), spec
        ),
        "misleading_selected_count": _available(
            Fraction(len(set(selected) & misleading)), spec
        ),
        "misleading_selected_rate": _ratio(
            len(set(selected) & misleading), selected_count, spec
        ),
        "irrelevant_selected_count": _available(
            Fraction(len(set(selected) & irrelevant)), spec
        ),
        "irrelevant_selected_rate": _ratio(
            len(set(selected) & irrelevant), selected_count, spec
        ),
        "provider_input_tokens": _available(
            Fraction(task.provider_execution.input_tokens), spec
        ),
        "provider_output_tokens": _available(
            Fraction(task.provider_execution.output_tokens), spec
        ),
        "provider_latency_ms": _available(
            _fraction_number("provider latency", task.provider_execution.latency_ms),
            spec,
        ),
        "selector_latency_ms": _available(
            _fraction_number(
                "selector latency", task.selection_decision.decision_latency_ms
            ),
            spec,
        ),
    }
    if spec.pass_threshold is None:
        values["pass"] = _unavailable("pass_threshold_not_supplied", spec)
    else:
        threshold = Fraction(_decimal("pass_threshold", spec.pass_threshold))
        values["pass"] = _available(
            Fraction(int(scored_outcome and quality >= threshold and not critical)),
            spec,
        )
    if spec.pricing is None:
        values["estimated_cost"] = _unavailable("pricing_not_supplied", spec)
    else:
        rate = _rate_for(task, spec.pricing)
        cost = (
            Fraction(task.provider_execution.input_tokens)
            * Fraction(_decimal("input_per_million", rate.input_per_million))
            + Fraction(task.provider_execution.output_tokens)
            * Fraction(_decimal("output_per_million", rate.output_per_million))
        ) / 1_000_000
        values["estimated_cost"] = _available(cost, spec)
    return MappingProxyType(values)


def _mean(values: Sequence[MetricValue], spec: ReportingSpec) -> MetricValue:
    if not values:
        return _unavailable("no_observations", spec)
    unavailable = [item for item in values if not item.available]
    if unavailable:
        reasons = sorted(set(cast(str, item.reason) for item in unavailable))
        if len(reasons) == 1:
            return _unavailable(reasons[0], spec)
        return _unavailable("nested_metric_unavailable:" + ",".join(reasons), spec)
    exact = [
        Fraction(cast(int, item.numerator), cast(int, item.denominator))
        for item in values
    ]
    return _available(sum(exact, Fraction(0)) / len(exact), spec)


def _family_balanced_mean(
    case_values_by_family: Mapping[str, Sequence[MetricValue]],
    spec: ReportingSpec,
) -> MetricValue:
    if not case_values_by_family:
        return _unavailable("no_families", spec)
    family_means = [
        _mean(case_values_by_family[family], spec)
        for family in sorted(case_values_by_family)
    ]
    return _mean(family_means, spec)


def _pair_value(
    baseline: MetricValue, adaptive: MetricValue, spec: ReportingSpec
) -> MetricValue:
    if not baseline.available or not adaptive.available:
        reasons = sorted(
            set(
                cast(str, item.reason)
                for item in (baseline, adaptive)
                if not item.available
            )
        )
        if reasons == ["pricing_not_supplied"]:
            return _unavailable("pricing_not_supplied", spec)
        return _unavailable("paired_metric_unavailable:" + ",".join(reasons), spec)
    left = Fraction(cast(int, baseline.numerator), cast(int, baseline.denominator))
    right = Fraction(cast(int, adaptive.numerator), cast(int, adaptive.denominator))
    return _available(right - left, spec)


def _relative_improvement(
    baseline: MetricValue,
    adaptive: MetricValue,
    favorable_direction: str,
    spec: ReportingSpec,
) -> MetricValue:
    if not baseline.available or not adaptive.available:
        reasons = sorted(
            set(
                cast(str, item.reason)
                for item in (baseline, adaptive)
                if not item.available
            )
        )
        if len(reasons) == 1:
            return _unavailable(reasons[0], spec)
        return _unavailable("paired_metric_unavailable:" + ",".join(reasons), spec)
    left = Fraction(cast(int, baseline.numerator), cast(int, baseline.denominator))
    if left == 0:
        return _unavailable("baseline_zero", spec, 0, 0)
    right = Fraction(cast(int, adaptive.numerator), cast(int, adaptive.denominator))
    difference = right - left
    if favorable_direction == "lower":
        difference = -difference
    return _available(difference / abs(left), spec)


def _metric_summary(
    kind: str,
    phase: str,
    family: Optional[str],
    arm: Optional[str],
    metric: str,
    values: Sequence[MetricValue],
    spec: ReportingSpec,
) -> MetricSummary:
    return _derived(
        MetricSummary,
        summary_kind=kind,
        phase=phase,
        family_id=family,
        arm_id=arm,
        metric=metric,
        favorable_direction=_DIRECTION[metric],
        observation_count=len(values),
        available_count=sum(1 for item in values if item.available),
        mean=_mean(values, spec),
    )


def _percentile(sorted_values: Sequence[Fraction], probability: Fraction) -> Fraction:
    # Frozen nearest-rank rule: rank=ceil(p*N), one-indexed and clamped.
    scaled_numerator = probability.numerator * len(sorted_values)
    rank = (scaled_numerator + probability.denominator - 1) // probability.denominator
    rank = max(1, min(len(sorted_values), rank))
    return sorted_values[rank - 1]


def _interval(
    summary: MetricSummary,
    relevant: Sequence[TrajectoryEffect],
    spec: ReportingSpec,
) -> IntervalEstimate:
    rep_count = len(relevant)
    common = dict(
        phase="evaluation",
        family_id=summary.family_id,
        metric=summary.metric,
        favorable_direction=summary.favorable_direction,
        confidence_level=spec.interval.confidence_level,
        draw_count=spec.interval.draw_count,
        repetition_count=rep_count,
        method_version=spec.interval.method_version,
        interval_spec_hash=spec.interval.interval_spec_hash,
    )
    if rep_count < 2:
        return _derived(
            IntervalEstimate,
            interval_available=False,
            reason="fewer_than_two_repetitions",
            lower=None,
            upper=None,
            evidence_label="insufficient_repetition_evidence",
            **common,
        )
    if any(not item.effect.available for item in relevant):
        return _derived(
            IntervalEstimate,
            interval_available=False,
            reason="trajectory_metric_unavailable",
            lower=None,
            upper=None,
            evidence_label="unavailable_nested_metric",
            **common,
        )
    vector = [
        Fraction(cast(int, item.effect.numerator), cast(int, item.effect.denominator))
        for item in relevant
    ]
    draws = []
    for draw in range(spec.interval.draw_count):
        sampled = []
        for position in range(rep_count):
            material = {
                "draw": draw,
                "interval_spec_hash": spec.interval.interval_spec_hash,
                "sample_position": position,
            }
            digest = hashlib.sha256(
                b"paired-repetition-cluster-sample-v1\0" + _canonical_bytes(material)
            ).digest()
            sampled.append(vector[int.from_bytes(digest, "big") % rep_count])
        draws.append(sum(sampled, Fraction(0)) / rep_count)
    draws.sort()
    confidence = Fraction(_decimal("confidence_level", spec.interval.confidence_level))
    alpha = (Fraction(1) - confidence) / 2
    return _derived(
        IntervalEstimate,
        interval_available=True,
        reason=None,
        lower=_available(_percentile(draws, alpha), spec),
        upper=_available(_percentile(draws, 1 - alpha), spec),
        evidence_label=(
            "coarse_control_evidence"
            if rep_count == 2
            else "repetition_cluster_interval"
        ),
        **common,
    )


def _learning_summary(
    arm_id: str, repetition: int, state: Any, spec: ReportingSpec
) -> LearningEvidenceSummary:
    payload = state.snapshot_payload
    if not isinstance(payload, Mapping):
        _fail("learning snapshot payload must be an object")
    feature_estimates = tuple(payload.get("feature_estimates", ()))
    id_estimates = tuple(payload.get("id_local_estimates", ()))
    feature_utilities = payload.get("feature_utilities", {})
    id_utilities = payload.get("id_local_utilities", {})
    if not isinstance(feature_utilities, Mapping) or not isinstance(
        id_utilities, Mapping
    ):
        _fail("learning utility payloads must be mappings")
    family = state.task_family_id
    active_features = feature_utilities.get(family, {})
    active_ids = id_utilities.get(family, {})
    if not isinstance(active_features, Mapping) or not isinstance(active_ids, Mapping):
        _fail("family learning utilities must be mappings")
    feature_ids = tuple(item["utility_estimate_id"] for item in feature_estimates)
    id_ids = tuple(item["id_local_utility_estimate_id"] for item in id_estimates)
    source_ids = tuple(
        dict.fromkeys(
            event
            for item in feature_estimates + id_estimates
            for event in item["source_event_ids"]
        )
    )
    estimates = []
    for estimate in feature_estimates:
        estimates.append(
            _derived(
                LearningEstimateEvidence,
                estimate_kind="feature",
                estimate_id=estimate["utility_estimate_id"],
                context_attributes=tuple(estimate["context_attributes"]),
                context_item_id=None,
                estimated_utility=_available(
                    _fraction_number(
                        "feature estimated utility", estimate["estimated_utility"]
                    ),
                    spec,
                ),
                confidence=_available(
                    _fraction_number("feature confidence", estimate["confidence"]), spec
                ),
                estimator_version=estimate["estimator_version"],
                provenance=estimate["provenance"],
                estimated_timestamp=estimate["estimated_timestamp"],
                source_event_ids=tuple(estimate["source_event_ids"]),
            )
        )
    for estimate in id_estimates:
        estimates.append(
            _derived(
                LearningEstimateEvidence,
                estimate_kind="id_local",
                estimate_id=estimate["id_local_utility_estimate_id"],
                context_attributes=(),
                context_item_id=estimate["context_item_id"],
                estimated_utility=_available(
                    _fraction_number(
                        "ID-local estimated utility", estimate["estimated_utility"]
                    ),
                    spec,
                ),
                confidence=_available(
                    _fraction_number("ID-local confidence", estimate["confidence"]),
                    spec,
                ),
                estimator_version=estimate["estimator_version"],
                provenance=estimate["provenance"],
                estimated_timestamp=estimate["estimated_timestamp"],
                source_event_ids=tuple(estimate["source_event_ids"]),
            )
        )
    return _derived(
        LearningEvidenceSummary,
        arm_id=arm_id,
        repetition_index=repetition,
        family_id=family,
        state_hash=state.state_hash,
        feedback_count=len(state.feedback_event_ids),
        feature_estimate_count=len(feature_estimates),
        active_feature_count=len(active_features),
        id_local_estimate_count=len(id_estimates),
        active_id_local_count=len(active_ids),
        feature_estimate_ids=feature_ids,
        id_local_estimate_ids=id_ids,
        source_event_ids=source_ids,
        estimates=tuple(estimates),
    )


@dataclass(frozen=True, init=False)
class ExperimentReport(_Record):
    report_version: str
    source_artifact: OrderedExperimentArtifact
    reporting_spec: ReportingSpec
    source_artifact_hash: str
    reporting_spec_hash: str
    primary_baseline_arm_id: str
    adaptive_candidate_arm_id: str
    arm_summaries: Tuple[MetricSummary, ...]
    pair_effects: Tuple[PairEffect, ...]
    trajectory_effects: Tuple[TrajectoryEffect, ...]
    adaptation_summaries: Tuple[MetricSummary, ...]
    primary_summaries: Tuple[MetricSummary, ...]
    intervals: Tuple[IntervalEstimate, ...]
    learning_evidence: Tuple[LearningEvidenceSummary, ...]
    id_local_outcome_ablation_available: bool
    contains_adverse_or_null_primary_effects: bool
    claim_scope: str
    limitations: Tuple[str, ...]
    report_hash: str

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("ExperimentReport is derived only by build_experiment_report")

    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError("ExperimentReport cannot be subclassed")

    @classmethod
    def _make(cls, **values: Any) -> "ExperimentReport":
        expected = {item.name for item in fields(cls)} - {"report_hash"}
        if set(values) != expected:
            raise TypeError("internal ExperimentReport fields are incomplete")
        result = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(result, name, value)
        object.__setattr__(
            result, "report_hash", _domain_hash("adaptive-experiment-report-v1", values)
        )
        return result

    @classmethod
    def from_dict(cls, data: Any) -> "ExperimentReport":
        try:
            values = _exact(
                data, tuple(item.name for item in fields(cls)), "ExperimentReport"
            )
            # Bound obvious amplification before traversing nested supplied derivations.
            for name in (
                "arm_summaries",
                "pair_effects",
                "trajectory_effects",
                "adaptation_summaries",
                "primary_summaries",
                "intervals",
                "learning_evidence",
            ):
                if type(values[name]) is not list or len(values[name]) > _MAX_RECORDS:
                    _fail("{} exceeds report resource bounds".format(name))
            artifact = OrderedExperimentArtifact.from_dict(values["source_artifact"])
            spec = ReportingSpec.from_dict(values["reporting_spec"])
            derived = build_experiment_report(artifact, spec)
            if derived.to_dict() != data:
                _fail("payload is not an exact canonical derived report")
            return derived
        except (KeyError, OverflowError, RecursionError, TypeError, ValueError) as exc:
            if isinstance(exc, ValueError) and "canonical derived report" in str(exc):
                raise
            raise ValueError(
                "payload is not an exact canonical derived report"
            ) from exc


def build_experiment_report(
    artifact: OrderedExperimentArtifact, spec: ReportingSpec
) -> ExperimentReport:
    """Derive the complete immutable Task 10 report without ambient state."""
    if type(artifact) is not OrderedExperimentArtifact:
        raise TypeError("artifact must be exact OrderedExperimentArtifact")
    if type(spec) is not ReportingSpec:
        raise TypeError("spec must be exact ReportingSpec")
    canonical_artifact = OrderedExperimentArtifact.from_dict(artifact.to_dict())
    canonical_spec = ReportingSpec.from_dict(spec.to_dict())
    if canonical_artifact.canonical_bytes() != artifact.canonical_bytes():
        _fail("artifact failed canonical byte reconstruction")

    baselines = [
        arm
        for arm in canonical_artifact.plan.arms
        if arm.classification == "primary_baseline"
    ]
    candidates = [
        arm
        for arm in canonical_artifact.plan.arms
        if arm.classification == "candidate"
        and arm.selector_mode == "adaptive_policy"
        and arm.uses_feature_learning is True
    ]
    candidate_classified = [
        arm for arm in canonical_artifact.plan.arms if arm.classification == "candidate"
    ]
    if len(baselines) != 1:
        _fail("report requires exactly one primary_baseline arm")
    if len(candidates) != 1 or len(candidate_classified) != 1:
        _fail("report requires exactly one adaptive_policy candidate arm")
    baseline, candidate = baselines[0], candidates[0]
    if baseline.arm_id == candidate.arm_id:
        _fail("baseline and candidate arms must be distinct")

    runs = {
        (run.arm_id, run.repetition_index): run for run in canonical_artifact.arm_runs
    }
    task_rows = []
    tasks_by_key: Dict[Tuple[str, int, str, str, str], OrderedTaskRecord] = {}
    for run in canonical_artifact.arm_runs:
        for task in run.task_records:
            key = (
                run.arm_id,
                run.repetition_index,
                task.phase,
                task.family_id,
                task.task_case_id,
            )
            if key in tasks_by_key:
                _fail("duplicate arm/repetition/phase/family/case task")
            tasks_by_key[key] = task
            task_rows.append((run, task))
    if len(task_rows) * len(_METRICS) > _MAX_RECORDS:
        _fail("report exceeds derived record resource bound")

    metrics_by_task: Dict[Tuple[str, int, str, str, str], Mapping[str, MetricValue]] = (
        {}
    )
    arm_metric_groups: Dict[Tuple[str, str, Optional[str], str], list] = defaultdict(
        list
    )
    for run, task in task_rows:
        key = (
            run.arm_id,
            run.repetition_index,
            task.phase,
            task.family_id,
            task.task_case_id,
        )
        metric_values = _task_metrics(task, canonical_spec)
        metrics_by_task[key] = metric_values
        for metric, _ in _METRICS:
            arm_metric_groups[(task.phase, run.arm_id, None, metric)].append(
                metric_values[metric]
            )
            arm_metric_groups[(task.phase, run.arm_id, task.family_id, metric)].append(
                metric_values[metric]
            )

    # Every arm gets separate phase, overall, and per-family descriptive summaries.
    arm_summaries = []
    for phase in ("adaptation", "evaluation"):
        for arm in canonical_artifact.plan.arms:
            for family in (None,) + canonical_artifact.plan.family_order:
                for metric, _ in _METRICS:
                    values = arm_metric_groups[(phase, arm.arm_id, family, metric)]
                    if not values:
                        _fail("arm descriptive summary coverage is incomplete")
                    arm_summaries.append(
                        _metric_summary(
                            "arm_descriptive",
                            phase,
                            family,
                            arm.arm_id,
                            metric,
                            values,
                            canonical_spec,
                        )
                    )

    pair_effects = []
    pair_groups: Dict[Tuple[str, int, str, str], list] = defaultdict(list)
    for phase in ("adaptation", "evaluation"):
        exemplar = runs[(baseline.arm_id, 0)]
        case_shape = [
            (task.family_id, task.task_case_id, task.ordinal)
            for task in exemplar.task_records
            if task.phase == phase
        ]
        for family, case_id, ordinal in case_shape:
            for repetition in (
                item.repetition_index for item in canonical_artifact.plan.repetitions
            ):
                baseline_run = runs[(baseline.arm_id, repetition)]
                adaptive_run = runs[(candidate.arm_id, repetition)]
                baseline_task = tasks_by_key.get(
                    (baseline.arm_id, repetition, phase, family, case_id)
                )
                adaptive_task = tasks_by_key.get(
                    (candidate.arm_id, repetition, phase, family, case_id)
                )
                if baseline_task is None or adaptive_task is None:
                    _fail("primary pair coverage is incomplete")
                if baseline_task.ordinal != ordinal or adaptive_task.ordinal != ordinal:
                    _fail("primary pair ordinal identity mismatch")
                base_values = metrics_by_task[
                    (baseline.arm_id, repetition, phase, family, case_id)
                ]
                adaptive_values = metrics_by_task[
                    (candidate.arm_id, repetition, phase, family, case_id)
                ]
                for metric, direction in _METRICS:
                    effect = _pair_value(
                        base_values[metric], adaptive_values[metric], canonical_spec
                    )
                    pair = _derived(
                        PairEffect,
                        phase=phase,
                        family_id=family,
                        task_case_id=case_id,
                        repetition_index=repetition,
                        metric=metric,
                        favorable_direction=direction,
                        orientation="adaptive_minus_baseline",
                        baseline_run_id=baseline_run.run_id,
                        adaptive_run_id=adaptive_run.run_id,
                        baseline_task_hash=baseline_task.task_record_hash,
                        adaptive_task_hash=adaptive_task.task_record_hash,
                        baseline_value=base_values[metric],
                        adaptive_value=adaptive_values[metric],
                        effect=effect,
                        relative_improvement=_relative_improvement(
                            base_values[metric],
                            adaptive_values[metric],
                            direction,
                            canonical_spec,
                        ),
                    )
                    pair_effects.append(pair)
                    pair_groups[(phase, repetition, family, metric)].append(pair.effect)

    trajectories = []
    trajectory_groups: Dict[Tuple[str, Optional[str], str], list] = defaultdict(list)
    for phase in ("adaptation", "evaluation"):
        for repetition in (
            item.repetition_index for item in canonical_artifact.plan.repetitions
        ):
            for metric, direction in _METRICS:
                family_values = []
                case_values_by_family = {}
                for family in canonical_artifact.plan.family_order:
                    nested = pair_groups[(phase, repetition, family, metric)]
                    if not nested:
                        _fail("trajectory family coverage is incomplete")
                    value = _mean(nested, canonical_spec)
                    family_values.append(value)
                    case_values_by_family[family] = nested
                    trajectory = _derived(
                        TrajectoryEffect,
                        phase=phase,
                        family_id=family,
                        repetition_index=repetition,
                        metric=metric,
                        favorable_direction=direction,
                        nested_case_count=len(nested),
                        effect=value,
                    )
                    trajectories.append(trajectory)
                    trajectory_groups[(phase, family, metric)].append(trajectory)
                trajectory = _derived(
                    TrajectoryEffect,
                    phase=phase,
                    family_id=None,
                    repetition_index=repetition,
                    metric=metric,
                    favorable_direction=direction,
                    nested_case_count=len(family_values),
                    effect=_family_balanced_mean(case_values_by_family, canonical_spec),
                )
                trajectories.append(trajectory)
                trajectory_groups[(phase, None, metric)].append(trajectory)

    adaptation_summaries = []
    primary_summaries = []
    for phase, kind, destination in (
        ("adaptation", "adaptation_paired_descriptive", adaptation_summaries),
        ("evaluation", "primary_paired", primary_summaries),
    ):
        for family in (None,) + canonical_artifact.plan.family_order:
            for metric, _ in _METRICS:
                destination.append(
                    _metric_summary(
                        kind,
                        phase,
                        family,
                        None,
                        metric,
                        [
                            item.effect
                            for item in trajectory_groups[(phase, family, metric)]
                        ],
                        canonical_spec,
                    )
                )
    repetition_count = len(canonical_artifact.plan.repetitions)
    bootstrap_operations = (
        len(primary_summaries) * canonical_spec.interval.draw_count * repetition_count
    )
    if bootstrap_operations > _MAX_BOOTSTRAP_OPERATIONS:
        _fail("interval specification exceeds bootstrap work bound")
    intervals = tuple(
        _interval(
            summary,
            trajectory_groups[("evaluation", summary.family_id, summary.metric)],
            canonical_spec,
        )
        for summary in primary_summaries
    )

    learning_states = [
        (run.arm_id, run.repetition_index, state)
        for run in canonical_artifact.arm_runs
        for state in run.final_learning_states
    ]
    learning_estimate_count = 0
    for _, _, state in learning_states:
        payload = state.snapshot_payload
        if not isinstance(payload, Mapping):
            _fail("learning snapshot payload must be an object")
        feature_estimates = payload.get("feature_estimates", ())
        id_local_estimates = payload.get("id_local_estimates", ())
        if type(feature_estimates) not in (tuple, list) or type(
            id_local_estimates
        ) not in (tuple, list):
            _fail("learning estimates must be exact bounded sequences")
        learning_estimate_count += len(feature_estimates) + len(id_local_estimates)
        if learning_estimate_count + len(learning_states) > _MAX_RECORDS:
            _fail("learning evidence exceeds report resource bound")
    learning = tuple(
        _learning_summary(arm_id, repetition, state, canonical_spec)
        for arm_id, repetition, state in learning_states
    )
    adverse_or_null = any(
        summary.mean.available
        and (
            cast(int, summary.mean.numerator) == 0
            or (
                _DIRECTION[summary.metric] == "higher"
                and cast(int, summary.mean.numerator) < 0
            )
            or (
                _DIRECTION[summary.metric] == "lower"
                and cast(int, summary.mean.numerator) > 0
            )
        )
        for summary in primary_summaries
    )
    return ExperimentReport._make(
        report_version=REPORT_VERSION,
        source_artifact=canonical_artifact,
        reporting_spec=canonical_spec,
        source_artifact_hash=canonical_artifact.artifact_hash,
        reporting_spec_hash=canonical_spec.reporting_spec_hash,
        primary_baseline_arm_id=baseline.arm_id,
        adaptive_candidate_arm_id=candidate.arm_id,
        arm_summaries=tuple(arm_summaries),
        pair_effects=tuple(pair_effects),
        trajectory_effects=tuple(trajectories),
        adaptation_summaries=tuple(adaptation_summaries),
        primary_summaries=tuple(primary_summaries),
        intervals=intervals,
        learning_evidence=learning,
        id_local_outcome_ablation_available=False,
        contains_adverse_or_null_primary_effects=adverse_or_null,
        claim_scope=canonical_spec.claim_scope,
        limitations=(
            "No p-values or statistical-significance claims are produced.",
            "ID-local utility is descriptive evidence, not a fifth outcome arm.",
            "The report does not prove provider authenticity, assessment validity, efficacy, or economic value.",
        ),
    )


__all__ = [
    "REPORT_VERSION",
    "INTERVAL_METHOD_VERSION",
    "PriceRate",
    "PricingSpec",
    "IntervalSpec",
    "ReportingSpec",
    "MetricValue",
    "PairEffect",
    "TrajectoryEffect",
    "IntervalEstimate",
    "MetricSummary",
    "LearningEstimateEvidence",
    "LearningEvidenceSummary",
    "ExperimentReport",
    "build_experiment_report",
]
