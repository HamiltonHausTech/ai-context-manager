"""Frozen provider adapters and run-manifest controls.

These records capture configuration, rendered requests, raw transport bytes, and
manifest bindings for local experiment reproducibility evidence. They do not claim
hosted determinism, authenticity, fairness, leakage prevention, or outcome quality.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

from .schema import RunManifest, SCHEMA_VERSION

TOKEN_ACCOUNTING_VERSION = "provider-reported-usage-v1"
PRIMARY_COMPARABILITY_FIELDS = (
    "experiment_version",
    "protocol_version",
    "dataset_version",
    "dataset_hash",
    "provider",
    "model_id",
    "prompt_template_hash",
    "config_hash",
    "code_revision",
    "temperature",
    "seed",
    "tool_availability",
)

_MAX_SHORT_STRING = 256
_MAX_PROVENANCE = 1024
_MAX_PROMPT_BYTES = 10 * 1024 * 1024
_MAX_RESPONSE_BYTES = 10 * 1024 * 1024
_MAX_RAW_BYTES = 50 * 1024 * 1024
_MAX_CANONICAL_JSON_BYTES = 1024 * 1024
_MAX_TREE_DEPTH = 32
_MAX_TREE_NODES = 100_000
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{6})?Z$")
T = TypeVar("T", bound="_CanonicalRecord")
ClockValue = Union[str, datetime]


class ProviderError(Exception):
    """Base class for frozen provider adapter failures."""


class ProviderValidationError(ProviderError, ValueError):
    """A provider boundary record is malformed or tampered."""


class ProviderIdentityMismatchError(ProviderValidationError):
    """Transport identity does not match the bound configuration/request."""


class ProviderFixtureNotFoundError(ProviderError, LookupError):
    """A deterministic fixture was not recorded for the request hash."""


class TokenAccountingUnavailableError(ProviderValidationError):
    """Provider token counts were missing or not exact provider-reported usage."""


class ManifestConsistencyError(ProviderError, ValueError):
    """Run manifest, request, configuration, or execution identities disagree."""


class IncompatibleManifestError(ManifestConsistencyError):
    """Run manifests differ on fields required for primary comparison."""

    def __init__(self, differences: Tuple["ManifestDifference", ...]) -> None:
        self.differences = differences
        fields_text = ", ".join(item.field for item in differences)
        super().__init__(f"manifests are incompatible on fields: {fields_text}")


class ProviderCallbackError(ProviderError):
    """Callback raised after the adapter captured start/completion/latency."""

    def __init__(
        self,
        category: str,
        started_timestamp: str,
        completed_timestamp: str,
        latency_ms: float,
    ) -> None:
        self.category = category
        self.started_timestamp = started_timestamp
        self.completed_timestamp = completed_timestamp
        self.latency_ms = latency_ms
        super().__init__(f"provider callback failed: {category}")


def _fail(message: str) -> None:
    raise ProviderValidationError(message)


def _short_string(name: str, value: Any, max_length: int = _MAX_SHORT_STRING) -> str:
    if type(value) is not str or not value.strip() or len(value) > max_length:
        _fail(f"{name} must be a nonempty string of at most {max_length} characters")
    return cast(str, value)


def _optional_string(name: str, value: Any, max_length: int) -> Optional[str]:
    if value is None:
        return None
    return _short_string(name, value, max_length)


def _hash(name: str, value: Any) -> str:
    if type(value) is not str or not _SHA256_RE.fullmatch(value):
        _fail(f"{name} must be sha256:<64 lowercase hex>")
    return cast(str, value)


def _number(name: str, value: Any) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        _fail(f"{name} must be a finite non-boolean number")
    return float(value)


def _temperature(value: Any) -> float:
    number = _number("temperature", value)
    if not 0.0 <= number <= 2.0:
        _fail("temperature must be between 0 and 2")
    return number


def _int64(name: str, value: Any) -> int:
    if type(value) is not int or not _INT64_MIN <= value <= _INT64_MAX:
        _fail(f"{name} must be a signed 64-bit integer")
    return cast(int, value)


def _token(name: str, value: Any) -> int:
    if type(value) is not int or not 0 <= value <= _INT64_MAX:
        raise TokenAccountingUnavailableError(
            f"{name} must be provider reported int64 tokens"
        )
    return cast(int, value)


def _tools(value: Any) -> Tuple[str, ...]:
    if type(value) not in (list, tuple):
        _fail("tool_availability must be a tuple/list of strings")
    items = tuple(value)
    if len(items) > 128:
        _fail("tool_availability must contain at most 128 entries")
    for item in items:
        _short_string("tool_availability item", item)
    if len(set(items)) != len(items):
        _fail("tool_availability duplicates are not allowed")
    return tuple(sorted(cast(Tuple[str, ...], items)))


def _canonical_jsonable(value: Any) -> Any:
    if isinstance(value, MappingProxyType) or type(value) is dict:
        return {key: _canonical_jsonable(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_canonical_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_canonical_jsonable(item) for item in value]
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise ProviderValidationError(
        f"unsupported canonical value: {type(value).__name__}"
    )


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _canonical_jsonable(value),
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


def _freeze_json(value: Any, path: str = "generation_options") -> Any:
    count = 0

    def visit(item: Any, item_path: str, depth: int) -> Any:
        nonlocal count
        count += 1
        if count > _MAX_TREE_NODES:
            _fail(f"{path} exceeds {_MAX_TREE_NODES} JSON nodes")
        if depth > _MAX_TREE_DEPTH:
            _fail(f"{path} exceeds depth {_MAX_TREE_DEPTH}")
        if item is None or type(item) in (bool, str):
            return item
        if type(item) is int:
            _int64(item_path, item)
            return item
        if type(item) is float:
            if not math.isfinite(item):
                _fail(f"{item_path} must be finite")
            return item
        if type(item) is dict:
            frozen: Dict[str, Any] = {}
            for key in item:
                if type(key) is not str:
                    _fail(f"{item_path} object keys must be strings")
                frozen[cast(str, key)] = visit(
                    item[key], f"{item_path}.{key}", depth + 1
                )
            return MappingProxyType({key: frozen[key] for key in sorted(frozen)})
        if type(item) in (list, tuple):
            return tuple(
                visit(child, f"{item_path}[{index}]", depth + 1)
                for index, child in enumerate(item)
            )
        _fail(f"{item_path} contains unsupported JSON value: {type(item).__name__}")

    frozen_value = visit(value, path, 0)
    if len(_canonical_bytes(frozen_value)) > _MAX_CANONICAL_JSON_BYTES:
        _fail(f"{path} canonical JSON exceeds 1MiB")
    return frozen_value


def _bytes_envelope(data: bytes) -> Dict[str, str]:
    return {"encoding": "base64", "data": base64.b64encode(data).decode("ascii")}


def _bytes_from_envelope(value: Any, name: str) -> bytes:
    if type(value) is not dict or set(value) != {"encoding", "data"}:
        _fail(f"{name} must be a canonical base64 envelope")
    if value["encoding"] != "base64" or type(value["data"]) is not str:
        _fail(f"{name} must be a canonical base64 envelope")
    try:
        decoded = base64.b64decode(value["data"].encode("ascii"), validate=True)
    except (binascii.Error, UnicodeEncodeError):
        _fail(f"{name} must be valid base64")
    if base64.b64encode(decoded).decode("ascii") != value["data"]:
        _fail(f"{name} must be canonical base64")
    return decoded


def _raw_bytes(value: Any) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        _fail("raw_response_bytes must be bytes-like")
    data = bytes(value)
    if not data or len(data) > _MAX_RAW_BYTES:
        _fail("raw_response_bytes must be nonempty and at most 50MiB")
    return data


def _prompt(value: Any) -> str:
    if type(value) is not str or not value:
        _fail("prompt_text must be a nonempty string")
    if len(value.encode("utf-8")) > _MAX_PROMPT_BYTES:
        _fail("prompt_text must be at most 10MiB UTF-8")
    return cast(str, value)


def _response_text(value: Any) -> str:
    """Capture provider response text exactly, including empty or whitespace-only text."""
    if type(value) is not str:
        _fail("response_text must be a string")
    if len(value.encode("utf-8")) > _MAX_RESPONSE_BYTES:
        _fail("response_text must be at most 10MiB UTF-8")
    return cast(str, value)


def _utc(value: ClockValue, name: str = "timestamp") -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value):
            _fail(f"{name} must be aware UTC")
        value = value.astimezone(timezone.utc)
        if value.microsecond:
            return value.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        return value.strftime("%Y-%m-%dT%H:%M:%SZ")
    if type(value) is str and _UTC_RE.fullmatch(value):
        try:
            datetime.strptime(
                value, "%Y-%m-%dT%H:%M:%S.%fZ" if "." in value else "%Y-%m-%dT%H:%M:%SZ"
            )
        except ValueError:
            _fail(f"{name} must be canonical UTC")
        return cast(str, value)
    _fail(f"{name} must be canonical UTC ending Z")


def _monotonic(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        _fail(f"{name} must be finite numeric monotonic time")
    return float(value)


def _exact_fields(data: Mapping[str, Any], expected: Tuple[str, ...]) -> None:
    missing = set(expected) - set(data)
    extra = set(data) - set(expected)
    if missing:
        _fail("missing fields: " + ", ".join(sorted(missing)))
    if extra:
        _fail("unexpected fields: " + ", ".join(sorted(extra)))


def _serialize(value: Any) -> Any:
    if isinstance(value, bytes):
        return _bytes_envelope(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _serialize(getattr(value, item.name)) for item in fields(value)
        }
    if isinstance(value, MappingProxyType):
        return {key: _serialize(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise ProviderValidationError(f"cannot serialize {type(value).__name__}")


class _CanonicalRecord:
    def to_dict(self) -> Dict[str, Any]:
        return cast(Dict[str, Any], _serialize(self))

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())


@dataclass(frozen=True)
class ProviderConfiguration(_CanonicalRecord):
    provider: str
    model_id: str
    provider_revision: str
    temperature: float
    seed: Optional[int]
    seed_supported: bool
    tool_availability: Tuple[str, ...]
    token_accounting_version: str
    generation_options: Any
    config_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self) is not ProviderConfiguration:
            _fail("ProviderConfiguration subclasses are not accepted")
        object.__setattr__(self, "provider", _short_string("provider", self.provider))
        object.__setattr__(self, "model_id", _short_string("model_id", self.model_id))
        object.__setattr__(
            self,
            "provider_revision",
            _short_string("provider_revision", self.provider_revision),
        )
        object.__setattr__(self, "temperature", _temperature(self.temperature))
        if type(self.seed_supported) is not bool:
            _fail("seed_supported must be boolean")
        if self.seed_supported:
            object.__setattr__(self, "seed", _int64("seed", self.seed))
        elif self.seed is not None:
            _fail("seed must be null when seed_supported is false")
        object.__setattr__(self, "tool_availability", _tools(self.tool_availability))
        if self.token_accounting_version != TOKEN_ACCOUNTING_VERSION:
            _fail(f"token_accounting_version must be {TOKEN_ACCOUNTING_VERSION}")
        object.__setattr__(
            self, "generation_options", _freeze_json(self.generation_options)
        )
        object.__setattr__(
            self,
            "config_hash",
            _domain_hash("adaptive-provider-configuration-v1", self._hash_payload()),
        )

    def _hash_payload(self) -> Dict[str, Any]:
        return {key: _serialize(getattr(self, key)) for key in _CONFIG_FIELDS}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProviderConfiguration":
        if type(data) is not dict:
            _fail("ProviderConfiguration payload must be an exact dict")
        _exact_fields(data, _CONFIG_FIELDS + ("config_hash",))
        expected = _hash("config_hash", data["config_hash"])
        record = cls(**{key: data[key] for key in _CONFIG_FIELDS})
        if record.config_hash != expected:
            _fail("config_hash does not match ProviderConfiguration payload")
        return record


_CONFIG_FIELDS = tuple(
    field.name for field in fields(ProviderConfiguration) if field.init
)


@dataclass(frozen=True)
class ProviderRequest(_CanonicalRecord):
    prompt_text: str
    prompt_template_hash: str
    request_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self) is not ProviderRequest:
            _fail("ProviderRequest subclasses are not accepted")
        object.__setattr__(self, "prompt_text", _prompt(self.prompt_text))
        object.__setattr__(
            self,
            "prompt_template_hash",
            _hash("prompt_template_hash", self.prompt_template_hash),
        )
        object.__setattr__(
            self,
            "request_hash",
            _domain_hash(
                "adaptive-provider-request-v1",
                {
                    "prompt_text": self.prompt_text,
                    "prompt_template_hash": self.prompt_template_hash,
                },
            ),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProviderRequest":
        if type(data) is not dict:
            _fail("ProviderRequest payload must be an exact dict")
        _exact_fields(data, ("prompt_text", "prompt_template_hash", "request_hash"))
        expected = _hash("request_hash", data["request_hash"])
        record = cls(data["prompt_text"], data["prompt_template_hash"])
        if record.request_hash != expected:
            _fail("request_hash does not match ProviderRequest payload")
        return record


@dataclass(frozen=True)
class RawTransportResult(_CanonicalRecord):
    observed_provider: str
    observed_model_id: str
    observed_provider_revision: str
    response_text: str
    raw_response_bytes: bytes
    input_tokens: int
    output_tokens: int
    provider_request_id: Optional[str] = None

    def __post_init__(self) -> None:
        if type(self) is not RawTransportResult:
            _fail("RawTransportResult subclasses are not accepted")
        object.__setattr__(
            self,
            "observed_provider",
            _short_string("observed_provider", self.observed_provider),
        )
        object.__setattr__(
            self,
            "observed_model_id",
            _short_string("observed_model_id", self.observed_model_id),
        )
        object.__setattr__(
            self,
            "observed_provider_revision",
            _short_string(
                "observed_provider_revision", self.observed_provider_revision
            ),
        )
        object.__setattr__(
            self,
            "response_text",
            _response_text(self.response_text),
        )
        object.__setattr__(
            self, "raw_response_bytes", _raw_bytes(self.raw_response_bytes)
        )
        object.__setattr__(
            self, "input_tokens", _token("input_tokens", self.input_tokens)
        )
        object.__setattr__(
            self, "output_tokens", _token("output_tokens", self.output_tokens)
        )
        object.__setattr__(
            self,
            "provider_request_id",
            _optional_string("provider_request_id", self.provider_request_id, 1024),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RawTransportResult":
        if type(data) is not dict:
            _fail("RawTransportResult payload must be an exact dict")
        keys = (
            "observed_provider",
            "observed_model_id",
            "observed_provider_revision",
            "response_text",
            "raw_response_bytes",
            "input_tokens",
            "output_tokens",
            "provider_request_id",
        )
        _exact_fields(data, keys)
        return cls(
            data["observed_provider"],
            data["observed_model_id"],
            data["observed_provider_revision"],
            data["response_text"],
            _bytes_from_envelope(data["raw_response_bytes"], "raw_response_bytes"),
            data["input_tokens"],
            data["output_tokens"],
            data["provider_request_id"],
        )


_EXECUTION_FACTORY_GUARD = object()


@dataclass(frozen=True, init=False)
class ProviderExecution(_CanonicalRecord):
    """Sealed, internally derived provider execution evidence.

    ``from_dict`` validates complete internal derivation and canonical encoding. It
    cannot prove that a provider actually emitted the supplied transport artifact.
    """

    configuration: ProviderConfiguration
    request: ProviderRequest
    provider: str
    model_id: str
    provider_revision: str
    config_hash: str
    request_hash: str
    prompt_template_hash: str
    response_text: str
    raw_response_bytes: bytes
    raw_response_hash: str
    input_tokens: int
    output_tokens: int
    token_accounting_version: str
    provider_request_id: Optional[str]
    started_timestamp: str
    completed_timestamp: str
    latency_ms: float

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("ProviderExecution cannot be constructed directly")

    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError("ProviderExecution cannot be subclassed")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProviderExecution":
        """Reconstruct a canonically encoded, internally self-consistent artifact.

        This validates all embedded records and re-derives every identity projection
        and byte hash. It establishes artifact integrity, not provider authenticity.
        """
        if cls is not ProviderExecution:
            _fail("ProviderExecution subclasses are not accepted")
        if type(data) is not dict:
            _fail("ProviderExecution payload must be an exact dict")
        keys = tuple(item.name for item in fields(ProviderExecution))
        _exact_fields(data, keys)
        config = ProviderConfiguration.from_dict(data["configuration"])
        req = ProviderRequest.from_dict(data["request"])
        record = _derive_provider_execution(
            _EXECUTION_FACTORY_GUARD,
            config,
            req,
            response_text=data["response_text"],
            raw_response_bytes=_bytes_from_envelope(
                data["raw_response_bytes"], "raw_response_bytes"
            ),
            input_tokens=data["input_tokens"],
            output_tokens=data["output_tokens"],
            provider_request_id=data["provider_request_id"],
            started_timestamp=data["started_timestamp"],
            completed_timestamp=data["completed_timestamp"],
            latency_ms=data["latency_ms"],
        )
        if data["raw_response_hash"] != record.raw_response_hash:
            _fail("raw_response_hash does not match raw_response_bytes")
        if record.canonical_bytes() != _canonical_bytes(data):
            _fail("ProviderExecution payload is not exact canonical derived form")
        return record


def _derive_provider_execution(
    guard: object,
    config: ProviderConfiguration,
    req: ProviderRequest,
    *,
    response_text: Any,
    raw_response_bytes: Any,
    input_tokens: Any,
    output_tokens: Any,
    provider_request_id: Any,
    started_timestamp: ClockValue,
    completed_timestamp: ClockValue,
    latency_ms: Any,
) -> ProviderExecution:
    """The sole guarded construction path used by adapters and artifact loading."""
    if guard is not _EXECUTION_FACTORY_GUARD:
        raise TypeError("ProviderExecution factory is internal")
    _require_exact(config, ProviderConfiguration, "configuration")
    _require_exact(req, ProviderRequest, "request")
    canonical_config = ProviderConfiguration.from_dict(config.to_dict())
    canonical_request = ProviderRequest.from_dict(req.to_dict())
    captured_text = _response_text(response_text)
    captured_raw = _raw_bytes(raw_response_bytes)
    captured_input_tokens = _token("input_tokens", input_tokens)
    captured_output_tokens = _token("output_tokens", output_tokens)
    captured_request_id = _optional_string(
        "provider_request_id", provider_request_id, 1024
    )
    captured_started = _utc(started_timestamp, "started_timestamp")
    captured_completed = _utc(completed_timestamp, "completed_timestamp")
    captured_latency = _number("latency_ms", latency_ms)
    if captured_latency < 0:
        _fail("latency_ms must be nonnegative")

    record = object.__new__(ProviderExecution)
    values = {
        "configuration": canonical_config,
        "request": canonical_request,
        "provider": canonical_config.provider,
        "model_id": canonical_config.model_id,
        "provider_revision": canonical_config.provider_revision,
        "config_hash": canonical_config.config_hash,
        "request_hash": canonical_request.request_hash,
        "prompt_template_hash": canonical_request.prompt_template_hash,
        "response_text": captured_text,
        "raw_response_bytes": captured_raw,
        "raw_response_hash": "sha256:" + hashlib.sha256(captured_raw).hexdigest(),
        "input_tokens": captured_input_tokens,
        "output_tokens": captured_output_tokens,
        "token_accounting_version": canonical_config.token_accounting_version,
        "provider_request_id": captured_request_id,
        "started_timestamp": captured_started,
        "completed_timestamp": captured_completed,
        "latency_ms": captured_latency,
    }
    for name, value in values.items():
        object.__setattr__(record, name, value)
    return record


class Provider(Protocol):
    @property
    def configuration(self) -> ProviderConfiguration: ...
    def execute(self, request: ProviderRequest) -> ProviderExecution: ...


def _require_exact(value: Any, cls: Type[Any], name: str) -> None:
    if type(value) is not cls:
        raise ProviderValidationError(f"{name} must be exact {cls.__name__}")
    # Canonical reconstruction catches object.__setattr__ tampering.
    if hasattr(cls, "from_dict"):
        reconstructed = cls.from_dict(value.to_dict())
        if reconstructed != value:
            raise ProviderValidationError(f"{name} failed canonical reconstruction")


class RecordedCallbackProvider:
    def __init__(
        self,
        configuration: ProviderConfiguration,
        callback: Callable[
            [ProviderConfiguration, ProviderRequest], RawTransportResult
        ],
        utc_clock: Callable[[], ClockValue],
        monotonic_clock: Callable[[], float],
    ) -> None:
        _require_exact(configuration, ProviderConfiguration, "configuration")
        self._configuration = configuration
        self._callback = callback
        self._utc_clock = utc_clock
        self._monotonic_clock = monotonic_clock

    @property
    def configuration(self) -> ProviderConfiguration:
        return self._configuration

    def execute(self, request: ProviderRequest) -> ProviderExecution:
        _require_exact(request, ProviderRequest, "request")
        started_timestamp = _utc(self._utc_clock(), "started_timestamp")
        monotonic_start = _monotonic(self._monotonic_clock(), "monotonic_start")
        try:
            result = self._callback(self.configuration, request)
        except Exception as exc:
            completed_timestamp = _utc(self._utc_clock(), "completed_timestamp")
            monotonic_end = _monotonic(self._monotonic_clock(), "monotonic_end")
            if monotonic_end < monotonic_start:
                _fail("monotonic_end must be >= monotonic_start")
            raise ProviderCallbackError(
                "provider_callback_exception",
                started_timestamp,
                completed_timestamp,
                (monotonic_end - monotonic_start) * 1000.0,
            ) from exc
        completed_timestamp = _utc(self._utc_clock(), "completed_timestamp")
        monotonic_end = _monotonic(self._monotonic_clock(), "monotonic_end")
        if monotonic_end < monotonic_start:
            _fail("monotonic_end must be >= monotonic_start")
        if type(result) is not RawTransportResult:
            raise ProviderValidationError(
                "callback must return exact RawTransportResult"
            )
        result = RawTransportResult.from_dict(result.to_dict())
        if (
            result.observed_provider,
            result.observed_model_id,
            result.observed_provider_revision,
        ) != (
            self.configuration.provider,
            self.configuration.model_id,
            self.configuration.provider_revision,
        ):
            raise ProviderIdentityMismatchError(
                "transport provider/model/revision identity mismatch"
            )
        return _derive_provider_execution(
            _EXECUTION_FACTORY_GUARD,
            self.configuration,
            request,
            response_text=result.response_text,
            raw_response_bytes=result.raw_response_bytes,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            provider_request_id=result.provider_request_id,
            started_timestamp=started_timestamp,
            completed_timestamp=completed_timestamp,
            latency_ms=(monotonic_end - monotonic_start) * 1000.0,
        )


class DeterministicFakeProvider(RecordedCallbackProvider):
    def __init__(
        self,
        configuration: ProviderConfiguration,
        fixtures: Iterable[Tuple[str, RawTransportResult]],
        utc_clock: Callable[[], ClockValue],
        monotonic_clock: Callable[[], float],
    ) -> None:
        _require_exact(configuration, ProviderConfiguration, "configuration")
        copied: Dict[str, RawTransportResult] = {}
        for request_hash, result in fixtures:
            request_hash = _hash("request_hash", request_hash)
            if request_hash in copied:
                _fail("duplicate deterministic fake fixture request_hash")
            if type(result) is not RawTransportResult:
                _fail("fixtures must contain exact RawTransportResult")
            copied[request_hash] = RawTransportResult.from_dict(result.to_dict())
        self._fixtures = MappingProxyType(dict(copied))

        def callback(
            configuration: ProviderConfiguration, request: ProviderRequest
        ) -> RawTransportResult:
            _require_exact(configuration, ProviderConfiguration, "configuration")
            if request.request_hash not in self._fixtures:
                raise ProviderFixtureNotFoundError(
                    f"no fixture for request_hash {request.request_hash}"
                )
            return RawTransportResult.from_dict(
                self._fixtures[request.request_hash].to_dict()
            )

        super().__init__(configuration, callback, utc_clock, monotonic_clock)

    def execute(self, request: ProviderRequest) -> ProviderExecution:
        _require_exact(request, ProviderRequest, "request")
        if request.request_hash not in self._fixtures:
            raise ProviderFixtureNotFoundError(
                f"no fixture for request_hash {request.request_hash}"
            )
        return super().execute(request)


@dataclass(frozen=True)
class ManifestInputs(_CanonicalRecord):
    run_id: str
    experiment_version: str
    protocol_version: str
    dataset_version: str
    dataset_hash: str
    selector_mode: str
    selector_version: str
    code_revision: str
    provenance: str

    def __post_init__(self) -> None:
        if type(self) is not ManifestInputs:
            _fail("ManifestInputs subclasses are not accepted")
        for name in (
            "run_id",
            "experiment_version",
            "protocol_version",
            "dataset_version",
            "selector_mode",
            "selector_version",
            "code_revision",
        ):
            object.__setattr__(self, name, _short_string(name, getattr(self, name)))
        object.__setattr__(
            self, "dataset_hash", _hash("dataset_hash", self.dataset_hash)
        )
        object.__setattr__(
            self,
            "provenance",
            _short_string("provenance", self.provenance, _MAX_PROVENANCE),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ManifestInputs":
        if type(data) is not dict:
            _fail("ManifestInputs payload must be an exact dict")
        keys = tuple(field.name for field in fields(cls))
        _exact_fields(data, keys)
        return cls(**data)


Scalar = Union[None, bool, int, float, str, Tuple[Any, ...]]


def _difference_value(name: str, value: Any) -> Scalar:
    if value is None or type(value) in (bool, str):
        if type(value) is str and len(value) > _MAX_PROVENANCE:
            _fail(f"{name} string is too long")
        return cast(Scalar, value)
    if type(value) is int:
        _int64(name, value)
        return value
    if type(value) is float:
        _number(name, value)
        return value
    if type(value) in (list, tuple):
        return tuple(_difference_value(name, item) for item in value)
    _fail(f"{name} must be a canonical scalar or tuple")


@dataclass(frozen=True)
class ManifestDifference(_CanonicalRecord):
    field: str
    left: Scalar
    right: Scalar

    def __post_init__(self) -> None:
        if type(self) is not ManifestDifference:
            _fail("ManifestDifference subclasses are not accepted")
        object.__setattr__(self, "field", _short_string("field", self.field))
        object.__setattr__(self, "left", _difference_value("left", self.left))
        object.__setattr__(self, "right", _difference_value("right", self.right))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ManifestDifference":
        if type(data) is not dict:
            _fail("ManifestDifference payload must be an exact dict")
        _exact_fields(data, ("field", "left", "right"))
        return cls(data["field"], data["left"], data["right"])


@dataclass(frozen=True)
class ManifestComparison(_CanonicalRecord):
    left_run_id: str
    right_run_id: str
    differences: Tuple[ManifestDifference, ...]
    valid_for_primary_comparison: bool
    override_applied: bool
    label: str
    override_reason: Optional[str]

    def __post_init__(self) -> None:
        if type(self) is not ManifestComparison:
            _fail("ManifestComparison subclasses are not accepted")
        object.__setattr__(
            self, "left_run_id", _short_string("left_run_id", self.left_run_id)
        )
        object.__setattr__(
            self, "right_run_id", _short_string("right_run_id", self.right_run_id)
        )
        object.__setattr__(self, "differences", tuple(self.differences))
        if any(type(item) is not ManifestDifference for item in self.differences):
            _fail("differences must contain exact ManifestDifference records")
        if (
            type(self.valid_for_primary_comparison) is not bool
            or type(self.override_applied) is not bool
        ):
            _fail("comparison booleans must be exact bool")
        object.__setattr__(self, "label", _short_string("label", self.label))
        object.__setattr__(
            self,
            "override_reason",
            _optional_string("override_reason", self.override_reason, _MAX_PROVENANCE),
        )
        if not self.differences:
            if (
                not self.valid_for_primary_comparison
                or self.override_applied
                or self.override_reason is not None
                or self.label != "primary_comparison"
            ):
                _fail(
                    "compatible comparison cannot carry override or invalid primary flag"
                )
        else:
            if self.override_applied:
                if (
                    self.valid_for_primary_comparison
                    or self.override_reason is None
                    or self.label != "invalid_primary_comparison"
                ):
                    _fail(
                        "override comparisons are invalid for primary comparison and require a reason"
                    )
            elif (
                self.valid_for_primary_comparison
                or self.override_reason is not None
                or self.label != "incompatible"
            ):
                _fail(
                    "incompatible comparison without override must be invalid and reasonless"
                )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ManifestComparison":
        if type(data) is not dict:
            _fail("ManifestComparison payload must be an exact dict")
        keys = tuple(field.name for field in fields(cls))
        _exact_fields(data, keys)
        return cls(
            data["left_run_id"],
            data["right_run_id"],
            tuple(ManifestDifference.from_dict(item) for item in data["differences"]),
            data["valid_for_primary_comparison"],
            data["override_applied"],
            data["label"],
            data["override_reason"],
        )


def _config_from_provider(
    provider: Union[Provider, ProviderConfiguration],
) -> ProviderConfiguration:
    config = (
        provider
        if type(provider) is ProviderConfiguration
        else getattr(provider, "configuration", None)
    )
    _require_exact(config, ProviderConfiguration, "provider.configuration")
    return cast(ProviderConfiguration, config)


def _reconstruct_manifest(manifest: RunManifest) -> RunManifest:
    if type(manifest) is not RunManifest:
        raise ManifestConsistencyError("manifest must be exact RunManifest")
    try:
        reconstructed = RunManifest.from_dict(manifest.to_dict())
    except ValueError as exc:
        raise ManifestConsistencyError(
            "manifest failed canonical reconstruction"
        ) from exc
    if reconstructed != manifest:
        raise ManifestConsistencyError("manifest failed canonical reconstruction")
    return reconstructed


def build_run_manifest(
    inputs: ManifestInputs,
    provider: Union[Provider, ProviderConfiguration],
    request: ProviderRequest,
    utc_clock: Callable[[], ClockValue],
) -> RunManifest:
    _require_exact(inputs, ManifestInputs, "inputs")
    _require_exact(request, ProviderRequest, "request")
    config = _config_from_provider(provider)
    return RunManifest(
        run_id=inputs.run_id,
        experiment_version=inputs.experiment_version,
        protocol_version=inputs.protocol_version,
        dataset_version=inputs.dataset_version,
        dataset_hash=inputs.dataset_hash,
        selector_mode=inputs.selector_mode,
        selector_version=inputs.selector_version,
        provider=config.provider,
        model_id=config.model_id,
        prompt_template_hash=request.prompt_template_hash,
        config_hash=config.config_hash,
        code_revision=inputs.code_revision,
        temperature=config.temperature,
        seed=config.seed,
        seed_supported=config.seed_supported,
        tool_availability=config.tool_availability,
        started_timestamp=_utc(utc_clock(), "started_timestamp"),
        provenance=inputs.provenance,
    )


def validate_request_manifest(
    manifest: RunManifest,
    provider: Union[Provider, ProviderConfiguration],
    request: ProviderRequest,
) -> None:
    manifest = _reconstruct_manifest(manifest)
    config = _config_from_provider(provider)
    _require_exact(request, ProviderRequest, "request")
    expected = {
        "provider": config.provider,
        "model_id": config.model_id,
        "prompt_template_hash": request.prompt_template_hash,
        "config_hash": config.config_hash,
        "temperature": config.temperature,
        "seed": config.seed,
        "seed_supported": config.seed_supported,
        "tool_availability": config.tool_availability,
    }
    mismatches = [
        name for name, value in expected.items() if getattr(manifest, name) != value
    ]
    if mismatches:
        raise ManifestConsistencyError(
            "manifest binding mismatch: " + ", ".join(mismatches)
        )


def validate_execution(
    manifest: RunManifest,
    provider: Union[Provider, ProviderConfiguration],
    request: ProviderRequest,
    execution: ProviderExecution,
) -> None:
    validate_request_manifest(manifest, provider, request)
    _require_exact(execution, ProviderExecution, "execution")
    config = _config_from_provider(provider)
    expected = {
        "configuration": config,
        "request": request,
        "provider": config.provider,
        "model_id": config.model_id,
        "provider_revision": config.provider_revision,
        "config_hash": config.config_hash,
        "request_hash": request.request_hash,
        "prompt_template_hash": request.prompt_template_hash,
    }
    mismatches = [
        name for name, value in expected.items() if getattr(execution, name) != value
    ]
    if mismatches:
        raise ManifestConsistencyError(
            "execution binding mismatch: " + ", ".join(mismatches)
        )


def compare_manifests(
    left: RunManifest, right: RunManifest, override_reason: Optional[str] = None
) -> ManifestComparison:
    left = _reconstruct_manifest(left)
    right = _reconstruct_manifest(right)
    differences = tuple(
        ManifestDifference(field, getattr(left, field), getattr(right, field))
        for field in PRIMARY_COMPARABILITY_FIELDS
        if getattr(left, field) != getattr(right, field)
    )
    if differences:
        if override_reason is not None:
            _short_string("override_reason", override_reason, _MAX_PROVENANCE)
            return ManifestComparison(
                left.run_id,
                right.run_id,
                differences,
                False,
                True,
                "invalid_primary_comparison",
                override_reason,
            )
        raise IncompatibleManifestError(differences)
    if override_reason is not None:
        _fail("override_reason is only allowed for incompatible manifests")
    return ManifestComparison(
        left.run_id, right.run_id, (), True, False, "primary_comparison", None
    )


__all__ = [
    "TOKEN_ACCOUNTING_VERSION",
    "PRIMARY_COMPARABILITY_FIELDS",
    "ProviderError",
    "ProviderCallbackError",
    "ProviderValidationError",
    "ProviderIdentityMismatchError",
    "ProviderFixtureNotFoundError",
    "TokenAccountingUnavailableError",
    "ManifestConsistencyError",
    "IncompatibleManifestError",
    "ProviderConfiguration",
    "ProviderRequest",
    "RawTransportResult",
    "ProviderExecution",
    "Provider",
    "RecordedCallbackProvider",
    "DeterministicFakeProvider",
    "ManifestInputs",
    "ManifestDifference",
    "ManifestComparison",
    "build_run_manifest",
    "validate_request_manifest",
    "validate_execution",
    "compare_manifests",
]
