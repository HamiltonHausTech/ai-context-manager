"""Offline-testable implementation of the frozen Terra one-call probe.

The OpenAI SDK, httpx transport, and ignored credential are loaded only by the
explicitly paid command path. Library callers inject a client through the existing
provider/schema package seam.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import logging
import os
import pwd
import re
import ssl
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, cast

from .providers import (
    TOKEN_ACCOUNTING_VERSION,
    ManifestInputs,
    ProviderCallbackError,
    ProviderConfiguration,
    ProviderExecution,
    ProviderRequest,
    RawTransportResult,
    RecordedCallbackProvider,
    build_run_manifest,
    validate_execution,
)
from .schema import RunManifest

PINNED_MANIFEST_SHA256 = (
    "bd9d481ad9287da993b09d6412f2df7de9131665958d31597db6eb810a6abedc"
)
PINNED_OPENAI_SDK_VERSION = "2.46.0"
PINNED_CONFIGURATION_HASH = (
    "sha256:3de3fe8fd510175bedd3e086ce19840831bc685e2e58381f215263a3b1144426"
)
PINNED_PROMPT_TEMPLATE_HASH = (
    "sha256:739aea3f24ef4ce01763a0506b475f651240f5f7e6e2c63269617244fc68ba35"
)
OFFICIAL_OPENAI_BASE_URL = "https://api.openai.com/v1"
OFFICIAL_OPENAI_WEBSOCKET_URL = "wss://api.openai.com/v1"
MAX_RAW_RESPONSE_BYTES = 50 * 1024 * 1024
MAX_DOTENV_BYTES = 16 * 1024
ARTIFACT_VERSION = "terra-development-probe-artifact-v1"
ATTEMPT_MARKER_VERSION = "terra-development-probe-attempt-v1"
PROTOCOL_VERSION = "terra-bridge-development-v1"
CONTRACT_VERSION = "terra-config-probe-v1"
OUTPUT_DIRECTORY = Path(".local/adaptive-selection-probes")
MANIFEST_PATH = Path("experiments/adaptive_selection/controls/terra_probe_v1.json")
PROTOCOL_PATH = Path("docs/research/terra-bridge-development-protocol.md")
IMPLEMENTATION_PATH = Path("experiments/adaptive_selection/openai_manifest_probe.py")
FROZEN_PROTECTED_PATHS = (
    ".hermes/plans/2026-07-30-terra-bridge-preparation.md",
    "docs/research/adaptive-context-protocol-amendments.md",
    "docs/research/terra-bridge-development-protocol.md",
    "experiments/adaptive_selection/controls/terra_probe_v1.json",
    "experiments/adaptive_selection/providers.py",
    "experiments/adaptive_selection/schema.py",
    "experiments/adaptive_selection/openai_manifest_probe.py",
    "tests/adaptive_selection/test_openai_manifest_probe.py",
    "tests/adaptive_selection/test_runner.py",
    "tests/adaptive_selection/test_schema.py",
    "tests/adaptive_selection/test_packaging.py",
)
FROZEN_COST_PROJECTION = {
    "projected_input_tokens": 1984,
    "projected_output_tokens": 512,
    "projected_cost": "0.010112",
    "maximum_projected_cost": "0.25",
}
_ALLOWED_FAILURES = frozenset(
    {
        "preflight_rejected_no_network_attempt",
        "transport_failure_server_acceptance_unknown",
        "http_error",
        "parse_error",
        "provider_identity_mismatch",
        "invalid_response_status",
        "refusal",
        "incomplete_response",
        "invalid_structured_output",
        "missing_or_invalid_usage",
        "missing_provider_request_id",
        "missing_response_id",
        "artifact_write_failure",
    }
)
_SERVER_ACCEPTANCE = frozenset({"no", "yes", "unknown"})


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _utc_now() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _json_copy(value: Any) -> Any:
    return json.loads(_canonical_bytes(value).decode("utf-8"))


def _valid_provider_request_id(value: Any) -> bool:
    return (
        type(value) is str
        and re.fullmatch(r"req[-_][A-Za-z0-9_-]{1,120}", value) is not None
    )


def _valid_response_id(value: Any) -> bool:
    return (
        type(value) is str
        and re.fullmatch(r"resp[-_][A-Za-z0-9_-]{1,120}", value) is not None
    )


class ProbeFailure(Exception):
    """A fixed-category failure containing no provider exception text."""

    def __init__(
        self,
        category: str,
        server_acceptance: str,
        http_status_code: Optional[int] = None,
        provider_request_id: Optional[str] = None,
    ) -> None:
        if category not in _ALLOWED_FAILURES:
            raise ValueError("unsupported probe failure category")
        if server_acceptance not in _SERVER_ACCEPTANCE:
            raise ValueError("unsupported server acceptance value")
        if http_status_code is not None and type(http_status_code) is not int:
            raise ValueError("HTTP status must be an exact integer or null")
        if provider_request_id is not None and not _valid_provider_request_id(
            provider_request_id
        ):
            raise ValueError("provider request ID must be nonempty or null")
        self.category = category
        self.server_acceptance = server_acceptance
        self.http_status_code = http_status_code
        self.provider_request_id = provider_request_id
        super().__init__("probe failed: " + category)


def load_probe_contract(path: Path) -> Dict[str, Any]:
    """Load and hash-check the frozen literal contract without ambient state."""
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != PINNED_MANIFEST_SHA256:
        raise ProbeFailure("preflight_rejected_no_network_attempt", "no") from None
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ProbeFailure("preflight_rejected_no_network_attempt", "no") from None
    if type(value) is not dict or value.get("contract_version") != CONTRACT_VERSION:
        raise ProbeFailure("preflight_rejected_no_network_attempt", "no") from None
    if type(value.get("request_body")) is not dict:
        raise ProbeFailure("preflight_rejected_no_network_attempt", "no") from None
    return cast(Dict[str, Any], value)


def build_probe_records(
    contract: Mapping[str, Any],
    code_revision: str,
    utc_clock: Callable[[], str] = _utc_now,
) -> Tuple[ProviderConfiguration, ProviderRequest, RunManifest]:
    """Bind the literal request projection to existing provider evidence records."""
    request_body = _json_copy(contract["request_body"])
    request_hash = _sha256(_canonical_bytes(request_body))
    configuration = ProviderConfiguration(
        provider=contract["identity"]["provider"],
        model_id=contract["identity"]["requested_model"],
        provider_revision=None,
        temperature=None,
        temperature_supported=False,
        seed=None,
        seed_supported=False,
        tool_availability=(),
        token_accounting_version=TOKEN_ACCOUNTING_VERSION,
        generation_options=request_body,
    )
    request = ProviderRequest(
        prompt_text=request_body["input"], prompt_template_hash=request_hash
    )
    inputs = ManifestInputs(
        run_id="terra-development-manifest-probe-v1",
        experiment_version=CONTRACT_VERSION,
        protocol_version=PROTOCOL_VERSION,
        dataset_version="no-dataset-development-probe-v1",
        dataset_hash=_sha256(b"no-dataset-development-probe-v1"),
        selector_mode="none-development-probe",
        selector_version="none-development-probe-v1",
        code_revision=code_revision,
        provenance="terra-development-probe-contract",
    )
    manifest = build_run_manifest(inputs, configuration, request, utc_clock)
    return configuration, request, manifest


def projected_cost(contract: Mapping[str, Any]) -> Dict[str, Any]:
    """Apply the contract's deliberately conservative byte-as-token cost guard."""
    try:
        body_bytes = len(_canonical_bytes(contract["request_body"]))
        budget = contract["budget"]
        projected_input = body_bytes + 1024
        maximum_input = budget["maximum_projected_input_tokens"]
        output_tokens = budget["projected_worst_case_output_tokens"]
        raw_input_rate = budget["input_per_million"]
        raw_output_rate = budget["output_per_million"]
        raw_maximum_cost = budget["maximum_projected_cost"]
        if (
            type(maximum_input) is not int
            or type(output_tokens) is not int
            or type(raw_input_rate) is not str
            or type(raw_output_rate) is not str
            or type(raw_maximum_cost) is not str
        ):
            raise TypeError
        input_rate = Decimal(raw_input_rate)
        output_rate = Decimal(raw_output_rate)
        maximum_cost = Decimal(raw_maximum_cost)
        cost = (
            Decimal(projected_input) * input_rate + Decimal(output_tokens) * output_rate
        ) / Decimal(1_000_000)
    except (InvalidOperation, KeyError, TypeError, ValueError):
        raise ProbeFailure("preflight_rejected_no_network_attempt", "no") from None
    if (
        maximum_input < 0
        or output_tokens < 0
        or not input_rate.is_finite()
        or not output_rate.is_finite()
        or not maximum_cost.is_finite()
        or input_rate < 0
        or output_rate < 0
        or maximum_cost < 0
        or projected_input > maximum_input
        or cost > maximum_cost
    ):
        raise ProbeFailure("preflight_rejected_no_network_attempt", "no") from None
    return {
        "projected_input_tokens": projected_input,
        "projected_output_tokens": output_tokens,
        "projected_cost": format(cost, "f"),
        "maximum_projected_cost": format(maximum_cost, "f"),
    }


def _run_git(repo_root: Path, args: Sequence[str]) -> bytes:
    try:
        return subprocess.run(
            ["git"] + list(args),
            cwd=str(repo_root),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        raise ProbeFailure("preflight_rejected_no_network_attempt", "no") from None


def _git_commit_for_path(repo_root: Path, path: str) -> str:
    value = _run_git(repo_root, ["log", "-1", "--format=%H", "--", path])
    commit = value.decode("ascii", "strict").strip()
    if not commit:
        raise ProbeFailure("preflight_rejected_no_network_attempt", "no") from None
    return commit


def run_preflight(
    repo_root: Path,
    contract: Mapping[str, Any],
    output_dir: Path,
    *,
    authority_dir: Optional[Path] = None,
    implementation_path: str = str(IMPLEMENTATION_PATH),
) -> Dict[str, Any]:
    """Enforce tracked/clean/hash/ancestry/output/budget gates before dispatch."""
    protected = contract.get("preflight", {}).get("protected_paths")
    if (
        type(protected) is not list
        or not protected
        or any(type(item) is not str or not item for item in protected)
        or tuple(protected) != FROZEN_PROTECTED_PATHS
    ):
        raise ProbeFailure("preflight_rejected_no_network_attempt", "no") from None
    for relative in protected:
        _run_git(repo_root, ["cat-file", "-e", "HEAD:" + relative])
        if _run_git(
            repo_root,
            ["status", "--porcelain", "--untracked-files=all", "--", relative],
        ).strip():
            raise ProbeFailure("preflight_rejected_no_network_attempt", "no") from None
        head_bytes = _run_git(repo_root, ["show", "HEAD:" + relative])
        index_bytes = _run_git(repo_root, ["show", ":" + relative])
        try:
            worktree_bytes = (repo_root / relative).read_bytes()
        except OSError:
            raise ProbeFailure("preflight_rejected_no_network_attempt", "no") from None
        if worktree_bytes != head_bytes or index_bytes != head_bytes:
            raise ProbeFailure("preflight_rejected_no_network_attempt", "no") from None

    manifest_relative = str(MANIFEST_PATH)
    manifest_bytes = (repo_root / manifest_relative).read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != PINNED_MANIFEST_SHA256:
        raise ProbeFailure("preflight_rejected_no_network_attempt", "no") from None
    try:
        protocol_text = (repo_root / PROTOCOL_PATH).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        raise ProbeFailure("preflight_rejected_no_network_attempt", "no") from None
    pinned_line = "- **Manifest SHA-256:** `" + PINNED_MANIFEST_SHA256 + "`"
    if protocol_text.count(pinned_line) != 1:
        raise ProbeFailure("preflight_rejected_no_network_attempt", "no") from None
    canonical_output = repo_root / OUTPUT_DIRECTORY
    if (
        output_dir != canonical_output
        or output_dir.is_symlink()
        or canonical_output.parent.is_symlink()
    ):
        raise ProbeFailure("preflight_rejected_no_network_attempt", "no") from None
    try:
        if output_dir.resolve() != canonical_output.resolve():
            raise OSError
        _run_git(repo_root, ["check-ignore", "-q", "--", str(OUTPUT_DIRECTORY)])
        if output_dir.exists() and (
            not output_dir.is_dir() or any(output_dir.iterdir())
        ):
            raise OSError
    except OSError:
        raise ProbeFailure("preflight_rejected_no_network_attempt", "no") from None
    if authority_dir is not None:
        try:
            if authority_dir.is_symlink() or (
                authority_dir.exists()
                and (not authority_dir.is_dir() or any(authority_dir.iterdir()))
            ):
                raise OSError
        except OSError:
            raise ProbeFailure("preflight_rejected_no_network_attempt", "no") from None

    contract_commit = _git_commit_for_path(repo_root, manifest_relative)
    implementation_commit = _git_commit_for_path(repo_root, implementation_path)
    if contract_commit == implementation_commit:
        raise ProbeFailure("preflight_rejected_no_network_attempt", "no") from None
    _run_git(
        repo_root,
        ["merge-base", "--is-ancestor", contract_commit, implementation_commit],
    )
    head = _run_git(repo_root, ["rev-parse", "HEAD"]).decode("ascii").strip()
    costs = projected_cost(contract)
    hashes = {
        relative: _sha256((repo_root / relative).read_bytes()) for relative in protected
    }
    return {
        "code_revision": head,
        "contract_commit": contract_commit,
        "implementation_commit": implementation_commit,
        "contract_hash": _sha256(manifest_bytes),
        "protected_file_hashes": hashes,
        "cost_projection": costs,
    }


def _exact_nonnegative_int(value: Any) -> int:
    if type(value) is not int or not 0 <= value <= 2**63 - 1:
        raise ProbeFailure("missing_or_invalid_usage", "yes") from None
    return cast(int, value)


def _member(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _required_sdk_usage(response: Any, raw_usage: Mapping[str, Any]) -> None:
    sdk_usage = getattr(response, "usage", None)
    if sdk_usage is None:
        raise ProbeFailure("missing_or_invalid_usage", "yes") from None
    pairs = (
        ("input_tokens", sdk_usage),
        ("output_tokens", sdk_usage),
        ("total_tokens", sdk_usage),
    )
    for name, parent in pairs:
        sdk_value = _member(parent, name)
        if type(sdk_value) is not int or sdk_value != raw_usage[name]:
            raise ProbeFailure("missing_or_invalid_usage", "yes") from None
    detail_pairs = (
        ("input_tokens_details", "cached_tokens"),
        ("output_tokens_details", "reasoning_tokens"),
    )
    for parent_name, child_name in detail_pairs:
        raw_parent = raw_usage.get(parent_name)
        if raw_parent is None or child_name not in raw_parent:
            continue
        sdk_parent = _member(sdk_usage, parent_name)
        sdk_value = _member(sdk_parent, child_name)
        if type(sdk_value) is not int or sdk_value != raw_parent[child_name]:
            raise ProbeFailure("missing_or_invalid_usage", "yes") from None


def _validate_raw_usage(
    raw_document: Mapping[str, Any],
) -> Tuple[Mapping[str, Any], int, int]:
    usage = raw_document.get("usage")
    if type(usage) is not dict:
        raise ProbeFailure("missing_or_invalid_usage", "yes") from None
    for name in ("input_tokens", "output_tokens", "total_tokens"):
        if name not in usage:
            raise ProbeFailure("missing_or_invalid_usage", "yes") from None
        _exact_nonnegative_int(usage[name])
    for parent_name, child_name in (
        ("input_tokens_details", "cached_tokens"),
        ("output_tokens_details", "reasoning_tokens"),
    ):
        parent = usage.get(parent_name)
        if parent is None:
            continue
        if type(parent) is not dict:
            raise ProbeFailure("missing_or_invalid_usage", "yes") from None
        if child_name in parent:
            _exact_nonnegative_int(parent[child_name])
    if usage["total_tokens"] != usage["input_tokens"] + usage["output_tokens"]:
        raise ProbeFailure("missing_or_invalid_usage", "yes") from None
    return usage, usage["input_tokens"], usage["output_tokens"]


def _request_id_from_exception(exc: BaseException) -> Optional[str]:
    value = getattr(exc, "request_id", None)
    if _valid_provider_request_id(value):
        return value
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        value = headers.get("x-request-id")
        if _valid_provider_request_id(value):
            return value
    return None


def _status_from_exception(exc: BaseException) -> Optional[int]:
    value = getattr(exc, "status_code", None)
    if type(value) is int:
        return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if type(value) is int else None


def _response_text(response: Any) -> str:
    value = getattr(response, "output_text", None)
    if type(value) is not str:
        raise ProbeFailure("invalid_structured_output", "yes") from None
    return value


def _raw_response_text(raw_document: Mapping[str, Any]) -> str:
    output = raw_document.get("output")
    if type(output) is not list or not output:
        raise ProbeFailure("invalid_structured_output", "yes") from None
    messages = []
    for item in output:
        if type(item) is not dict:
            raise ProbeFailure("invalid_structured_output", "yes") from None
        item_type = item.get("type")
        if item_type == "reasoning":
            if "content" in item or "action" in item or "arguments" in item:
                raise ProbeFailure("invalid_structured_output", "yes") from None
            continue
        if item_type != "message":
            raise ProbeFailure("invalid_structured_output", "yes") from None
        messages.append(item)
    if len(messages) != 1:
        raise ProbeFailure("invalid_structured_output", "yes") from None
    message = messages[0]
    if message.get("role") != "assistant":
        raise ProbeFailure("invalid_structured_output", "yes") from None
    content = message.get("content")
    if type(content) is not list or len(content) != 1:
        raise ProbeFailure("invalid_structured_output", "yes") from None
    part = content[0]
    if type(part) is not dict or part.get("type") != "output_text":
        raise ProbeFailure("invalid_structured_output", "yes") from None
    text = part.get("text")
    if type(text) is not str:
        raise ProbeFailure("invalid_structured_output", "yes") from None
    return text


def usage_cost_estimate(
    contract: Mapping[str, Any], usage: Mapping[str, Any]
) -> Dict[str, Any]:
    """Price provider-reported usage at frozen undiscounted standard rates."""
    try:
        input_tokens = _exact_nonnegative_int(usage["input_tokens"])
        output_tokens = _exact_nonnegative_int(usage["output_tokens"])
        input_rate = Decimal(contract["budget"]["input_per_million"])
        output_rate = Decimal(contract["budget"]["output_per_million"])
        cached_tokens = 0
        details = usage.get("input_tokens_details")
        if details is not None:
            if type(details) is not dict:
                raise TypeError
            if "cached_tokens" in details:
                cached_tokens = _exact_nonnegative_int(details["cached_tokens"])
        if cached_tokens > input_tokens:
            raise TypeError
        cost = (
            Decimal(input_tokens) * input_rate + Decimal(output_tokens) * output_rate
        ) / Decimal(1_000_000)
    except (InvalidOperation, KeyError, TypeError, ValueError):
        raise ProbeFailure("missing_or_invalid_usage", "yes") from None
    return {
        "currency": contract["budget"]["currency"],
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_input_tokens": cached_tokens,
        "input_per_million": format(input_rate, "f"),
        "output_per_million": format(output_rate, "f"),
        "estimated_cost": format(cost, "f"),
        "estimate_kind": (
            "frozen_standard_rates_exact_when_no_other_fees"
            if cached_tokens == 0
            else "upper_bound_cached_discount_not_frozen"
        ),
    }


def _contains_refusal(raw_document: Mapping[str, Any], response: Any) -> bool:
    def walk(value: Any) -> bool:
        if type(value) is dict:
            if value.get("type") == "refusal":
                return True
            refusal = value.get("refusal")
            if type(refusal) is str and refusal:
                return True
            return any(walk(item) for item in value.values())
        if type(value) is list:
            return any(walk(item) for item in value)
        return False

    if walk(raw_document):
        return True
    output = getattr(response, "output", ()) or ()
    for item in output:
        for content in getattr(item, "content", ()) or ():
            if getattr(content, "type", None) == "refusal" or getattr(
                content, "refusal", None
            ):
                return True
    return False


def _validate_structured_output(text: str) -> None:
    invalid_json = False
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        invalid_json = True
        value = None
    if invalid_json:
        raise ProbeFailure("invalid_structured_output", "yes") from None
    if type(value) is not dict or set(value) != {"probe_marker", "status", "notes"}:
        raise ProbeFailure("invalid_structured_output", "yes") from None
    if (
        value["probe_marker"] != "terra-manifest-probe-v1"
        or value["status"] != "ok"
        or type(value["notes"]) is not str
    ):
        raise ProbeFailure("invalid_structured_output", "yes") from None


def make_openai_callback(
    client: Any,
    request_body: Mapping[str, Any],
    before_dispatch: Optional[Callable[[], None]] = None,
) -> Callable[[ProviderConfiguration, ProviderRequest], RawTransportResult]:
    """Build the reusable, injected-client callback; never reads environment state."""
    literal_body = _json_copy(request_body)
    literal_bytes = _canonical_bytes(literal_body)

    def callback(
        configuration: ProviderConfiguration, request: ProviderRequest
    ) -> RawTransportResult:
        if (
            _canonical_bytes(configuration.to_dict()["generation_options"])
            != literal_bytes
            or request.prompt_text != literal_body["input"]
            or request.prompt_template_hash != _sha256(literal_bytes)
        ):
            raise ProbeFailure("preflight_rejected_no_network_attempt", "no") from None
        if before_dispatch is not None:
            before_dispatch()
        dispatch_failure: Optional[ProbeFailure] = None
        raw: Any = None
        try:
            raw = client.responses.with_raw_response.create(**_json_copy(literal_body))
        except Exception as exc:
            status = _status_from_exception(exc)
            request_id = _request_id_from_exception(exc)
            if status is not None:
                dispatch_failure = ProbeFailure("http_error", "yes", status, request_id)
            else:
                dispatch_failure = ProbeFailure(
                    "transport_failure_server_acceptance_unknown", "unknown"
                )
        if dispatch_failure is not None:
            raise dispatch_failure from None
        request_id: Optional[str] = None
        sanitized_failure: Optional[ProbeFailure] = None
        try:
            raw_bytes = raw.content
            if (
                type(raw_bytes) is not bytes
                or not raw_bytes
                or len(raw_bytes) > MAX_RAW_RESPONSE_BYTES
            ):
                raise ProbeFailure("parse_error", "yes") from None
            request_id = raw.request_id
            if not _valid_provider_request_id(request_id):
                request_id = None
                raise ProbeFailure("missing_provider_request_id", "yes") from None
            raw_document: Any = None
            try:
                raw_document = json.loads(raw_bytes)
            except (UnicodeDecodeError, json.JSONDecodeError):
                pass
            if type(raw_document) is not dict:
                raise ProbeFailure(
                    "parse_error", "yes", provider_request_id=request_id
                ) from None
            raw_usage, input_tokens, output_tokens = _validate_raw_usage(raw_document)
            response = raw.parse()
            response_id = getattr(response, "id", None)
            if not _valid_response_id(response_id):
                raise ProbeFailure(
                    "missing_response_id", "yes", provider_request_id=request_id
                ) from None
            if raw_document.get("id") != response_id or response_id == request_id:
                raise ProbeFailure(
                    "missing_response_id", "yes", provider_request_id=request_id
                ) from None
            observed_model = getattr(response, "model", None)
            if (
                type(observed_model) is not str
                or observed_model != configuration.model_id
                or raw_document.get("model") != observed_model
            ):
                raise ProbeFailure(
                    "provider_identity_mismatch",
                    "yes",
                    provider_request_id=request_id,
                ) from None
            status = getattr(response, "status", None)
            raw_status = raw_document.get("status")
            incomplete = getattr(response, "incomplete_details", None)
            raw_incomplete = raw_document.get("incomplete_details")
            if (
                status == "incomplete"
                or raw_status == "incomplete"
                or incomplete is not None
                or raw_incomplete is not None
            ):
                raise ProbeFailure(
                    "incomplete_response", "yes", provider_request_id=request_id
                ) from None
            if status != "completed" or raw_status != status:
                raise ProbeFailure(
                    "invalid_response_status", "yes", provider_request_id=request_id
                ) from None
            if _contains_refusal(raw_document, response):
                raise ProbeFailure(
                    "refusal", "yes", provider_request_id=request_id
                ) from None
            text = _response_text(response)
            if text != _raw_response_text(raw_document):
                raise ProbeFailure(
                    "invalid_structured_output", "yes", provider_request_id=request_id
                ) from None
            _validate_structured_output(text)
            _required_sdk_usage(response, raw_usage)
            return RawTransportResult(
                observed_provider="openai",
                observed_model_id=observed_model,
                observed_provider_revision=None,
                response_text=text,
                raw_response_bytes=raw_bytes,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                provider_request_id=request_id,
            )
        except ProbeFailure as caught_failure:
            if caught_failure.provider_request_id is None:
                sanitized_failure = ProbeFailure(
                    caught_failure.category,
                    caught_failure.server_acceptance,
                    caught_failure.http_status_code,
                    request_id,
                )
            else:
                sanitized_failure = caught_failure
        except Exception:
            sanitized_failure = ProbeFailure(
                "parse_error", "yes", provider_request_id=request_id
            )
        if sanitized_failure is not None:
            raise sanitized_failure from None
        raise ProbeFailure(
            "parse_error", "yes", provider_request_id=request_id
        ) from None

    return callback


def execute_probe(
    client: Any,
    contract: Mapping[str, Any],
    code_revision: str,
    *,
    before_dispatch: Optional[Callable[[], None]] = None,
    utc_clock: Callable[[], str] = _utc_now,
    monotonic_clock: Callable[[], float] = time.monotonic,
) -> Tuple[RunManifest, ProviderExecution, str]:
    """Execute once through RecordedCallbackProvider using an injected client."""
    configuration, request, manifest = build_probe_records(
        contract, code_revision, utc_clock
    )
    callback = make_openai_callback(client, contract["request_body"], before_dispatch)
    provider = RecordedCallbackProvider(
        configuration, callback, utc_clock, monotonic_clock
    )
    execution = provider.execute(request)
    validate_execution(manifest, provider, request, execution)
    raw_document = json.loads(execution.raw_response_bytes)
    response_id = raw_document["id"]
    return manifest, execution, response_id


def _ensure_private_directory(path: Path) -> None:
    if path.is_symlink():
        raise OSError("private directory must not be a symlink")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    before = path.lstat()
    if not stat.S_ISDIR(before.st_mode) or before.st_uid != os.geteuid():
        raise OSError("private directory ownership/type is invalid")
    os.chmod(str(path), 0o700)
    after = path.lstat()
    if (
        not stat.S_ISDIR(after.st_mode)
        or after.st_uid != os.geteuid()
        or stat.S_IMODE(after.st_mode) != 0o700
    ):
        raise OSError("private directory mode could not be established")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_no_overwrite(path: Path, data: bytes) -> None:
    """Publish fully fsynced bytes atomically without replacing an existing path."""
    _ensure_private_directory(path.parent)
    temporary_name: Optional[str] = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix="." + path.name + ".", dir=str(path.parent)
        )
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        except Exception:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise
        os.link(temporary_name, str(path))
        _fsync_directory(path.parent)
        os.unlink(temporary_name)
        temporary_name = None
        _fsync_directory(path.parent)
    except Exception:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass
        raise


def _artifact_envelope(kind: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
    if kind not in {"success", "failure"} or {
        "artifact_version",
        "kind",
    } & set(payload):
        raise ValueError("artifact kind/version are reserved")
    artifact = {"artifact_version": ARTIFACT_VERSION, "kind": kind, **dict(payload)}
    return {"artifact": artifact, "artifact_hash": _sha256(_canonical_bytes(artifact))}


def write_artifact(
    path: Path,
    kind: str,
    payload: Mapping[str, Any],
    *,
    repo_root: Optional[Path] = None,
    _protected_hash_resolver: Optional[Callable[[Path, str], Dict[str, str]]] = None,
) -> Dict[str, Any]:
    envelope = _artifact_envelope(kind, payload)
    data = _canonical_bytes(envelope) + b"\n"
    _atomic_write_no_overwrite(path, data)
    verified = verify_artifact(
        path,
        repo_root=repo_root,
        _protected_hash_resolver=_protected_hash_resolver,
    )
    if verified != envelope:
        raise OSError("artifact verification did not reproduce written bytes")
    return envelope


def _valid_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and value.startswith("sha256:")
        and len(value) == 71
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _valid_git_revision(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) in {40, 64}
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_timestamp(value: Any) -> bool:
    if type(value) is not str or not value.endswith("Z"):
        return False
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return False
    return True


def _protected_hashes_at_revision(repo_root: Path, revision: str) -> Dict[str, str]:
    if not _valid_git_revision(revision):
        raise ValueError("code revision is invalid")
    hashes = {}
    for relative in FROZEN_PROTECTED_PATHS:
        try:
            data = _run_git(repo_root, ["show", revision + ":" + relative])
        except ProbeFailure:
            raise ValueError("protected path is absent from code revision") from None
        hashes[relative] = _sha256(data)
    return hashes


def verify_artifact(
    path: Path,
    *,
    repo_root: Optional[Path] = None,
    _protected_hash_resolver: Optional[Callable[[Path, str], Dict[str, str]]] = None,
) -> Dict[str, Any]:
    if stat.S_IMODE(path.parent.stat().st_mode) != 0o700:
        raise ValueError("artifact directory mode is not private")
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise ValueError("artifact file mode is not private")
    data = path.read_bytes()
    if not data.endswith(b"\n"):
        raise ValueError("artifact is not canonical newline-terminated JSON")
    value = json.loads(data)
    if type(value) is not dict or set(value) != {"artifact", "artifact_hash"}:
        raise ValueError("artifact envelope fields are not exact")
    artifact = value["artifact"]
    if type(artifact) is not dict or value["artifact_hash"] != _sha256(
        _canonical_bytes(artifact)
    ):
        raise ValueError("whole-artifact hash mismatch")
    if data != _canonical_bytes(value) + b"\n":
        raise ValueError("artifact bytes are not canonical")
    if artifact.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError("artifact version is unsupported")

    if artifact.get("kind") == "success":
        required = {
            "artifact_version",
            "kind",
            "contract_hash",
            "configuration_hash",
            "code_revision",
            "request_attempted",
            "server_acceptance",
            "protected_file_hashes",
            "cost_projection",
            "provider_usage",
            "cost_estimate",
            "response_id",
            "run_manifest",
            "provider_execution",
        }
        if set(artifact) != required:
            raise ValueError("success artifact fields are not exact")
        execution = ProviderExecution.from_dict(artifact["provider_execution"])
        if execution.raw_response_hash != _sha256(execution.raw_response_bytes):
            raise ValueError("embedded raw response hash mismatch")
        manifest = RunManifest.from_dict(artifact["run_manifest"])
        if (
            manifest.config_hash != execution.config_hash
            or artifact["configuration_hash"] != execution.config_hash
            or artifact["configuration_hash"] != PINNED_CONFIGURATION_HASH
            or manifest.code_revision != artifact["code_revision"]
            or manifest.prompt_template_hash != PINNED_PROMPT_TEMPLATE_HASH
            or execution.provider_request_id is None
            or not _valid_provider_request_id(execution.provider_request_id)
        ):
            raise ValueError("embedded manifest/configuration hash mismatch")
        manifest_document = manifest.to_dict()
        expected_manifest_identity = {
            "run_id": "terra-development-manifest-probe-v1",
            "experiment_version": "terra-config-probe-v1",
            "protocol_version": "terra-bridge-development-v1",
            "dataset_version": "no-dataset-development-probe-v1",
            "dataset_hash": "sha256:b3595092b8a08d59f4571e3f3ea6641e61fdf5fc2469d6ef8888cbc242592124",
            "selector_mode": "none-development-probe",
            "selector_version": "none-development-probe-v1",
            "provider": "openai",
            "model_id": "gpt-5.6-terra",
            "provider_revision": None,
            "prompt_template_hash": PINNED_PROMPT_TEMPLATE_HASH,
            "config_hash": PINNED_CONFIGURATION_HASH,
            "temperature": None,
            "temperature_supported": False,
            "seed": None,
            "seed_supported": False,
            "tool_availability": [],
            "provenance": "terra-development-probe-contract",
            "schema_version": "3",
        }
        if any(
            manifest_document[name] != value
            for name, value in expected_manifest_identity.items()
        ):
            raise ValueError("embedded manifest contract mismatch")
        try:
            raw_document = json.loads(execution.raw_response_bytes)
            raw_usage, _, _ = _validate_raw_usage(raw_document)
        except (UnicodeDecodeError, json.JSONDecodeError, ProbeFailure):
            raise ValueError("embedded raw usage is invalid") from None
        if artifact["provider_usage"] != _json_copy(raw_usage):
            raise ValueError("provider usage does not match raw response")
        try:
            raw_text = _raw_response_text(raw_document)
            _validate_structured_output(raw_text)
        except ProbeFailure:
            raise ValueError("embedded structured output is invalid") from None
        if (
            raw_document.get("model") != "gpt-5.6-terra"
            or raw_document.get("status") != "completed"
            or raw_document.get("incomplete_details") is not None
            or _contains_refusal(raw_document, None)
            or raw_usage["input_tokens"] != execution.input_tokens
            or raw_usage["output_tokens"] != execution.output_tokens
            or raw_text != execution.response_text
        ):
            raise ValueError("embedded response does not re-establish success")
        cost = artifact["cost_estimate"]
        if (
            type(cost) is not dict
            or cost.get("currency") != "USD"
            or cost.get("input_per_million") != "2.00"
            or cost.get("output_per_million") != "12.00"
        ):
            raise ValueError("cost estimate is malformed")
        synthetic_contract = {
            "budget": {
                "currency": "USD",
                "input_per_million": "2.00",
                "output_per_million": "12.00",
            }
        }
        try:
            expected_cost = usage_cost_estimate(synthetic_contract, raw_usage)
        except ProbeFailure:
            raise ValueError("cost estimate is malformed") from None
        if cost != expected_cost:
            raise ValueError("cost estimate does not match provider usage")
        if (
            artifact["request_attempted"] is not True
            or artifact["server_acceptance"] != "yes"
            or not _valid_response_id(artifact["response_id"])
            or artifact["response_id"] == execution.provider_request_id
            or raw_document.get("id") != artifact["response_id"]
            or raw_text != execution.response_text
        ):
            raise ValueError("success request/response identity is invalid")
        protected = artifact["protected_file_hashes"]
        if (
            type(protected) is not dict
            or set(protected) != set(FROZEN_PROTECTED_PATHS)
            or any(
                type(name) is not str or not name or not _valid_sha256(digest)
                for name, digest in protected.items()
            )
        ):
            raise ValueError("protected file hashes are invalid")
        if _protected_hash_resolver is None:
            if repo_root is None:
                raise ValueError("repository root is required for success verification")
            expected_protected = _protected_hashes_at_revision(
                repo_root, artifact["code_revision"]
            )
        else:
            expected_protected = _protected_hash_resolver(
                repo_root or Path(), artifact["code_revision"]
            )
        if protected != expected_protected:
            raise ValueError("protected hashes do not match the code revision")
        if artifact["cost_projection"] != FROZEN_COST_PROJECTION:
            raise ValueError("projected cost does not match the frozen contract")
    elif artifact.get("kind") == "failure":
        required = {
            "artifact_version",
            "kind",
            "contract_hash",
            "configuration_hash",
            "code_revision",
            "request_attempted",
            "server_acceptance",
            "started_timestamp",
            "completed_timestamp",
            "latency_ms",
            "failure_category",
            "http_status_code",
            "provider_request_id",
        }
        if set(artifact) != required:
            raise ValueError("failure artifact fields are not exact")
        if (
            artifact["failure_category"] not in _ALLOWED_FAILURES
            or artifact["server_acceptance"] not in _SERVER_ACCEPTANCE
            or type(artifact["request_attempted"]) is not bool
            or artifact["contract_hash"] != "sha256:" + PINNED_MANIFEST_SHA256
            or artifact["configuration_hash"] != PINNED_CONFIGURATION_HASH
            or not _valid_git_revision(artifact["code_revision"])
            or not _valid_timestamp(artifact["started_timestamp"])
            or not _valid_timestamp(artifact["completed_timestamp"])
        ):
            raise ValueError("failure artifact values are malformed")
        latency = artifact["latency_ms"]
        if (
            type(latency) not in {int, float}
            or not Decimal(str(latency)).is_finite()
            or latency < 0
        ):
            raise ValueError("failure latency is invalid")
        status = artifact["http_status_code"]
        if status is not None and (type(status) is not int or not 100 <= status <= 599):
            raise ValueError("failure HTTP status is invalid")
        request_id = artifact["provider_request_id"]
        if request_id is not None and not _valid_provider_request_id(request_id):
            raise ValueError("failure provider request ID is invalid")
        started = datetime.fromisoformat(artifact["started_timestamp"][:-1] + "+00:00")
        completed = datetime.fromisoformat(
            artifact["completed_timestamp"][:-1] + "+00:00"
        )
        if completed < started:
            raise ValueError("failure timestamps are reversed")
    else:
        raise ValueError("artifact kind is unsupported")
    if artifact["contract_hash"] != "sha256:" + PINNED_MANIFEST_SHA256:
        raise ValueError("contract hash is invalid")
    if not _valid_git_revision(artifact["code_revision"]):
        raise ValueError("code revision is invalid")
    return cast(Dict[str, Any], value)


def verify_attempt_marker(path: Path) -> Dict[str, Any]:
    if stat.S_IMODE(path.parent.stat().st_mode) != 0o700:
        raise ValueError("attempt marker directory mode is not private")
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise ValueError("attempt marker mode is not private")
    data = path.read_bytes()
    if not data.endswith(b"\n"):
        raise ValueError("attempt marker is not newline terminated")
    value = json.loads(data)
    if type(value) is not dict or set(value) != {"attempt", "attempt_hash"}:
        raise ValueError("attempt marker envelope fields are not exact")
    attempt = value["attempt"]
    required = {
        "marker_version",
        "contract_hash",
        "code_revision",
        "request_attempted",
        "attempted_timestamp",
    }
    if type(attempt) is not dict or set(attempt) != required:
        raise ValueError("attempt marker fields are not exact")
    if (
        attempt["marker_version"] != ATTEMPT_MARKER_VERSION
        or attempt["contract_hash"] != "sha256:" + PINNED_MANIFEST_SHA256
        or not _valid_git_revision(attempt["code_revision"])
        or attempt["request_attempted"] is not True
        or not _valid_timestamp(attempt["attempted_timestamp"])
        or value["attempt_hash"] != _sha256(_canonical_bytes(attempt))
        or data != _canonical_bytes(value) + b"\n"
    ):
        raise ValueError("attempt marker values are invalid")
    return cast(Dict[str, Any], value)


def write_attempt_marker(
    output_dir: Path, contract_hash: str, code_revision: str, timestamp: str
) -> Dict[str, Any]:
    payload = {
        "marker_version": ATTEMPT_MARKER_VERSION,
        "contract_hash": contract_hash,
        "code_revision": code_revision,
        "request_attempted": True,
        "attempted_timestamp": timestamp,
    }
    marker = {"attempt": payload, "attempt_hash": _sha256(_canonical_bytes(payload))}
    path = output_dir / "attempt-consumed.json"
    _atomic_write_no_overwrite(path, _canonical_bytes(marker) + b"\n")
    if verify_attempt_marker(path) != marker:
        raise OSError("attempt marker verification did not reproduce written bytes")
    return marker


def _default_authority_directory() -> Path:
    account_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    if not account_home.is_absolute():
        raise ProbeFailure("preflight_rejected_no_network_attempt", "no") from None
    return (
        account_home
        / ".local/state/ai-context-manager/terra-probe-authority"
        / PINNED_MANIFEST_SHA256
    )


def _load_live_modules() -> Tuple[Any, Any]:
    try:
        openai = importlib.import_module("openai")
        httpx = importlib.import_module("httpx")
    except Exception:
        raise ProbeFailure("preflight_rejected_no_network_attempt", "no") from None
    if getattr(openai, "__version__", None) != PINNED_OPENAI_SDK_VERSION:
        raise ProbeFailure("preflight_rejected_no_network_attempt", "no") from None
    return openai, httpx


PAID_AMBIENT_ENVIRONMENT = (
    "OPENAI_LOG",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
)


def _begin_paid_ambient_guard() -> Tuple[Dict[str, Optional[str]], int]:
    saved = {name: os.environ.get(name) for name in PAID_AMBIENT_ENVIRONMENT}
    for name in PAID_AMBIENT_ENVIRONMENT:
        os.environ.pop(name, None)
    previous_logging_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    return saved, previous_logging_disable


def _end_paid_ambient_guard(
    saved: Mapping[str, Optional[str]], previous_logging_disable: int
) -> None:
    for name in PAID_AMBIENT_ENVIRONMENT:
        value = saved[name]
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    logging.disable(previous_logging_disable)


def _create_live_client(api_key: str) -> Any:
    """Build the sole official-host client with no retries, redirects, or env trust."""
    ambient_names = (
        "OPENAI_API_KEY",
        "OPENAI_ADMIN_KEY",
        "OPENAI_ORG_ID",
        "OPENAI_PROJECT_ID",
        "OPENAI_WEBHOOK_SECRET",
        "OPENAI_BASE_URL",
        "OPENAI_CUSTOM_HEADERS",
        "OPENAI_LOG",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "SSLKEYLOGFILE",
    )
    saved = {name: os.environ.get(name) for name in ambient_names}
    for name in ambient_names:
        os.environ.pop(name, None)
    client: Any = None
    http_client: Any = None
    failed = False
    try:
        openai, httpx = _load_live_modules()
        ssl_context = ssl.create_default_context()
        if ssl_context.keylog_filename is not None:
            raise RuntimeError("TLS key logging remained enabled")
        http_client = httpx.Client(
            verify=ssl_context,
            timeout=30.0,
            trust_env=False,
            follow_redirects=False,
        )
        client = openai.OpenAI(
            api_key=api_key,
            base_url=OFFICIAL_OPENAI_BASE_URL,
            websocket_base_url=OFFICIAL_OPENAI_WEBSOCKET_URL,
            max_retries=0,
            timeout=30.0,
            default_headers={},
            default_query={},
            http_client=http_client,
        )
    except Exception:
        failed = True
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    if failed or client is None:
        if http_client is not None:
            try:
                http_client.close()
            except Exception:
                pass
        raise ProbeFailure("preflight_rejected_no_network_attempt", "no") from None
    return client


def _dotenv_is_untracked(repo_root: Path) -> bool:
    try:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", ".env"],
            cwd=str(repo_root),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False
    return result.returncode == 1


def _load_ignored_api_key(repo_root: Path) -> str:
    path = repo_root / ".env"
    _run_git(repo_root, ["check-ignore", "-q", "--", ".env"])
    if not _dotenv_is_untracked(repo_root):
        raise ProbeFailure("preflight_rejected_no_network_attempt", "no") from None
    descriptor: Optional[int] = None
    read_failed = False
    text = ""
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(str(path), flags)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_nlink != 1
            or not 0 < metadata.st_size <= MAX_DOTENV_BYTES
        ):
            raise OSError
        data = b""
        while len(data) <= MAX_DOTENV_BYTES:
            chunk = os.read(descriptor, min(4096, MAX_DOTENV_BYTES + 1 - len(data)))
            if not chunk:
                break
            data += chunk
        if len(data) > MAX_DOTENV_BYTES:
            raise OSError
        text = data.decode("utf-8")
    except (OSError, UnicodeDecodeError):
        read_failed = True
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if read_failed:
        raise ProbeFailure("preflight_rejected_no_network_attempt", "no") from None
    values = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ProbeFailure("preflight_rejected_no_network_attempt", "no") from None
        name, raw_value = line.split("=", 1)
        if name != "OPENAI_API_KEY":
            continue
        if raw_value != raw_value.strip() or not raw_value:
            raise ProbeFailure("preflight_rejected_no_network_attempt", "no") from None
        if raw_value[0] in "\"'":
            if len(raw_value) < 2 or raw_value[-1] != raw_value[0]:
                raise ProbeFailure(
                    "preflight_rejected_no_network_attempt", "no"
                ) from None
            value = raw_value[1:-1]
        else:
            if '"' in raw_value or "'" in raw_value:
                raise ProbeFailure(
                    "preflight_rejected_no_network_attempt", "no"
                ) from None
            value = raw_value
        values.append(value)
    if len(values) != 1:
        raise ProbeFailure("preflight_rejected_no_network_attempt", "no") from None
    value = values[0]
    if not 20 <= len(value) <= 1024 or any(
        ord(character) < 33 or ord(character) > 126 for character in value
    ):
        raise ProbeFailure("preflight_rejected_no_network_attempt", "no") from None
    return value


def _contains_secret(value: Any, secret: str) -> bool:
    return secret.encode("utf-8") in _canonical_bytes(value)


def _assert_api_key_absent_from_git(repo_root: Path, api_key: str) -> None:
    archive = _run_git(repo_root, ["archive", "--format=tar", "HEAD"])
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as stream:
            for member in stream.getmembers():
                if not member.isfile():
                    continue
                extracted = stream.extractfile(member)
                if (
                    extracted is not None
                    and api_key.encode("utf-8") in extracted.read()
                ):
                    raise ProbeFailure(
                        "preflight_rejected_no_network_attempt", "no"
                    ) from None
    except (OSError, tarfile.TarError):
        raise ProbeFailure("preflight_rejected_no_network_attempt", "no") from None


def _failure_from_callback(exc: ProviderCallbackError) -> ProbeFailure:
    cause = exc.__cause__
    if type(cause) is ProbeFailure:
        return cast(ProbeFailure, cause)
    return ProbeFailure("parse_error", "unknown")


def _construct_client_safely(factory: Callable[[str], Any], api_key: str) -> Any:
    failed = False
    client: Any = None
    try:
        client = factory(api_key)
    except ProbeFailure as caught_failure:
        return_failure = caught_failure
    except Exception:
        failed = True
        return_failure = None
    else:
        return client
    if return_failure is not None:
        raise return_failure from None
    if failed:
        raise ProbeFailure("preflight_rejected_no_network_attempt", "no") from None
    raise ProbeFailure("preflight_rejected_no_network_attempt", "no") from None


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    client_factory: Callable[[str], Any] = _create_live_client,
    repo_root_override: Optional[Path] = None,
    output_dir_override: Optional[Path] = None,
    authority_dir_override: Optional[Path] = None,
    dependency_validator: Callable[[], Any] = _load_live_modules,
    _protected_hash_resolver: Optional[Callable[[Path, str], Dict[str, str]]] = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Frozen Terra one-call development probe"
    )
    parser.add_argument("--execute-development-probe", action="store_true")
    args = parser.parse_args(argv)
    repo_root = (repo_root_override or Path.cwd()).resolve()
    output_dir = (
        output_dir_override
        if output_dir_override is not None
        else repo_root / OUTPUT_DIRECTORY
    )
    authority_dir = authority_dir_override or _default_authority_directory()
    try:
        contract = load_probe_contract(repo_root / MANIFEST_PATH)
        preflight = run_preflight(
            repo_root, contract, output_dir, authority_dir=authority_dir
        )
    except Exception:
        print(
            "terra-probe status=blocked category=preflight_rejected_no_network_attempt"
        )
        return 2
    if not args.execute_development_probe:
        print(
            "terra-probe status=dry-run network_attempts=0 "
            "repository_preflight=passed credential_readiness=not_checked"
        )
        return 0

    started = _utc_now()
    monotonic_start = time.monotonic()
    configuration, _, _ = build_probe_records(contract, preflight["code_revision"])
    api_key: Optional[str] = None
    attempt_marker: Optional[Dict[str, Any]] = None
    saved_paid_environment, previous_logging_disable = _begin_paid_ambient_guard()
    try:
        dependency_validator()
        api_key = _load_ignored_api_key(repo_root)
        _assert_api_key_absent_from_git(repo_root, api_key)
        client = _construct_client_safely(client_factory, api_key)

        def marker() -> None:
            nonlocal attempt_marker
            try:
                write_attempt_marker(
                    authority_dir,
                    preflight["contract_hash"],
                    preflight["code_revision"],
                    _utc_now(),
                )
                attempt_marker = write_attempt_marker(
                    output_dir,
                    preflight["contract_hash"],
                    preflight["code_revision"],
                    _utc_now(),
                )
            except Exception:
                raise ProbeFailure("artifact_write_failure", "no") from None

        manifest, execution, response_id = execute_probe(
            client,
            contract,
            preflight["code_revision"],
            before_dispatch=marker,
        )
        raw_document = json.loads(execution.raw_response_bytes)
        raw_usage, _, _ = _validate_raw_usage(raw_document)
        payload = {
            "contract_hash": preflight["contract_hash"],
            "configuration_hash": execution.config_hash,
            "code_revision": preflight["code_revision"],
            "request_attempted": True,
            "server_acceptance": "yes",
            "protected_file_hashes": preflight["protected_file_hashes"],
            "cost_projection": preflight["cost_projection"],
            "provider_usage": _json_copy(raw_usage),
            "cost_estimate": usage_cost_estimate(contract, raw_usage),
            "response_id": response_id,
            "run_manifest": manifest.to_dict(),
            "provider_execution": execution.to_dict(),
        }
        if _contains_secret(payload, api_key):
            raise ProbeFailure(
                "artifact_write_failure",
                "yes",
                provider_request_id=execution.provider_request_id,
            ) from None
        write_artifact(
            output_dir / "success.json",
            "success",
            payload,
            repo_root=repo_root,
            _protected_hash_resolver=_protected_hash_resolver,
        )
        print("terra-probe status=pass request_attempts=1 artifact=success.json")
        return 0
    except ProviderCallbackError as wrapped:
        failure = _failure_from_callback(wrapped)
        started_timestamp = wrapped.started_timestamp
        completed_timestamp = wrapped.completed_timestamp
        latency_ms = wrapped.latency_ms
    except ProbeFailure as caught_failure:
        failure = caught_failure
        started_timestamp = started
        completed_timestamp = _utc_now()
        latency_ms = (time.monotonic() - monotonic_start) * 1000.0
    except Exception:
        failure = ProbeFailure("artifact_write_failure", "unknown")
        started_timestamp = started
        completed_timestamp = _utc_now()
        latency_ms = (time.monotonic() - monotonic_start) * 1000.0
    finally:
        _end_paid_ambient_guard(saved_paid_environment, previous_logging_disable)

    failure_payload = {
        "contract_hash": preflight["contract_hash"],
        "configuration_hash": configuration.config_hash,
        "code_revision": preflight["code_revision"],
        "request_attempted": attempt_marker is not None,
        "server_acceptance": failure.server_acceptance,
        "started_timestamp": started_timestamp,
        "completed_timestamp": completed_timestamp,
        "latency_ms": latency_ms,
        "failure_category": failure.category,
        "http_status_code": failure.http_status_code,
        "provider_request_id": failure.provider_request_id,
    }
    try:
        if api_key is not None and _contains_secret(failure_payload, api_key):
            raise ValueError("secret detected in failure artifact")
        write_artifact(output_dir / "failure.json", "failure", failure_payload)
    except Exception:
        print("terra-probe status=stop category=artifact_write_failure")
        return 3
    print(
        "terra-probe status=stop category="
        + failure.category
        + " artifact=failure.json"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
