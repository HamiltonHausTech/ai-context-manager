import ast
import base64
import hashlib
import importlib
import inspect
import json
import os
import stat
import subprocess
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from experiments.adaptive_selection.openai_manifest_probe import (
    FROZEN_PROTECTED_PATHS,
    MANIFEST_PATH,
    PINNED_CONFIGURATION_HASH,
    PINNED_MANIFEST_SHA256,
    ProbeFailure,
    _assert_api_key_absent_from_git,
    _create_live_client,
    _default_authority_directory,
    _load_ignored_api_key,
    _protected_hashes_at_revision,
    build_probe_records,
    execute_probe,
    load_probe_contract,
    main,
    make_openai_callback,
    projected_cost,
    run_preflight,
    usage_cost_estimate,
)
from experiments.adaptive_selection.openai_manifest_probe import (
    verify_artifact as _verify_artifact,
)
from experiments.adaptive_selection.openai_manifest_probe import (
    verify_attempt_marker,
)
from experiments.adaptive_selection.openai_manifest_probe import (
    write_artifact as _write_artifact,
)
from experiments.adaptive_selection.openai_manifest_probe import (
    write_attempt_marker,
)
from experiments.adaptive_selection.providers import (
    ProviderCallbackError,
    ProviderExecution,
)
from experiments.adaptive_selection.schema import RunManifest

ROOT = Path(__file__).parents[2]
CONTRACT_PATH = ROOT / MANIFEST_PATH
TEST_REVISION = "a" * 40
PINNED_CONTRACT_HASH = "sha256:" + PINNED_MANIFEST_SHA256


def _synthetic_protected_hashes(repo_root, revision):
    assert revision == TEST_REVISION
    return {path: "sha256:" + "2" * 64 for path in FROZEN_PROTECTED_PATHS}


def verify_artifact(path):
    return _verify_artifact(path, _protected_hash_resolver=_synthetic_protected_hashes)


def write_artifact(path, kind, payload):
    return _write_artifact(
        path,
        kind,
        payload,
        _protected_hash_resolver=_synthetic_protected_hashes,
    )


class FakeRaw:
    def __init__(self, content, response, request_id="req-provider-1"):
        self.content = content
        self._response = response
        self.request_id = request_id
        self.parse_calls = 0

    def parse(self):
        self.parse_calls += 1
        if isinstance(self._response, BaseException):
            raise self._response
        return self._response


class FakeCreate:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class FakeClient:
    def __init__(self, result):
        self.create = FakeCreate(result)
        self.responses = SimpleNamespace(
            with_raw_response=SimpleNamespace(create=self.create.create)
        )


class FakeHTTPError(Exception):
    def __init__(self, status, secret="sk-secret-do-not-print"):
        super().__init__(secret)
        self.status_code = status
        self.request_id = "req-http-error"


def contract():
    return load_probe_contract(CONTRACT_PATH)


def fake_preflight(frozen):
    return {
        "code_revision": TEST_REVISION,
        "contract_hash": PINNED_CONTRACT_HASH,
        "protected_file_hashes": {
            path: "sha256:" + "2" * 64 for path in FROZEN_PROTECTED_PATHS
        },
        "cost_projection": projected_cost(frozen),
    }


def successful_pair(**raw_changes):
    text = json.dumps(
        {
            "probe_marker": "terra-manifest-probe-v1",
            "status": "ok",
            "notes": "no tools were used",
        },
        separators=(",", ":"),
    )
    document = {
        "id": "resp-1",
        "model": "gpt-5.6-terra",
        "status": "completed",
        "incomplete_details": None,
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
        "usage": {
            "input_tokens": 31,
            "output_tokens": 17,
            "total_tokens": 48,
            "input_tokens_details": {"cached_tokens": 3},
            "output_tokens_details": {"reasoning_tokens": 5},
        },
    }
    for key, value in raw_changes.items():
        if key.startswith("usage__"):
            document["usage"][key.split("__", 1)[1]] = value
        else:
            document[key] = value
    usage = SimpleNamespace(
        input_tokens=document.get("usage", {}).get("input_tokens"),
        output_tokens=document.get("usage", {}).get("output_tokens"),
        total_tokens=document.get("usage", {}).get("total_tokens"),
        input_tokens_details=SimpleNamespace(cached_tokens=3),
        output_tokens_details=SimpleNamespace(reasoning_tokens=5),
    )
    response = SimpleNamespace(
        id=document.get("id"),
        model=document.get("model"),
        status=document.get("status"),
        incomplete_details=document.get("incomplete_details"),
        output_text=text,
        output=(),
        usage=usage,
    )
    raw_bytes = json.dumps(document, separators=(",", ":")).encode("utf-8")
    return raw_bytes, response


def execute_with(raw, response, request_id="req-provider-1"):
    client = FakeClient(FakeRaw(raw, response, request_id))
    clocks = iter(
        [
            "2026-07-30T11:59:59Z",
            "2026-07-30T12:00:00Z",
            "2026-07-30T12:00:01Z",
        ]
    )
    monotonic = iter([5.0, 5.25])
    result = execute_probe(
        client,
        contract(),
        TEST_REVISION,
        utc_clock=lambda: next(clocks),
        monotonic_clock=lambda: next(monotonic),
    )
    return client, result


def failure_category(client):
    with pytest.raises(ProviderCallbackError) as caught:
        clocks = iter(
            [
                "2026-07-30T11:59:59Z",
                "2026-07-30T12:00:00Z",
                "2026-07-30T12:00:01Z",
            ]
        )
        monotonic = iter([5.0, 5.25])
        execute_probe(
            client,
            contract(),
            TEST_REVISION,
            utc_clock=lambda: next(clocks),
            monotonic_clock=lambda: next(monotonic),
        )
    assert type(caught.value.__cause__) is ProbeFailure
    return caught.value, caught.value.__cause__


def test_contract_hash_literal_request_and_omitted_controls():
    frozen = contract()
    assert (
        CONTRACT_PATH.read_bytes()
        == (
            ROOT / "experiments/adaptive_selection/controls/terra_probe_v1.json"
        ).read_bytes()
    )
    assert __import__("hashlib").sha256(CONTRACT_PATH.read_bytes()).hexdigest() == (
        PINNED_MANIFEST_SHA256
    )
    configuration, request, manifest = build_probe_records(
        frozen, "code-revision", lambda: "2026-07-30T12:00:00Z"
    )
    assert configuration.to_dict()["generation_options"] == frozen["request_body"]
    assert request.prompt_text == frozen["request_body"]["input"]
    assert manifest.provider_revision is None
    assert manifest.temperature is None
    assert manifest.temperature_supported is False
    assert manifest.seed is None
    assert manifest.seed_supported is False
    for omitted in (
        "temperature",
        "top_p",
        "seed",
        "conversation",
        "previous_response_id",
    ):
        assert omitted not in frozen["request_body"]


def test_success_passes_exact_body_once_and_preserves_raw_bytes_and_ids():
    raw_bytes, response = successful_pair()
    client, (manifest, execution, response_id) = execute_with(raw_bytes, response)
    assert client.create.calls == [contract()["request_body"]]
    assert execution.raw_response_bytes == raw_bytes
    assert execution.provider_request_id == "req-provider-1"
    assert response_id == "resp-1"
    assert execution.provider_request_id != response_id
    assert execution.provider_revision is None
    assert manifest.provider_revision is None
    assert execution.input_tokens == 31
    assert execution.output_tokens == 17


@pytest.mark.parametrize(
    "result, category, acceptance, status",
    [
        (FakeHTTPError(429), "http_error", "yes", 429),
        (FakeHTTPError(500), "http_error", "yes", 500),
        (
            ConnectionError("sk-secret-do-not-print"),
            "transport_failure_server_acceptance_unknown",
            "unknown",
            None,
        ),
    ],
)
def test_retryable_failures_make_exactly_one_attempt(
    result, category, acceptance, status
):
    client = FakeClient(result)
    _, failure = failure_category(client)
    assert len(client.create.calls) == 1
    assert failure.category == category
    assert failure.server_acceptance == acceptance
    assert failure.http_status_code == status
    assert failure.__context__ is None


def test_success_makes_exactly_one_attempt():
    raw_bytes, response = successful_pair()
    client, _ = execute_with(raw_bytes, response)
    assert len(client.create.calls) == 1


@pytest.mark.parametrize("bad", ["31", True, -1])
@pytest.mark.parametrize("field", ["input_tokens", "output_tokens", "total_tokens"])
def test_raw_usage_rejects_string_boolean_and_negative(field, bad):
    raw, response = successful_pair(**{"usage__" + field: bad})
    client = FakeClient(FakeRaw(raw, response))
    _, failure = failure_category(client)
    assert failure.category == "missing_or_invalid_usage"
    assert len(client.create.calls) == 1


@pytest.mark.parametrize(
    "mutator",
    [
        lambda doc: doc["usage"].pop("input_tokens"),
        lambda doc: doc["usage"].pop("output_tokens"),
        lambda doc: doc["usage"].pop("total_tokens"),
        lambda doc: doc["usage"].update(input_tokens_details={"cached_tokens": True}),
        lambda doc: doc["usage"].update(
            output_tokens_details={"reasoning_tokens": "5"}
        ),
    ],
)
def test_raw_usage_rejects_missing_and_invalid_details(mutator):
    raw, response = successful_pair()
    document = json.loads(raw)
    mutator(document)
    raw = json.dumps(document, separators=(",", ":")).encode()
    _, failure = failure_category(FakeClient(FakeRaw(raw, response)))
    assert failure.category == "missing_or_invalid_usage"


def test_sdk_usage_coercion_or_mismatch_is_rejected():
    raw, response = successful_pair()
    response.usage.input_tokens = "31"
    _, failure = failure_category(FakeClient(FakeRaw(raw, response)))
    assert failure.category == "missing_or_invalid_usage"


@pytest.mark.parametrize(
    "change, response_change, expected",
    [
        ({"status": "failed"}, {"status": "failed"}, "invalid_response_status"),
        (
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
            },
            {
                "status": "incomplete",
                "incomplete_details": SimpleNamespace(reason="max_output_tokens"),
            },
            "incomplete_response",
        ),
        (
            {"incomplete_details": {"reason": "max_output_tokens"}},
            {"incomplete_details": SimpleNamespace(reason="max_output_tokens")},
            "incomplete_response",
        ),
        (
            {"model": "other-model"},
            {"model": "other-model"},
            "provider_identity_mismatch",
        ),
        ({"id": ""}, {"id": ""}, "missing_response_id"),
    ],
)
def test_status_incomplete_identity_and_response_id_fail(
    change, response_change, expected
):
    raw, response = successful_pair(**change)
    for name, value in response_change.items():
        setattr(response, name, value)
    _, failure = failure_category(FakeClient(FakeRaw(raw, response)))
    assert failure.category == expected


def test_missing_provider_request_id_fails():
    raw, response = successful_pair()
    _, failure = failure_category(FakeClient(FakeRaw(raw, response, "")))
    assert failure.category == "missing_provider_request_id"


def test_refusal_fails_deterministically():
    raw, response = successful_pair(
        output=[{"type": "message", "content": [{"type": "refusal", "refusal": "no"}]}]
    )
    _, failure = failure_category(FakeClient(FakeRaw(raw, response)))
    assert failure.category == "refusal"


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        "{}",
        '{"probe_marker":"wrong","status":"ok","notes":"x"}',
        '{"probe_marker":"terra-manifest-probe-v1","status":"ok","notes":1}',
        '{"probe_marker":"terra-manifest-probe-v1","status":"ok","notes":"x","extra":1}',
    ],
)
def test_malformed_structured_output_fails(text):
    raw, response = successful_pair()
    response.output_text = text
    _, failure = failure_category(FakeClient(FakeRaw(raw, response)))
    assert failure.category == "invalid_structured_output"


def test_sdk_output_text_must_equal_exact_raw_output_text():
    raw, response = successful_pair()
    response.output_text = json.dumps(
        {
            "probe_marker": "terra-manifest-probe-v1",
            "status": "ok",
            "notes": "different but schema-valid SDK projection",
        },
        separators=(",", ":"),
    )
    _, failure = failure_category(FakeClient(FakeRaw(raw, response)))
    assert failure.category == "invalid_structured_output"


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "raw-malformed"])
def test_raw_output_text_shape_is_independently_strict(mutation):
    raw, response = successful_pair()
    document = json.loads(raw)
    if mutation == "missing":
        document["output"][0]["content"] = []
    elif mutation == "duplicate":
        document["output"][0]["content"].append(
            dict(document["output"][0]["content"][0])
        )
    else:
        document["output"][0]["content"][0]["text"] = "not json"
    raw = json.dumps(document, separators=(",", ":")).encode()
    _, failure = failure_category(FakeClient(FakeRaw(raw, response)))
    assert failure.category == "invalid_structured_output"


def test_refusal_plus_output_text_is_rejected_as_refusal():
    raw, response = successful_pair()
    document = json.loads(raw)
    document["output"][0]["content"].append({"type": "refusal", "refusal": "no"})
    raw = json.dumps(document, separators=(",", ":")).encode()
    _, failure = failure_category(FakeClient(FakeRaw(raw, response)))
    assert failure.category == "refusal"


def test_equal_or_secret_shaped_ids_are_rejected():
    raw, response = successful_pair()
    _, equal = failure_category(FakeClient(FakeRaw(raw, response, "resp-1")))
    assert equal.category == "missing_provider_request_id"
    response.id = "sk-secret-value"
    document = json.loads(raw)
    document["id"] = response.id
    raw = json.dumps(document, separators=(",", ":")).encode()
    _, secret = failure_category(FakeClient(FakeRaw(raw, response)))
    assert secret.category == "missing_response_id"


def test_usage_cost_estimate_qualifies_cached_input_pricing():
    raw, _ = successful_pair()
    usage = json.loads(raw)["usage"]
    estimate = usage_cost_estimate(contract(), usage)
    assert estimate["estimated_cost"] == "0.000266"
    assert estimate["cached_input_tokens"] == 3
    assert estimate["estimate_kind"] == "upper_bound_cached_discount_not_frozen"
    usage["input_tokens_details"]["cached_tokens"] = 0
    assert usage_cost_estimate(contract(), usage)["estimate_kind"] == (
        "frozen_standard_rates_exact_when_no_other_fees"
    )


def test_malformed_raw_and_parse_failure_are_sanitized():
    _, response = successful_pair()
    _, malformed = failure_category(FakeClient(FakeRaw(b"{secret", response)))
    assert malformed.category == "parse_error"
    raw, _ = successful_pair()
    _, parse = failure_category(
        FakeClient(FakeRaw(raw, RuntimeError("sk-secret-do-not-print")))
    )
    assert parse.category == "parse_error"


def test_raw_size_limit_is_enforced_before_sdk_parse(monkeypatch):
    _, response = successful_pair()
    raw = FakeRaw(b"x" * 11, response)
    monkeypatch.setattr(
        "experiments.adaptive_selection.openai_manifest_probe.MAX_RAW_RESPONSE_BYTES",
        10,
    )
    _, failure = failure_category(FakeClient(raw))
    assert failure.category == "parse_error"
    assert raw.parse_calls == 0


def test_secret_bearing_transport_exception_is_absent_from_traceback_and_exception_text():
    secret = "sk-secret-do-not-print"
    client = FakeClient(ConnectionError(secret))
    wrapped, failure = failure_category(client)
    rendered = "".join(
        traceback.format_exception(type(wrapped), wrapped, wrapped.__traceback__)
    )
    assert secret not in rendered
    assert secret not in str(wrapped)
    assert secret not in str(failure)
    assert failure.__context__ is None
    assert repr(client) not in rendered


def test_before_dispatch_runs_immediately_before_single_create():
    raw, response = successful_pair()
    events = []

    class OrderedCreate(FakeCreate):
        def create(self, **kwargs):
            events.append("create")
            return super().create(**kwargs)

    client = FakeClient(FakeRaw(raw, response))
    ordered = OrderedCreate(client.create.result)
    client.create = ordered
    client.responses.with_raw_response.create = ordered.create
    config, request, _ = build_probe_records(
        contract(), "code", lambda: "2026-07-30T12:00:00Z"
    )
    callback = make_openai_callback(
        client, contract()["request_body"], lambda: events.append("marker")
    )
    callback(config, request)
    assert events == ["marker", "create"]
    assert len(ordered.calls) == 1


def test_attempt_marker_is_atomic_private_hashed_and_no_overwrite(tmp_path):
    output = tmp_path / "probe"
    marker = write_attempt_marker(
        output, PINNED_CONTRACT_HASH, TEST_REVISION, "2026-07-30T12:00:00Z"
    )
    path = output / "attempt-consumed.json"
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert json.loads(path.read_bytes())["attempt_hash"] == marker["attempt_hash"]
    assert verify_attempt_marker(path) == marker
    with pytest.raises(FileExistsError):
        write_attempt_marker(
            output, PINNED_CONTRACT_HASH, TEST_REVISION, "2026-07-30T12:00:01Z"
        )


def test_concurrent_callbacks_allow_exactly_one_marker_winner_and_dispatch(tmp_path):
    output = tmp_path / "probe"
    raw, response = successful_pair()
    clients = [FakeClient(FakeRaw(raw, response)), FakeClient(FakeRaw(raw, response))]

    def attempt(client):
        def mark():
            write_attempt_marker(
                output,
                PINNED_CONTRACT_HASH,
                TEST_REVISION,
                "2026-07-30T12:00:00Z",
            )

        try:
            execute_probe(
                client,
                contract(),
                TEST_REVISION,
                before_dispatch=mark,
            )
            return True
        except ProviderCallbackError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(attempt, clients))
    assert sorted(outcomes) == [False, True]
    assert sum(len(client.create.calls) for client in clients) == 1
    assert verify_attempt_marker(output / "attempt-consumed.json")


def success_payload():
    raw, response = successful_pair()
    _, (manifest, execution, response_id) = execute_with(raw, response)
    provider_usage = json.loads(raw)["usage"]
    return {
        "contract_hash": PINNED_CONTRACT_HASH,
        "configuration_hash": execution.config_hash,
        "code_revision": TEST_REVISION,
        "request_attempted": True,
        "server_acceptance": "yes",
        "protected_file_hashes": {
            path: "sha256:" + "2" * 64 for path in FROZEN_PROTECTED_PATHS
        },
        "cost_projection": projected_cost(contract()),
        "provider_usage": provider_usage,
        "cost_estimate": usage_cost_estimate(contract(), provider_usage),
        "response_id": response_id,
        "run_manifest": manifest.to_dict(),
        "provider_execution": execution.to_dict(),
    }


def _rewrite_with_valid_whole_hash(path, value):
    artifact_bytes = json.dumps(
        value["artifact"], sort_keys=True, separators=(",", ":")
    ).encode()
    value["artifact_hash"] = "sha256:" + hashlib.sha256(artifact_bytes).hexdigest()
    path.write_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )


def test_success_artifact_permissions_determinism_no_overwrite_and_hashes(tmp_path):
    first = tmp_path / "first" / "success.json"
    second = tmp_path / "second" / "success.json"
    payload = success_payload()
    one = write_artifact(first, "success", payload)
    two = write_artifact(second, "success", payload)
    assert first.read_bytes() == second.read_bytes()
    assert one == two == verify_artifact(first)
    assert stat.S_IMODE(first.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(first.stat().st_mode) == 0o600
    execution = ProviderExecution.from_dict(one["artifact"]["provider_execution"])
    assert execution.raw_response_bytes
    assert (
        RunManifest.from_dict(one["artifact"]["run_manifest"]).provider_revision is None
    )
    with pytest.raises(FileExistsError):
        write_artifact(first, "success", payload)


def test_artifact_tampering_breaks_whole_and_embedded_hash_verification(tmp_path):
    path = tmp_path / "probe" / "success.json"
    write_artifact(path, "success", success_payload())
    value = json.loads(path.read_bytes())
    value["artifact"]["response_id"] = "tampered"
    path.write_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )
    with pytest.raises(ValueError, match="whole-artifact hash"):
        verify_artifact(path)


def test_validly_rehashed_artifact_cannot_add_fields_or_fabricate_cost(tmp_path):
    extra = tmp_path / "extra" / "success.json"
    write_artifact(extra, "success", success_payload())
    value = json.loads(extra.read_bytes())
    value["artifact"]["unexpected"] = "fabricated"
    _rewrite_with_valid_whole_hash(extra, value)
    with pytest.raises(ValueError, match="fields are not exact"):
        verify_artifact(extra)

    cost = tmp_path / "cost" / "success.json"
    write_artifact(cost, "success", success_payload())
    value = json.loads(cost.read_bytes())
    value["artifact"]["cost_estimate"]["estimated_cost"] = "0.000001"
    _rewrite_with_valid_whole_hash(cost, value)
    with pytest.raises(ValueError, match="cost estimate does not match"):
        verify_artifact(cost)

    rates = tmp_path / "rates" / "success.json"
    write_artifact(rates, "success", success_payload())
    value = json.loads(rates.read_bytes())
    value["artifact"]["cost_estimate"]["input_per_million"] = "3.00"
    value["artifact"]["cost_estimate"]["estimated_cost"] = "0.000297"
    _rewrite_with_valid_whole_hash(rates, value)
    with pytest.raises(ValueError, match="cost estimate is malformed"):
        verify_artifact(rates)


@pytest.mark.parametrize(
    "mutation",
    ["failed-status", "wrong-model", "refusal", "schema-invalid", "extra-output"],
)
def test_validly_rehashed_success_replays_raw_acceptance_checks(tmp_path, mutation):
    path = tmp_path / mutation / "success.json"
    write_artifact(path, "success", success_payload())
    envelope = json.loads(path.read_bytes())
    execution = envelope["artifact"]["provider_execution"]
    raw = json.loads(base64.b64decode(execution["raw_response_bytes"]["data"]))
    if mutation == "failed-status":
        raw["status"] = "failed"
    elif mutation == "wrong-model":
        raw["model"] = "other-model"
    elif mutation == "refusal":
        raw["output"][0]["content"].append({"type": "refusal", "refusal": "no"})
    elif mutation == "schema-invalid":
        raw["output"][0]["content"][0]["text"] = "not json"
        execution["response_text"] = "not json"
    else:
        raw["output"].append({"type": "tool_call", "name": "forbidden"})
    raw_bytes = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
    execution["raw_response_bytes"]["data"] = base64.b64encode(raw_bytes).decode()
    execution["raw_response_hash"] = "sha256:" + hashlib.sha256(raw_bytes).hexdigest()
    _rewrite_with_valid_whole_hash(path, envelope)
    with pytest.raises(ValueError):
        verify_artifact(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("request_attempted", "true"),
        ("server_acceptance", "bogus"),
        ("provider_request_id", "sk-secret-value"),
        ("contract_hash", "sha256:" + "0" * 64),
        ("code_revision", "not-a-git-revision"),
    ],
)
def test_validly_rehashed_failure_rejects_malformed_typed_evidence(
    tmp_path, field, value
):
    payload = {
        "contract_hash": PINNED_CONTRACT_HASH,
        "configuration_hash": PINNED_CONFIGURATION_HASH,
        "code_revision": TEST_REVISION,
        "request_attempted": True,
        "server_acceptance": "yes",
        "started_timestamp": "2026-07-30T12:00:00Z",
        "completed_timestamp": "2026-07-30T12:00:01Z",
        "latency_ms": 1000.0,
        "failure_category": "http_error",
        "http_status_code": 429,
        "provider_request_id": "req-http-error",
    }
    path = tmp_path / field / "failure.json"
    write_artifact(path, "failure", payload)
    envelope = json.loads(path.read_bytes())
    envelope["artifact"][field] = value
    _rewrite_with_valid_whole_hash(path, envelope)
    with pytest.raises(ValueError):
        verify_artifact(path)


def test_validly_rehashed_failure_rejects_reversed_timestamps(tmp_path):
    payload = {
        "contract_hash": PINNED_CONTRACT_HASH,
        "configuration_hash": PINNED_CONFIGURATION_HASH,
        "code_revision": TEST_REVISION,
        "request_attempted": True,
        "server_acceptance": "yes",
        "started_timestamp": "2026-07-30T12:00:00Z",
        "completed_timestamp": "2026-07-30T12:00:01Z",
        "latency_ms": 1000.0,
        "failure_category": "http_error",
        "http_status_code": 429,
        "provider_request_id": "req-http-error",
    }
    path = tmp_path / "reversed" / "failure.json"
    write_artifact(path, "failure", payload)
    envelope = json.loads(path.read_bytes())
    envelope["artifact"]["completed_timestamp"] = "2026-07-30T11:59:59Z"
    _rewrite_with_valid_whole_hash(path, envelope)
    with pytest.raises(ValueError, match="timestamps are reversed"):
        verify_artifact(path)


def test_failure_artifact_exact_sanitized_fields_and_server_acceptance(tmp_path):
    payload = {
        "contract_hash": PINNED_CONTRACT_HASH,
        "configuration_hash": PINNED_CONFIGURATION_HASH,
        "code_revision": TEST_REVISION,
        "request_attempted": True,
        "server_acceptance": "yes",
        "started_timestamp": "2026-07-30T12:00:00Z",
        "completed_timestamp": "2026-07-30T12:00:01Z",
        "latency_ms": 1000.0,
        "failure_category": "http_error",
        "http_status_code": 429,
        "provider_request_id": "req-http-error",
    }
    path = tmp_path / "probe" / "failure.json"
    envelope = write_artifact(path, "failure", payload)
    assert verify_artifact(path) == envelope
    serialized = path.read_text()
    for forbidden in (
        "api_key",
        "authorization_header",
        "complete_headers",
        "exception_message",
        "traceback",
        "request_body",
        "response_body",
        "sk-secret",
    ):
        assert forbidden not in serialized


def test_projected_cost_guard_rejects_oversized_and_over_budget_contracts():
    frozen = contract()
    projection = projected_cost(frozen)
    assert projection == {
        "projected_input_tokens": 1984,
        "projected_output_tokens": 512,
        "projected_cost": "0.010112",
        "maximum_projected_cost": "0.25",
    }
    oversized = json.loads(json.dumps(frozen))
    oversized["request_body"]["input"] = "x" * 5000
    with pytest.raises(ProbeFailure):
        projected_cost(oversized)
    expensive = json.loads(json.dumps(frozen))
    expensive["budget"]["maximum_projected_cost"] = "0.000001"
    with pytest.raises(ProbeFailure):
        projected_cost(expensive)


@pytest.mark.parametrize(
    "field,value",
    [
        ("projected_worst_case_output_tokens", "512"),
        ("maximum_projected_input_tokens", True),
        ("input_per_million", True),
        ("output_per_million", "-1"),
        ("maximum_projected_cost", "NaN"),
    ],
)
def test_projected_cost_rejects_malformed_budget_domains(field, value):
    frozen = json.loads(json.dumps(contract()))
    frozen["budget"][field] = value
    with pytest.raises(ProbeFailure):
        projected_cost(frozen)


def git(repo, *args):
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def prepared_repo(tmp_path):
    frozen = contract()
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    git(repo, "init")
    git(repo, "config", "user.email", "test@example.invalid")
    git(repo, "config", "user.name", "Test")
    implementation = "experiments/adaptive_selection/openai_manifest_probe.py"
    implementation_test = "tests/adaptive_selection/test_openai_manifest_probe.py"
    for relative in frozen["preflight"]["protected_paths"]:
        if relative in (implementation, implementation_test):
            continue
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == str(MANIFEST_PATH):
            path.write_bytes(CONTRACT_PATH.read_bytes())
        elif relative == "docs/research/terra-bridge-development-protocol.md":
            path.write_bytes(
                (
                    ROOT / "docs/research/terra-bridge-development-protocol.md"
                ).read_bytes()
            )
        else:
            path.write_text("contract\n")
    (repo / ".gitignore").write_text(".env\n.local/\n")
    git(repo, "add", "-f", ".")
    git(repo, "commit", "-m", "contract")
    for relative in (implementation, implementation_test):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("implementation\n")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "implementation")
    return repo, frozen


def test_success_verifier_rederives_protected_hashes_from_code_revision(tmp_path):
    repo, _ = prepared_repo(tmp_path)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    hashes = _protected_hashes_at_revision(repo, revision)
    assert hashes == {
        relative: "sha256:" + hashlib.sha256((repo / relative).read_bytes()).hexdigest()
        for relative in FROZEN_PROTECTED_PATHS
    }
    payload = success_payload()
    payload["code_revision"] = revision
    payload["run_manifest"]["code_revision"] = revision
    payload["protected_file_hashes"] = hashes
    path = repo / ".local/artifact-test/success.json"
    _write_artifact(path, "success", payload, repo_root=repo)
    assert (
        _verify_artifact(path, repo_root=repo)["artifact"]["protected_file_hashes"]
        == hashes
    )
    envelope = json.loads(path.read_bytes())
    first = FROZEN_PROTECTED_PATHS[0]
    envelope["artifact"]["protected_file_hashes"][first] = "sha256:" + "0" * 64
    _rewrite_with_valid_whole_hash(path, envelope)
    with pytest.raises(ValueError, match="do not match the code revision"):
        _verify_artifact(path, repo_root=repo)


def test_preflight_accepts_tracked_clean_protected_paths_hash_ancestry_and_empty_output(
    tmp_path,
):
    repo, frozen = prepared_repo(tmp_path)
    result = run_preflight(repo, frozen, repo / ".local/adaptive-selection-probes")
    assert result["contract_hash"].endswith(PINNED_MANIFEST_SHA256)
    assert result["contract_commit"] != result["implementation_commit"]
    assert set(result["protected_file_hashes"]) == set(
        frozen["preflight"]["protected_paths"]
    )


def test_preflight_rejects_dirty_protected_path_and_nonempty_output(tmp_path):
    repo, frozen = prepared_repo(tmp_path)
    protected = repo / frozen["preflight"]["protected_paths"][0]
    protected.write_text(protected.read_text() + "dirty")
    with pytest.raises(ProbeFailure):
        run_preflight(repo, frozen, repo / ".local/adaptive-selection-probes")
    git(repo, "checkout", "--", str(protected.relative_to(repo)))
    output = repo / ".local/adaptive-selection-probes"
    output.mkdir(parents=True)
    (output / "attempt-consumed.json").write_text("consumed")
    with pytest.raises(ProbeFailure):
        run_preflight(repo, frozen, output)


def test_preflight_rejects_unignored_or_outside_repository_output(tmp_path):
    repo, frozen = prepared_repo(tmp_path)
    with pytest.raises(ProbeFailure):
        run_preflight(repo, frozen, repo / "unignored-output")
    with pytest.raises(ProbeFailure):
        run_preflight(repo, frozen, tmp_path / "outside-output")


def test_consumed_canonical_marker_cannot_be_bypassed_by_alternate_ignored_path(
    tmp_path,
):
    repo, frozen = prepared_repo(tmp_path)
    canonical = repo / ".local/adaptive-selection-probes"
    write_attempt_marker(
        canonical, PINNED_CONTRACT_HASH, TEST_REVISION, "2026-07-30T12:00:00Z"
    )
    with pytest.raises(ProbeFailure):
        run_preflight(repo, frozen, canonical)
    alternate = repo / ".local/alternate-empty"
    with pytest.raises(ProbeFailure):
        run_preflight(repo, frozen, alternate)


def test_preflight_rejects_symlinked_canonical_output(tmp_path):
    repo, frozen = prepared_repo(tmp_path)
    external = tmp_path / "external"
    external.mkdir()
    local = repo / ".local"
    local.mkdir()
    (local / "adaptive-selection-probes").symlink_to(external, target_is_directory=True)
    with pytest.raises(ProbeFailure):
        run_preflight(repo, frozen, local / "adaptive-selection-probes")


def test_machine_global_authority_marker_blocks_another_checkout(tmp_path):
    first_repo, frozen = prepared_repo(tmp_path / "first")
    second_repo, _ = prepared_repo(tmp_path / "second")
    authority = tmp_path / "shared-authority"
    write_attempt_marker(
        authority, PINNED_CONTRACT_HASH, TEST_REVISION, "2026-07-30T12:00:00Z"
    )
    for repo in (first_repo, second_repo):
        with pytest.raises(ProbeFailure):
            run_preflight(
                repo,
                frozen,
                repo / ".local/adaptive-selection-probes",
                authority_dir=authority,
            )


def test_machine_global_authority_path_does_not_trust_ambient_home(
    monkeypatch, tmp_path
):
    expected = _default_authority_directory()
    monkeypatch.setenv("HOME", str(tmp_path / "attacker-selected-home"))
    assert _default_authority_directory() == expected


def test_manifest_hash_gate_rejects_changed_bytes(tmp_path):
    changed = tmp_path / "manifest.json"
    changed.write_bytes(CONTRACT_PATH.read_bytes() + b" ")
    with pytest.raises(ProbeFailure) as caught:
        load_probe_contract(changed)
    assert caught.value.category == "preflight_rejected_no_network_attempt"


def test_api_key_loader_accepts_only_ignored_owner_private_regular_dotenv(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    git(repo, "init")
    (repo / ".gitignore").write_text(".env\n")
    git(repo, "add", ".gitignore")
    git(repo, "config", "user.email", "test@example.invalid")
    git(repo, "config", "user.name", "Test")
    git(repo, "commit", "-m", "ignore dotenv")
    dotenv = repo / ".env"
    dotenv.write_text("OPENAI_API_KEY=test-only-placeholder\n")
    dotenv.chmod(0o600)
    assert _load_ignored_api_key(repo) == "test-only-placeholder"

    second_link = repo / "second-link"
    os.link(dotenv, second_link)
    with pytest.raises(ProbeFailure):
        _load_ignored_api_key(repo)
    second_link.unlink()

    for invalid in (
        "OPENAI_API_KEY='unmatched-quote\n",
        "OPENAI_API_KEY=test value with spaces\n",
        "OPENAI_API_KEY=short\n",
        "OPENAI_API_KEY=" + "x" * 17000 + "\n",
    ):
        dotenv.write_text(invalid)
        dotenv.chmod(0o600)
        with pytest.raises(ProbeFailure):
            _load_ignored_api_key(repo)
    dotenv.write_bytes(b"OPENAI_API_KEY=test-only-placeholder\xff\n")
    dotenv.chmod(0o600)
    with pytest.raises(ProbeFailure) as decoding_failure:
        _load_ignored_api_key(repo)
    assert decoding_failure.value.__context__ is None
    dotenv.write_text("OPENAI_API_KEY=test-only-placeholder\n")

    dotenv.chmod(0o644)
    with pytest.raises(ProbeFailure):
        _load_ignored_api_key(repo)
    dotenv.unlink()
    external = tmp_path / "external.env"
    external.write_text("OPENAI_API_KEY=test-only-placeholder\n")
    external.chmod(0o600)
    dotenv.symlink_to(external)
    with pytest.raises(ProbeFailure):
        _load_ignored_api_key(repo)


def test_api_key_loader_rejects_tracked_dotenv(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    git(repo, "init")
    git(repo, "config", "user.email", "test@example.invalid")
    git(repo, "config", "user.name", "Test")
    (repo / ".gitignore").write_text(".env\n")
    dotenv = repo / ".env"
    dotenv.write_text("OPENAI_API_KEY=test-only-placeholder\n")
    dotenv.chmod(0o600)
    git(repo, "add", "-f", ".gitignore", ".env")
    git(repo, "commit", "-m", "tracked dotenv")
    with pytest.raises(ProbeFailure):
        _load_ignored_api_key(repo)


def test_api_key_scan_rejects_exact_value_in_committed_git_snapshot(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    git(repo, "init")
    git(repo, "config", "user.email", "test@example.invalid")
    git(repo, "config", "user.name", "Test")
    (repo / "safe.txt").write_text("safe\n")
    git(repo, "add", "safe.txt")
    git(repo, "commit", "-m", "safe")
    key = "test-only-unique-api-key-value"
    _assert_api_key_absent_from_git(repo, key)
    (repo / "leak.txt").write_text(key)
    git(repo, "add", "leak.txt")
    git(repo, "commit", "-m", "leak")
    with pytest.raises(ProbeFailure) as caught:
        _assert_api_key_absent_from_git(repo, key)
    assert caught.value.category == "preflight_rejected_no_network_attempt"


@pytest.mark.parametrize("option", ["--output-dir", "--repo-root"])
def test_cli_rejects_public_path_overrides(tmp_path, option):
    with pytest.raises(SystemExit) as caught:
        main([option, str(tmp_path / "alternate")])
    assert caught.value.code == 2


def test_paid_main_contains_no_development_gate_runner():
    source = inspect.getsource(main)
    assert "pytest" not in source
    assert "compileall" not in source
    assert '"-m", "build"' not in source
    assert "_run_required_gates" not in source


def test_default_main_is_dry_run_and_never_builds_client(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        "experiments.adaptive_selection.openai_manifest_probe.load_probe_contract",
        lambda path: contract(),
    )
    monkeypatch.setattr(
        "experiments.adaptive_selection.openai_manifest_probe.run_preflight",
        lambda *args, **kwargs: {"code_revision": "revision"},
    )
    status = main([], client_factory=lambda key: calls.append(key))
    captured = capsys.readouterr()
    assert status == 0
    assert calls == []
    assert captured.err == ""
    assert (
        captured.out == "terra-probe status=dry-run network_attempts=0 "
        "repository_preflight=passed credential_readiness=not_checked\n"
    )


def test_unexpected_preflight_exception_is_rendered_without_text_or_traceback(
    monkeypatch, capsys
):
    secret = "secret-bearing-preflight-error"
    monkeypatch.setattr(
        "experiments.adaptive_selection.openai_manifest_probe.load_probe_contract",
        lambda path: (_ for _ in ()).throw(RuntimeError(secret)),
    )
    assert main([]) == 2
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err
    assert captured.err == ""
    assert captured.out == (
        "terra-probe status=blocked " "category=preflight_rejected_no_network_attempt\n"
    )


def test_paid_command_pre_dispatch_failure_writes_no_attempt_artifact(
    monkeypatch, tmp_path, capsys
):
    frozen = contract()
    output = tmp_path / "artifacts"
    preflight = fake_preflight(frozen)

    monkeypatch.setattr(
        "experiments.adaptive_selection.openai_manifest_probe.load_probe_contract",
        lambda path: frozen,
    )
    monkeypatch.setattr(
        "experiments.adaptive_selection.openai_manifest_probe.run_preflight",
        lambda *args, **kwargs: preflight,
    )
    monkeypatch.setattr(
        "experiments.adaptive_selection.openai_manifest_probe._load_ignored_api_key",
        lambda path: (_ for _ in ()).throw(
            ProbeFailure("preflight_rejected_no_network_attempt", "no")
        ),
    )
    assert (
        main(
            ["--execute-development-probe"],
            repo_root_override=tmp_path,
            output_dir_override=output,
            authority_dir_override=tmp_path / "authority",
            dependency_validator=lambda: None,
            _protected_hash_resolver=_synthetic_protected_hashes,
        )
        == 1
    )
    captured = capsys.readouterr()
    assert captured.err == ""
    artifact = verify_artifact(output / "failure.json")["artifact"]
    assert artifact["failure_category"] == "preflight_rejected_no_network_attempt"
    assert artifact["request_attempted"] is False
    assert artifact["server_acceptance"] == "no"
    assert not (output / "attempt-consumed.json").exists()


def test_live_command_fake_failure_writes_sanitized_artifact_once(
    monkeypatch, tmp_path, capsys
):
    frozen = contract()
    output = tmp_path / "artifacts"
    client = FakeClient(FakeHTTPError(500, "sk-secret-do-not-print"))
    preflight = fake_preflight(frozen)
    monkeypatch.setattr(
        "experiments.adaptive_selection.openai_manifest_probe.load_probe_contract",
        lambda path: frozen,
    )
    monkeypatch.setattr(
        "experiments.adaptive_selection.openai_manifest_probe.run_preflight",
        lambda *args, **kwargs: preflight,
    )
    monkeypatch.setattr(
        "experiments.adaptive_selection.openai_manifest_probe._load_ignored_api_key",
        lambda path: "test-only-api-key-value",
    )
    monkeypatch.setattr(
        "experiments.adaptive_selection.openai_manifest_probe._assert_api_key_absent_from_git",
        lambda *args: None,
    )

    status = main(
        ["--execute-development-probe"],
        client_factory=lambda key: client,
        repo_root_override=tmp_path,
        output_dir_override=output,
        authority_dir_override=tmp_path / "authority",
        dependency_validator=lambda: None,
        _protected_hash_resolver=_synthetic_protected_hashes,
    )

    captured = capsys.readouterr()
    assert status == 1
    assert len(client.create.calls) == 1
    assert "sk-secret-do-not-print" not in captured.out + captured.err
    assert "500" not in captured.out + captured.err
    assert captured.err == ""
    artifact = verify_artifact(output / "failure.json")["artifact"]
    assert artifact["request_attempted"] is True
    assert artifact["server_acceptance"] == "yes"
    assert artifact["failure_category"] == "http_error"
    assert artifact["http_status_code"] == 500
    assert artifact["provider_request_id"] == "req-http-error"
    assert (output / "attempt-consumed.json").exists()


def test_live_command_blocks_secret_echo_before_success_artifact_write(
    monkeypatch, tmp_path, capsys
):
    frozen = contract()
    output = tmp_path / "artifacts"
    secret = "test-only-api-key-value"
    raw, response = successful_pair()
    text = json.dumps(
        {
            "probe_marker": "terra-manifest-probe-v1",
            "status": "ok",
            "notes": secret,
        },
        separators=(",", ":"),
    )
    document = json.loads(raw)
    document["output"][0]["content"][0]["text"] = text
    raw = json.dumps(document, separators=(",", ":")).encode()
    response.output_text = text
    client = FakeClient(FakeRaw(raw, response))
    preflight = fake_preflight(frozen)
    monkeypatch.setattr(
        "experiments.adaptive_selection.openai_manifest_probe.load_probe_contract",
        lambda path: frozen,
    )
    monkeypatch.setattr(
        "experiments.adaptive_selection.openai_manifest_probe.run_preflight",
        lambda *args, **kwargs: preflight,
    )
    monkeypatch.setattr(
        "experiments.adaptive_selection.openai_manifest_probe._load_ignored_api_key",
        lambda path: secret,
    )
    monkeypatch.setattr(
        "experiments.adaptive_selection.openai_manifest_probe._assert_api_key_absent_from_git",
        lambda *args: None,
    )
    status = main(
        ["--execute-development-probe"],
        client_factory=lambda key: client,
        repo_root_override=tmp_path,
        output_dir_override=output,
        authority_dir_override=tmp_path / "authority",
        dependency_validator=lambda: None,
        _protected_hash_resolver=_synthetic_protected_hashes,
    )
    captured = capsys.readouterr()
    assert status == 1
    assert len(client.create.calls) == 1
    assert secret not in captured.out + captured.err
    assert not (output / "success.json").exists()
    failure = verify_artifact(output / "failure.json")["artifact"]
    assert failure["failure_category"] == "artifact_write_failure"
    assert secret not in (output / "failure.json").read_text()


def test_openai_sdk_success_mock_preserves_raw_entity_and_sdk_projection():
    httpx = pytest.importorskip("httpx")
    openai = pytest.importorskip("openai")
    raw_bytes, _ = successful_pair()
    attempts = []

    def handler(request):
        attempts.append(request)
        return httpx.Response(
            200,
            content=raw_bytes,
            headers={
                "content-type": "application/json",
                "x-request-id": "req-sdk-success",
            },
            request=request,
        )

    http_client = httpx.Client(
        transport=httpx.MockTransport(handler), trust_env=False, follow_redirects=False
    )
    client = openai.OpenAI(
        api_key="test-only-placeholder",
        base_url="https://mock.invalid/v1",
        max_retries=0,
        timeout=30.0,
        http_client=http_client,
    )
    try:
        manifest, execution, response_id = execute_probe(
            client, contract(), TEST_REVISION
        )
    finally:
        client.close()
    assert len(attempts) == 1
    assert json.loads(attempts[0].content) == contract()["request_body"]
    assert execution.raw_response_bytes == raw_bytes
    assert execution.provider_request_id == "req-sdk-success"
    assert response_id == "resp-1"
    assert manifest.provider_revision is None


@pytest.mark.parametrize("outcome", [307, 429, 500, "connection"])
def test_openai_sdk_mock_transport_makes_exactly_one_http_attempt(outcome):
    httpx = pytest.importorskip("httpx")
    openai = pytest.importorskip("openai")
    attempts = []

    def handler(request):
        attempts.append(request)
        if outcome == "connection":
            raise httpx.ConnectError(
                "secret-bearing mock connection failure", request=request
            )
        return httpx.Response(
            outcome,
            json={
                "error": {
                    "message": "secret-bearing mock HTTP failure",
                    "type": "mock_error",
                }
            },
            headers={"x-request-id": "req-sdk-mock"},
            request=request,
        )

    http_client = httpx.Client(
        transport=httpx.MockTransport(handler), trust_env=False, follow_redirects=False
    )
    client = openai.OpenAI(
        api_key="test-only-placeholder",
        base_url="https://mock.invalid/v1",
        max_retries=0,
        timeout=30.0,
        http_client=http_client,
    )
    try:
        _, failure = failure_category(client)
    finally:
        client.close()
    assert len(attempts) == 1
    if outcome == "connection":
        assert failure.category == "transport_failure_server_acceptance_unknown"
        assert failure.server_acceptance == "unknown"
    else:
        assert failure.category == "http_error"
        assert failure.server_acceptance == "yes"
        assert failure.http_status_code == outcome
        assert failure.provider_request_id == "req-sdk-mock"


def test_live_client_factory_is_lazy_and_disables_retries(monkeypatch):
    observed = {}

    class OpenAI:
        def __init__(self, **kwargs):
            observed.update(kwargs)

    fake_module = ModuleType("openai")
    setattr(fake_module, "OpenAI", OpenAI)
    setattr(fake_module, "__version__", "2.46.0")
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    _create_live_client("injected-test-key")
    assert observed["api_key"] == "injected-test-key"
    assert observed["base_url"] == "https://api.openai.com/v1"
    assert observed["websocket_base_url"] == "wss://api.openai.com/v1"
    assert observed["max_retries"] == 0
    assert observed["timeout"] == 30.0
    assert observed["default_headers"] == {}
    assert observed["default_query"] == {}
    assert observed["http_client"].follow_redirects is False
    assert observed["http_client"]._trust_env is False
    observed["http_client"].close()


def test_live_client_factory_rejects_unfrozen_sdk_version(monkeypatch):
    fake_module = ModuleType("openai")
    setattr(fake_module, "OpenAI", lambda **kwargs: object())
    setattr(fake_module, "__version__", "different")
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    with pytest.raises(ProbeFailure) as caught:
        _create_live_client("injected-test-key")
    assert caught.value.category == "preflight_rejected_no_network_attempt"


def test_live_client_ignores_ambient_routing_headers_and_proxy_settings(
    monkeypatch, tmp_path
):
    keylog = tmp_path / "tls-secrets.log"
    monkeypatch.setenv("OPENAI_BASE_URL", "https://evil.invalid/v1")
    monkeypatch.setenv("OPENAI_ORG_ID", "evil-org")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "evil-project")
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", "X-Evil: exfiltrate")
    monkeypatch.setenv("HTTPS_PROXY", "http://evil.invalid:8080")
    monkeypatch.setenv("SSLKEYLOGFILE", str(keylog))
    client = _create_live_client("test-only-api-key-value")
    try:
        assert str(client.base_url) == "https://api.openai.com/v1/"
        assert client.organization is None
        assert client.project is None
        assert "X-Evil" not in client.default_headers
        assert client._client.follow_redirects is False
        assert client._client._trust_env is False
        assert client._client._transport._pool._ssl_context.keylog_filename is None
        assert not keylog.exists()
    finally:
        client.close()


def test_module_import_does_not_import_openai():
    existing = sys.modules.pop("openai", None)
    try:
        module = importlib.reload(
            sys.modules["experiments.adaptive_selection.openai_manifest_probe"]
        )
        assert module is not None
        assert "openai" not in sys.modules
    finally:
        if existing is not None:
            sys.modules["openai"] = existing


def test_source_uses_python39_compatible_grammar():
    source = (
        ROOT / "experiments/adaptive_selection/openai_manifest_probe.py"
    ).read_text()
    compile(source, "openai_manifest_probe.py", "exec", dont_inherit=True)
    ast.parse(source, filename="openai_manifest_probe.py", feature_version=(3, 9))
    assert " | None" not in source
