import base64
import inspect
import json
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from types import MappingProxyType

import pytest

from experiments.adaptive_selection.providers import (
    PRIMARY_COMPARABILITY_FIELDS,
    TOKEN_ACCOUNTING_VERSION,
    DeterministicFakeProvider,
    IncompatibleManifestError,
    ManifestComparison,
    ManifestConsistencyError,
    ManifestDifference,
    ManifestInputs,
    ProviderCallbackError,
    ProviderConfiguration,
    ProviderExecution,
    ProviderFixtureNotFoundError,
    ProviderIdentityMismatchError,
    ProviderRequest,
    ProviderValidationError,
    RawTransportResult,
    RecordedCallbackProvider,
    TokenAccountingUnavailableError,
    build_run_manifest,
    compare_manifests,
    validate_execution,
    validate_request_manifest,
)
from experiments.adaptive_selection.schema import RunManifest

UTC_1 = "2026-07-29T12:00:00Z"
UTC_2 = "2026-07-29T12:00:00.123456Z"


def configuration(**changes):
    values = dict(
        provider="recorded",
        model_id="model-1",
        provider_revision="revision-7",
        temperature=0.25,
        seed=17,
        seed_supported=True,
        tool_availability=("search", "calculator"),
        token_accounting_version=TOKEN_ACCOUNTING_VERSION,
        generation_options={"top_p": 0.9, "nested": [True, None, {"n": 2}]},
    )
    values.update(changes)
    return ProviderConfiguration(**values)


def request(**changes):
    values = dict(
        prompt_text="Rendered prompt\nwithout normalization.",
        prompt_template_hash="sha256:" + "1" * 64,
    )
    values.update(changes)
    return ProviderRequest(**values)


def transport(**changes):
    values = dict(
        observed_provider="recorded",
        observed_model_id="model-1",
        observed_provider_revision="revision-7",
        response_text="answer",
        raw_response_bytes=b'{"answer":"answer"}',
        input_tokens=12,
        output_tokens=3,
        provider_request_id="req-1",
    )
    values.update(changes)
    return RawTransportResult(**values)


def inputs(**changes):
    values = dict(
        run_id="run-1",
        experiment_version="experiment-v1",
        protocol_version="protocol-v1",
        dataset_version="dataset-v1",
        dataset_hash="sha256:" + "2" * 64,
        selector_mode="adaptive",
        selector_version="selector-v1",
        code_revision="abc123",
        provenance="test:task8",
    )
    values.update(changes)
    return ManifestInputs(**values)


def clocks():
    utc_values = iter((UTC_1, UTC_2))
    monotonic_values = iter((10.0, 10.125))
    return lambda: next(utc_values), lambda: next(monotonic_values)


def execution(config=None, req=None, result=None):
    config = config or configuration()
    req = req or request()
    result = result or transport()
    utc_clock, monotonic_clock = clocks()
    return RecordedCallbackProvider(
        config, lambda actual_config, actual: result, utc_clock, monotonic_clock
    ).execute(req)


def manifest(config=None, req=None, **changes):
    config = config or configuration()
    req = req or request()
    return build_run_manifest(inputs(**changes), config, req, lambda: UTC_1)


def test_configuration_roundtrip_hash_canonicalization_and_recursive_immutability():
    source = {"z": [3, {"b": 2, "a": 1}], "a": "first"}
    config = configuration(
        tool_availability=("search", "calculator"), generation_options=source
    )
    source["z"][1]["a"] = 999

    assert config.tool_availability == ("calculator", "search")
    assert isinstance(config.generation_options, MappingProxyType)
    assert config.to_dict()["generation_options"] == {
        "a": "first",
        "z": [3, {"a": 1, "b": 2}],
    }
    assert config.config_hash.startswith("sha256:")
    assert len(config.config_hash) == 71
    assert ProviderConfiguration.from_dict(config.to_dict()) == config
    assert json.loads(config.canonical_bytes()) == config.to_dict()
    with pytest.raises(TypeError):
        config.generation_options["new"] = 1
    with pytest.raises(TypeError):
        config.generation_options["z"][1]["a"] = 4


def test_configuration_hash_is_order_independent_and_sensitive_to_every_input():
    left = configuration(generation_options={"b": 2, "a": 1})
    right = configuration(generation_options={"a": 1, "b": 2})
    assert left.config_hash == right.config_hash

    for field, value in (
        ("provider", "other"),
        ("model_id", "other"),
        ("provider_revision", "other"),
        ("temperature", 0.5),
        ("seed", 18),
        ("seed_supported", False),
        ("tool_availability", ("other",)),
        ("generation_options", {"different": True}),
    ):
        changes = {field: value}
        if field == "seed_supported":
            changes["seed"] = None
        assert configuration(**changes).config_hash != configuration().config_hash


def test_configuration_from_dict_requires_and_validates_derived_hash_and_exact_shape():
    payload = configuration().to_dict()
    for key in tuple(payload):
        broken = dict(payload)
        broken.pop(key)
        with pytest.raises(ProviderValidationError):
            ProviderConfiguration.from_dict(broken)
    with pytest.raises(ProviderValidationError, match="unexpected fields"):
        ProviderConfiguration.from_dict({**payload, "extra": True})
    with pytest.raises(ProviderValidationError, match="config_hash"):
        ProviderConfiguration.from_dict(
            {**payload, "config_hash": "sha256:" + "0" * 64}
        )
    with pytest.raises(TypeError):
        ProviderConfiguration(**{**payload, "config_hash": payload["config_hash"]})


def test_configuration_strict_bounds_and_seed_tool_invariants():
    bad_changes = (
        {"provider": ""},
        {"model_id": "x" * 257},
        {"temperature": True},
        {"temperature": float("nan")},
        {"temperature": 2.01},
        {"seed": 2**63},
        {"seed": None},
        {"seed_supported": False, "seed": 1},
        {"seed_supported": "yes"},
        {"tool_availability": {"search"}},
        {"tool_availability": ("search", "search")},
        {"tool_availability": tuple(str(i) for i in range(129))},
        {"token_accounting_version": "another-convention"},
    )
    for changes in bad_changes:
        with pytest.raises(ProviderValidationError):
            configuration(**changes)


def test_generation_options_reject_custom_json_types_depth_nodes_numbers_and_size():
    class DictSubclass(dict):
        pass

    class ListSubclass(list):
        pass

    for options in (
        DictSubclass(a=1),
        {"x": ListSubclass([1])},
        {1: "bad"},
        {"bad": object()},
        {"bad": float("inf")},
        {"bad": 2**63},
        {"too_large": "x" * (1024 * 1024)},
    ):
        with pytest.raises(ProviderValidationError):
            configuration(generation_options=options)
    deep = value = {}
    for _ in range(33):
        child = {}
        value["x"] = child
        value = child
    with pytest.raises(ProviderValidationError, match="depth"):
        configuration(generation_options=deep)


def test_request_roundtrip_hash_sensitivity_no_normalization_and_bounds():
    req = request()
    assert ProviderRequest.from_dict(req.to_dict()) == req
    assert json.loads(req.canonical_bytes()) == req.to_dict()
    assert (
        request(prompt_text="x").request_hash != request(prompt_text="x ").request_hash
    )
    assert (
        request(prompt_template_hash="sha256:" + "3" * 64).request_hash
        != req.request_hash
    )
    for bad in ("", "x" * (10 * 1024 * 1024 + 1)):
        with pytest.raises(ProviderValidationError):
            request(prompt_text=bad)
    with pytest.raises(ProviderValidationError):
        request(prompt_template_hash="sha256:not-a-hash")
    with pytest.raises(ProviderValidationError, match="request_hash"):
        ProviderRequest.from_dict(
            {**req.to_dict(), "request_hash": "sha256:" + "0" * 64}
        )


def test_raw_transport_roundtrip_exact_base64_defensive_bytes_and_bounds():
    source = bytearray(b"raw\x00bytes")
    result = transport(raw_response_bytes=source)
    source[0] = ord("X")
    assert result.raw_response_bytes == b"raw\x00bytes"
    payload = result.to_dict()
    assert payload["raw_response_bytes"] == {
        "encoding": "base64",
        "data": base64.b64encode(b"raw\x00bytes").decode("ascii"),
    }
    assert RawTransportResult.from_dict(payload) == result
    assert json.loads(result.canonical_bytes()) == payload

    for changes in (
        {"raw_response_bytes": b""},
        {"raw_response_bytes": "not-bytes"},
        {"input_tokens": None},
        {"input_tokens": True},
        {"input_tokens": -1},
        {"output_tokens": 2**63},
        {"provider_request_id": "x" * 1025},
    ):
        with pytest.raises(ProviderValidationError):
            transport(**changes)


def test_raw_transport_from_dict_rejects_noncanonical_or_invalid_base64():
    payload = transport().to_dict()
    for envelope in (
        {"encoding": "base64", "data": "%%%"},
        {"encoding": "base64", "data": "YQ"},
        {"encoding": "hex", "data": "61"},
        {"encoding": "base64", "data": "YQ==", "extra": 1},
    ):
        with pytest.raises(ProviderValidationError):
            RawTransportResult.from_dict({**payload, "raw_response_bytes": envelope})


def test_adapter_executes_callback_once_binds_identity_tokens_bytes_and_timing():
    calls = []
    req = request()
    config = configuration()
    utc_clock, monotonic_clock = clocks()
    provider = RecordedCallbackProvider(
        config,
        lambda actual_config, actual: calls.append((actual_config, actual))
        or transport(),
        utc_clock,
        monotonic_clock,
    )

    result = provider.execute(req)

    assert calls == [(config, req)]
    assert provider.configuration == config
    assert result.provider == config.provider
    assert result.model_id == config.model_id
    assert result.provider_revision == config.provider_revision
    assert result.config_hash == config.config_hash
    assert result.request_hash == req.request_hash
    assert result.prompt_template_hash == req.prompt_template_hash
    assert result.raw_response_bytes == transport().raw_response_bytes
    assert result.input_tokens == 12 and result.output_tokens == 3
    assert result.token_accounting_version == TOKEN_ACCOUNTING_VERSION
    assert result.started_timestamp == UTC_1
    assert result.completed_timestamp == UTC_2
    assert result.latency_ms == 125.0


def test_execution_raw_hash_depends_only_on_raw_bytes_and_roundtrips_exact_payload():
    one = execution(result=transport(response_text="one", raw_response_bytes=b"same"))
    two = execution(result=transport(response_text="two", raw_response_bytes=b"same"))
    changed = execution(
        result=transport(response_text="one", raw_response_bytes=b"samf")
    )
    assert one.raw_response_hash == two.raw_response_hash
    assert one.raw_response_hash != changed.raw_response_hash
    assert ProviderExecution.from_dict(one.to_dict()) == one
    assert json.loads(one.canonical_bytes()) == one.to_dict()

    payload = one.to_dict()
    with pytest.raises(ProviderValidationError, match="raw_response_hash"):
        ProviderExecution.from_dict(
            {**payload, "raw_response_hash": "sha256:" + "0" * 64}
        )
    raw = dict(payload["raw_response_bytes"])
    raw["data"] = base64.b64encode(b"changed").decode("ascii")
    with pytest.raises(ProviderValidationError, match="raw_response_hash"):
        ProviderExecution.from_dict({**payload, "raw_response_bytes": raw})


def test_adapter_rejects_callback_type_identity_and_tampered_records():
    config = configuration()
    req = request()
    for callback, error in (
        (lambda actual_config, actual: object(), ProviderValidationError),
        (
            lambda actual_config, actual: transport(observed_model_id="spoofed"),
            ProviderIdentityMismatchError,
        ),
    ):
        utc_clock, monotonic_clock = clocks()
        with pytest.raises(error):
            RecordedCallbackProvider(
                config, callback, utc_clock, monotonic_clock
            ).execute(req)

    bad = transport()
    object.__setattr__(bad, "input_tokens", None)
    utc_clock, monotonic_clock = clocks()
    with pytest.raises(TokenAccountingUnavailableError):
        RecordedCallbackProvider(
            config, lambda actual_config, actual: bad, utc_clock, monotonic_clock
        ).execute(req)


def test_callback_exception_is_chained_sanitized_and_records_timing():
    class SecretFailure(Exception):
        pass

    def callback(actual_config, actual):
        raise SecretFailure("credential=do-not-copy")

    utc_clock, monotonic_clock = clocks()
    with pytest.raises(ProviderCallbackError) as raised:
        RecordedCallbackProvider(
            configuration(), callback, utc_clock, monotonic_clock
        ).execute(request())

    error = raised.value
    assert isinstance(error.__cause__, SecretFailure)
    assert "do-not-copy" not in str(error)
    assert error.category == "provider_callback_exception"
    assert error.started_timestamp == UTC_1
    assert error.completed_timestamp == UTC_2
    assert error.latency_ms == 125.0


def test_adapter_does_not_catch_base_exception():
    utc_clock, monotonic_clock = clocks()

    def callback(actual_config, actual):
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        RecordedCallbackProvider(
            configuration(), callback, utc_clock, monotonic_clock
        ).execute(request())


def test_clocks_require_canonical_utc_and_finite_ordered_monotonic_values():
    bad_utc = (
        datetime(2026, 7, 29, 12, 0),
        datetime(2026, 7, 29, 13, 0, tzinfo=timezone(timedelta(hours=1))),
        "2026-07-29T12:00:00.1Z",
        "2026-07-29T12:00:00.1234567Z",
        "2026-07-29T12:00:00+00:00",
    )
    for value in bad_utc:
        with pytest.raises(ProviderValidationError):
            RecordedCallbackProvider(
                configuration(),
                lambda actual_config, actual: transport(),
                lambda: value,
                iter((1.0, 2.0)).__next__,
            ).execute(request())

    for values in ((True, 2.0), (1.0, float("inf")), (2.0, 1.0)):
        utc_values = iter((UTC_1, UTC_2))
        with pytest.raises(ProviderValidationError):
            RecordedCallbackProvider(
                configuration(),
                lambda actual_config, actual: transport(),
                utc_values.__next__,
                iter(values).__next__,
            ).execute(request())


def test_datetime_clock_is_rendered_as_seconds_or_exactly_six_fractional_digits():
    utc_values = iter(
        (
            datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc),
            datetime(2026, 7, 29, 12, 0, 0, 1, tzinfo=timezone.utc),
        )
    )
    execution_result = RecordedCallbackProvider(
        configuration(),
        lambda actual_config, actual: transport(),
        utc_values.__next__,
        iter((1.0, 1.0)).__next__,
    ).execute(request())
    assert execution_result.started_timestamp == UTC_1
    assert execution_result.completed_timestamp == "2026-07-29T12:00:00.000001Z"


def test_fake_provider_routes_only_by_request_hash_and_is_order_independent():
    first = request(prompt_text="first")
    second = request(prompt_text="second")
    fixtures = (
        (
            second.request_hash,
            transport(response_text="second", raw_response_bytes=b"2"),
        ),
        (first.request_hash, transport(response_text="first", raw_response_bytes=b"1")),
    )
    utc_clock, monotonic_clock = clocks()
    provider = DeterministicFakeProvider(
        configuration(), reversed(fixtures), utc_clock, monotonic_clock
    )
    assert provider.execute(first).response_text == "first"

    utc_clock, monotonic_clock = clocks()
    with pytest.raises(ProviderFixtureNotFoundError):
        DeterministicFakeProvider(
            configuration(), fixtures, utc_clock, monotonic_clock
        ).execute(request(prompt_text="unknown"))


def test_fake_provider_rejects_duplicate_hashes_and_defensively_copies_fixtures():
    req = request()
    item = transport()
    with pytest.raises(ProviderValidationError, match="duplicate"):
        DeterministicFakeProvider(
            configuration(),
            ((req.request_hash, item), (req.request_hash, item)),
            lambda: UTC_1,
            lambda: 1.0,
        )

    utc_clock, monotonic_clock = clocks()
    provider = DeterministicFakeProvider(
        configuration(), ((req.request_hash, item),), utc_clock, monotonic_clock
    )
    object.__setattr__(item, "response_text", "tampered")
    assert provider.execute(req).response_text == "answer"


def test_manifest_inputs_and_difference_records_are_frozen_bounded_and_roundtrip():
    record = inputs()
    assert ManifestInputs.from_dict(record.to_dict()) == record
    assert json.loads(record.canonical_bytes()) == record.to_dict()
    with pytest.raises(FrozenInstanceError):
        record.run_id = "other"
    with pytest.raises(ProviderValidationError):
        inputs(provenance="x" * 1025)

    difference = ManifestDifference("temperature", 0.0, 0.25)
    assert ManifestDifference.from_dict(difference.to_dict()) == difference
    with pytest.raises(ProviderValidationError):
        ManifestDifference("field", {"not": "scalar"}, "right")


def test_build_manifest_is_unspoofable_deterministic_and_binds_one_timestamp():
    config = configuration()
    req = request()
    calls = []

    def clock():
        calls.append(True)
        return UTC_1

    first = build_run_manifest(inputs(), config, req, clock)
    second = build_run_manifest(inputs(), config, req, lambda: UTC_1)
    assert first == second
    assert calls == [True]
    assert first.provider == config.provider
    assert first.model_id == config.model_id
    assert first.config_hash == config.config_hash
    assert first.temperature == config.temperature
    assert first.seed == config.seed
    assert first.seed_supported == config.seed_supported
    assert first.tool_availability == config.tool_availability
    assert first.prompt_template_hash == req.prompt_template_hash
    assert first.started_timestamp == UTC_1
    assert set(inspect.signature(ManifestInputs).parameters) == {
        "run_id",
        "experiment_version",
        "protocol_version",
        "dataset_version",
        "dataset_hash",
        "selector_mode",
        "selector_version",
        "code_revision",
        "provenance",
    }


def test_build_and_validators_reconstruct_exact_records_and_reject_subclasses_tampering():
    class ConfigSubclass(ProviderConfiguration):
        pass

    class ManifestSubclass(RunManifest):
        pass

    with pytest.raises(ProviderValidationError):
        ConfigSubclass("p", "m", "r", 0.0, 1, True, (), TOKEN_ACCOUNTING_VERSION, {})

    config = configuration()
    req = request()
    run = manifest(config, req)
    validate_request_manifest(run, config, req)
    validate_execution(run, config, req, execution(config, req))

    with pytest.raises(ManifestConsistencyError):
        validate_request_manifest(
            replace(run, config_hash="sha256:" + "0" * 64), config, req
        )
    with pytest.raises(ManifestConsistencyError):
        validate_execution(
            run, config, req, execution(config, request(prompt_text="other"))
        )
    with pytest.raises(ManifestConsistencyError):
        validate_request_manifest(ManifestSubclass(**run.to_dict()), config, req)

    object.__setattr__(config, "provider", "tampered")
    with pytest.raises(ProviderValidationError):
        build_run_manifest(inputs(), config, req, lambda: UTC_1)


def test_primary_comparability_fields_are_exact_and_all_included_mutations_fail():
    assert PRIMARY_COMPARABILITY_FIELDS == (
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
    base = manifest()
    replacements = {
        "experiment_version": "other",
        "protocol_version": "other",
        "dataset_version": "other",
        "dataset_hash": "sha256:" + "3" * 64,
        "provider": "other",
        "model_id": "other",
        "prompt_template_hash": "sha256:" + "4" * 64,
        "config_hash": "sha256:" + "5" * 64,
        "code_revision": "other",
        "temperature": 1.0,
        "seed": 99,
        "tool_availability": ("other",),
    }
    for field in PRIMARY_COMPARABILITY_FIELDS:
        with pytest.raises(IncompatibleManifestError) as raised:
            compare_manifests(
                base, replace(base, run_id="other", **{field: replacements[field]})
            )
        assert tuple(item.field for item in raised.value.differences) == (field,)
        assert str(replacements[field]) not in str(raised.value)


def test_excluded_manifest_fields_are_compatible_and_projection_reconstructs_tampering():
    base = manifest()
    changed = replace(
        base,
        run_id="other-run",
        selector_mode="baseline",
        selector_version="selector-v2",
        started_timestamp=UTC_2,
        provenance="other:provenance",
    )
    comparison = compare_manifests(base, changed)
    assert comparison.differences == ()
    assert comparison.valid_for_primary_comparison is True
    assert comparison.override_applied is False
    assert comparison.label == "primary_comparison"
    assert comparison.override_reason is None

    object.__setattr__(changed, "temperature", float("nan"))
    with pytest.raises(ManifestConsistencyError):
        compare_manifests(base, changed)


def test_manifest_override_is_explicit_invalid_for_primary_and_invariant_checked():
    left = manifest()
    right = replace(left, run_id="run-2", dataset_version="dataset-v2")
    comparison = compare_manifests(left, right, override_reason="exploratory only")
    assert comparison.valid_for_primary_comparison is False
    assert comparison.override_applied is True
    assert comparison.label == "invalid_primary_comparison"
    assert comparison.override_reason == "exploratory only"
    assert comparison.differences == (
        ManifestDifference("dataset_version", "dataset-v1", "dataset-v2"),
    )
    assert ManifestComparison.from_dict(comparison.to_dict()) == comparison

    with pytest.raises(ProviderValidationError):
        compare_manifests(left, right, override_reason=" ")
    with pytest.raises(ProviderValidationError):
        compare_manifests(
            left, replace(left, run_id="run-2"), override_reason="unneeded"
        )
    with pytest.raises(ProviderValidationError):
        ManifestComparison(
            left.run_id, right.run_id, (), True, True, "primary_comparison", "reason"
        )


def test_fixed_fake_provider_execution_and_manifest_are_byte_identical():
    config = configuration()
    req = request()
    fixture = ((req.request_hash, transport()),)

    def run_once():
        utc_clock, monotonic_clock = clocks()
        provider = DeterministicFakeProvider(
            config, fixture, utc_clock, monotonic_clock
        )
        return (
            build_run_manifest(inputs(), provider, req, lambda: UTC_1),
            provider.execute(req),
        )

    first_manifest, first_execution = run_once()
    second_manifest, second_execution = run_once()
    assert first_manifest.to_dict() == second_manifest.to_dict()
    assert first_execution.canonical_bytes() == second_execution.canonical_bytes()


def test_provider_api_signatures_are_narrow_and_no_automatic_efficacy_claims():
    assert list(inspect.signature(RecordedCallbackProvider).parameters) == [
        "configuration",
        "callback",
        "utc_clock",
        "monotonic_clock",
    ]
    assert list(inspect.signature(DeterministicFakeProvider).parameters) == [
        "configuration",
        "fixtures",
        "utc_clock",
        "monotonic_clock",
    ]
    assert list(inspect.signature(build_run_manifest).parameters) == [
        "inputs",
        "provider",
        "request",
        "utc_clock",
    ]
    assert not any(
        word in ProviderExecution.__doc__.lower()
        for word in ("efficacy", "fairness", "authenticity", "hosted determinism")
    )
