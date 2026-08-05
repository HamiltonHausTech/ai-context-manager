import hashlib
import json
import os
import socket
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from experiments.adaptive_selection import context_sensitivity_execution_v2 as execution
from experiments.adaptive_selection.context_sensitivity_replication_v2 import (
    build_schedule,
    load_contract,
    render_unit_requests,
)

ROOT = Path(__file__).parents[2]
REVISION = "a" * 40
HOST = "sha256:" + "1" * 64
ACCOUNT = "sha256:" + "2" * 64
CREDENTIAL = "sha256:" + "3" * 64
NOW = "2026-08-04T12:00:00.000000Z"


def valid_transport_result(response_id="resp_test"):
    structured = {
        "diagnosis": "ok",
        "supporting_evidence_numbers": [1],
        "missing_evidence": [],
        "confidence": "medium",
        "next_safe_actions": ["verify"],
        "actions_to_avoid": ["guessing"],
    }
    usage = {
        "input_tokens": 1,
        "cached_input_tokens": 0,
        "cache_write_input_tokens": 0,
        "output_tokens": 1,
        "total_tokens": 2,
    }
    raw = json.dumps(
        {
            "id": response_id,
            "model": execution.PINNED_MODEL,
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": json.dumps(structured)}
                    ],
                }
            ],
            "usage": {
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
                "input_tokens_details": {
                    "cached_tokens": 0,
                    "cache_write_tokens": 0,
                },
            },
        },
        separators=(",", ":"),
    ).encode()
    return execution.TransportResult(
        raw,
        {
            "http_status": 200,
            "content_type": "application/json",
            "provider_request_id": "req_test",
            "response_id": response_id,
            "observed_model": execution.PINNED_MODEL,
        },
        structured,
        usage,
        "0.000014",
        "0.0000145",
    )


def private(path):
    path.mkdir(parents=True, mode=0o700)
    os.chmod(path, 0o700)
    return path


def prepared(tmp_path, shuffle=None):
    local = private(tmp_path / "local")
    global_root = private(tmp_path / "global")
    candidate_path = local / "candidate.json"
    mapping_path = local / "mapping.json"
    candidate = execution.prepare_non_authorizing_candidate_v2(
        candidate_path,
        mapping_path,
        code_revision=REVISION,
        owner_identity="repository-owner",
        host_fingerprint=HOST,
        account_fingerprint=ACCOUNT,
        credential_fingerprint=CREDENTIAL,
        issued_at=NOW,
        expires_at="2026-08-04T13:00:00.000000Z",
        maximum_execution_window_seconds=3600,
        token_bytes=lambda n: bytes(range(n)),
        shuffle=shuffle or (lambda values: values.reverse()),
    )
    approval_path = local / "approval.json"
    approval_payload = {
        "version": execution.APPROVAL_VERSION,
        "authorization_id": candidate["candidate"]["authorization_id"],
        "candidate_sha256": candidate["candidate_sha256"],
        "owner_echoed_candidate_sha256_out_of_band": True,
        "owner_identity": "repository-owner",
        "approved_at": NOW,
        "operational_process_evidence_only": True,
    }
    approval = {
        "approval": approval_payload,
        "approval_digest": execution.sha256_canonical(approval_payload),
    }
    execution.write_test_private_record(approval_path, approval)
    context = execution.verify_authority_v2(
        candidate_path,
        approval_path,
        mapping_path,
        REVISION,
        HOST,
        ACCOUNT,
        CREDENTIAL,
        NOW,
    )
    return (
        local,
        global_root,
        candidate_path,
        mapping_path,
        approval_path,
        candidate,
        context,
    )


def test_readiness_manifest_is_zero_authority_and_exactly_pinned():
    value, raw = execution.load_readiness_manifest(ROOT / execution.READINESS_PATH)
    assert hashlib.sha256(raw).hexdigest() == execution.PINNED_READINESS_MANIFEST_SHA256
    assert value["execution"]["network_requests_authorized_by_manifest"] == 0
    assert value["execution"]["approved_scheduled_unit_count"] == 45
    assert value["execution"]["candidate_grants_network_authority"] is False
    assert value["budget"]["conservative_execution_ceiling"] == "1.470820"
    assert value["budget"]["hard_owner_cap"] == "2.00"


def test_private_randomization_has_five_balanced_blocks_and_independent_blind_order():
    mapping = execution.build_private_randomization(
        "auth-" + "1" * 32,
        "2" * 48,
        token_bytes=lambda n: bytes(range(n)),
        shuffle=lambda values: values.reverse(),
    )
    units = build_schedule(load_contract()[0])
    assert len(mapping["execution_blocks"]) == 5
    for block in mapping["execution_blocks"]:
        assert len(block) == 9
        assert {x["base_cell_id"] for x in block} == {u.base_cell_id for u in units}
        assert len({x["unit_id"] for x in block}) == 9
    order = mapping["execution_order"]
    aliases = [entry["assessment_alias"] for entry in mapping["entries"]]
    assert len(order) == len(set(order)) == 45
    assert len(aliases) == len(set(aliases)) == 45
    assert all(
        alias.startswith("assessment-") and len(alias) == 43 for alias in aliases
    )
    assert mapping["assessment_order"] != [
        next(x["assessment_alias"] for x in mapping["entries"] if x["unit_id"] == unit)
        for unit in order
    ]


def test_candidate_is_non_authorizing_binds_both_orders_and_never_writes_approval(
    tmp_path,
):
    local, _, candidate_path, mapping_path, approval_path, envelope, _ = prepared(
        tmp_path
    )
    candidate = envelope["candidate"]
    assert candidate["candidate_grants_network_authority"] is False
    assert candidate["approved_scheduled_unit_count"] == 45
    assert candidate["execution_order_commitment"].startswith("sha256:")
    assert candidate["assessment_mapping_commitment"].startswith("sha256:")
    assert execution.verify_candidate_v2(candidate_path, REVISION) == envelope
    assert (
        execution.verify_mapping_v2(mapping_path, candidate)["mapping_digest"]
        == candidate["private_mapping_commitment"]
    )
    # The fixture, not candidate preparation, creates owner approval.
    assert (
        approval_path.exists()
        and not (local / "owner-approval-written-by-candidate.json").exists()
    )


def test_exact_owner_approval_is_required_before_authority(tmp_path):
    _, _, candidate_path, mapping_path, approval_path, _, _ = prepared(tmp_path)
    value = execution.read_private_record(approval_path)
    value["approval"]["candidate_sha256"] = "sha256:" + "0" * 64
    value["approval_digest"] = execution.sha256_canonical(value["approval"])
    approval_path.unlink()
    execution.write_test_private_record(approval_path, value)
    with pytest.raises(execution.ExecutionFailure):
        execution.verify_authority_v2(
            candidate_path,
            approval_path,
            mapping_path,
            REVISION,
            HOST,
            ACCOUNT,
            CREDENTIAL,
            NOW,
        )


def test_v2_global_namespace_is_home_stable_and_distinct_from_v1(monkeypatch):
    first = execution.default_authority_directory_v2()
    monkeypatch.setenv("HOME", "/tmp/forged-home")
    second = execution.default_authority_directory_v2()
    from experiments.adaptive_selection import context_sensitivity_execution as v1

    assert first == second
    assert first != v1._default_authority_directory()
    assert first.name == execution.AUTHORITY_NAMESPACE


def test_claims_are_unit_keyed_no_overwrite_and_repeated_hashes_do_not_collapse(
    tmp_path,
):
    local, global_root, _, _, _, _, context = prepared(tmp_path)
    units = [
        u for u in build_schedule(load_contract()[0]) if u.base_cell_id == "cell-k4m2"
    ][:2]
    first = execution.publish_unit_claim(global_root, local, units[0], context, NOW)
    second = execution.publish_unit_claim(global_root, local, units[1], context, NOW)
    assert first["claim"]["request_hash"] == second["claim"]["request_hash"]
    assert first["claim"]["unit_id"] != second["claim"]["unit_id"]
    with pytest.raises(FileExistsError):
        execution.publish_unit_claim(global_root, local, units[0], context, NOW)


def test_reconciliation_detects_orphan_terminal_only_raw_only_and_mirror_conflicts(
    tmp_path,
):
    local, global_root, _, _, _, _, context = prepared(tmp_path)
    unit = build_schedule(load_contract()[0])[0]
    assert execution.classify_unit_state(global_root, local, unit, context) == "pending"
    execution.publish_unit_claim(global_root, local, unit, context, NOW)
    assert (
        execution.classify_unit_state(global_root, local, unit, context)
        == "blocked_orphan_claim"
    )
    rogue = build_schedule(load_contract()[0])[1]
    path = global_root / "terminals" / f"{rogue.unit_id}.json"
    execution.write_test_private_record(path, {"bad": True})
    with pytest.raises(execution.ExecutionFailure):
        execution.classify_unit_state(global_root, local, rogue, context)


def test_fake_dispatch_executes_45_once_reuses_exact_repeated_requests_and_resume_skips(
    tmp_path,
):
    local, global_root, candidate_path, mapping_path, approval_path, _, _ = prepared(
        tmp_path
    )
    calls = []
    clients = []

    class Client:
        def close(self):
            pass

    def factory(_key):
        clients.append(Client())
        return clients[-1]

    def fake_dispatch(_client, body, request_hash):
        calls.append((json.dumps(body, sort_keys=True), request_hash))
        return valid_transport_result()

    result = execution.execute_authorized_45_unit_manifest(
        ROOT,
        global_root,
        local,
        candidate_path,
        approval_path,
        mapping_path,
        api_key="test-key",
        client_factory=factory,
        now=NOW,
        code_revision=REVISION,
        host_fingerprint_value=HOST,
        account_fingerprint_value=ACCOUNT,
        credential_fingerprint_value=CREDENTIAL,
        dispatch=fake_dispatch,
        clock=lambda: NOW,
    )
    assert result == {
        "status": "run_complete",
        "dispatches": 45,
        "successes": 45,
        "failures": 0,
        "pending": 0,
    }
    assert len(calls) == 45
    assert len({body for body, _ in calls}) == 9
    assert all("draw_index" not in body and "unit_id" not in body for body, _ in calls)
    again = execution.execute_authorized_45_unit_manifest(
        ROOT,
        global_root,
        local,
        candidate_path,
        approval_path,
        mapping_path,
        api_key="test-key",
        client_factory=lambda key: pytest.fail("client built on terminal resume"),
        now=NOW,
        code_revision=REVISION,
        host_fingerprint_value=HOST,
        account_fingerprint_value=ACCOUNT,
        credential_fingerprint_value=CREDENTIAL,
        dispatch=fake_dispatch,
        clock=lambda: NOW,
    )
    assert again["dispatches"] == 0 and len(calls) == 45


def test_one_failed_fake_dispatch_is_terminal_and_never_retried(tmp_path):
    local, global_root, candidate_path, mapping_path, approval_path, _, _ = prepared(
        tmp_path
    )
    count = 0

    class Client:
        def close(self):
            pass

    def fake(_client, body, request_hash):
        nonlocal count
        count += 1
        if count == 1:
            raise execution.ExecutionFailure("transport_error")
        return valid_transport_result()

    result = execution.execute_authorized_45_unit_manifest(
        ROOT,
        global_root,
        local,
        candidate_path,
        approval_path,
        mapping_path,
        api_key="key",
        client_factory=lambda key: Client(),
        now=NOW,
        code_revision=REVISION,
        host_fingerprint_value=HOST,
        account_fingerprint_value=ACCOUNT,
        credential_fingerprint_value=CREDENTIAL,
        dispatch=fake,
        clock=lambda: NOW,
    )
    assert count == 45 and result["failures"] == 1 and result["successes"] == 44


def test_client_is_constructed_only_after_candidate_and_approval_verification(tmp_path):
    local, global_root, candidate_path, mapping_path, approval_path, _, _ = prepared(
        tmp_path
    )
    approval_path.unlink()
    built = []
    with pytest.raises(execution.ExecutionFailure):
        execution.execute_authorized_45_unit_manifest(
            ROOT,
            global_root,
            local,
            candidate_path,
            approval_path,
            mapping_path,
            api_key="key",
            client_factory=lambda key: built.append(key),
            now=NOW,
            code_revision=REVISION,
            host_fingerprint_value=HOST,
            account_fingerprint_value=ACCOUNT,
            credential_fingerprint_value=CREDENTIAL,
        )
    assert built == []


def test_rehashed_terminal_semantic_contradictions_are_rejected(tmp_path):
    local, global_root, _, _, _, _, context = prepared(tmp_path)
    unit = build_schedule(load_contract()[0])[0]
    visible = execution._evidence_by_cell(ROOT)[unit.base_cell_id]
    result = valid_transport_result()
    execution.publish_unit_claim(global_root, local, unit, context, NOW)
    published = execution.publish_unit_terminal(
        global_root,
        local,
        unit,
        context,
        kind="success",
        dispatch_invoked=True,
        server_acceptance="yes",
        recorded_at=NOW,
        provider_visible_evidence=visible,
        structured_response=result.structured_response,
        raw_bytes=result.raw_bytes,
        usage=result.usage,
        actual_cost=result.actual_cost,
        conservative_cost_upper_bound=result.conservative_cost_upper_bound,
        response_metadata=result.response_metadata,
    )
    assert published["terminal"]["response_metadata"] == result.response_metadata
    path = global_root / "terminals" / f"{unit.unit_id}.json"
    value = execution.read_private_record(path)
    value["terminal"]["server_acceptance"] = "no"
    value["terminal_digest"] = execution.sha256_canonical(value["terminal"])
    path.unlink()
    execution.write_test_private_record(path, value)
    with pytest.raises(execution.ExecutionFailure):
        execution.verify_unit_terminal(global_root, unit, context, visible)


def test_expiry_and_cap_are_checked_fresh_before_each_claim(tmp_path, monkeypatch):
    local, global_root, candidate_path, mapping_path, approval_path, _, _ = prepared(
        tmp_path
    )

    class Client:
        def close(self):
            pass

    calls = []
    times = iter([NOW, NOW, "2026-08-04T13:00:00.000001Z"])
    result = execution.execute_authorized_45_unit_manifest(
        ROOT,
        global_root,
        local,
        candidate_path,
        approval_path,
        mapping_path,
        api_key="key",
        client_factory=lambda key: Client(),
        now=NOW,
        code_revision=REVISION,
        host_fingerprint_value=HOST,
        account_fingerprint_value=ACCOUNT,
        credential_fingerprint_value=CREDENTIAL,
        dispatch=lambda *args: calls.append(1) or valid_transport_result(),
        clock=lambda: next(times),
    )
    assert len(calls) == 1 and result["pending"] == 44

    local2, global2, candidate2, mapping2, approval2, _, _ = prepared(tmp_path / "cap")
    built = []

    def factory(key):
        built.append(key)
        monkeypatch.setattr(
            execution, "_static_upper", lambda body: execution.Decimal("3")
        )
        return Client()

    capped = execution.execute_authorized_45_unit_manifest(
        ROOT,
        global2,
        local2,
        candidate2,
        approval2,
        mapping2,
        api_key="key",
        client_factory=factory,
        now=NOW,
        code_revision=REVISION,
        host_fingerprint_value=HOST,
        account_fingerprint_value=ACCOUNT,
        credential_fingerprint_value=CREDENTIAL,
        dispatch=lambda *args: pytest.fail("dispatch after fresh cap failure"),
        clock=lambda: NOW,
    )
    assert built == ["key"] and capped["dispatches"] == 0


def test_lock_contention_blocks_before_client_construction(tmp_path):
    local, global_root, candidate_path, mapping_path, approval_path, _, _ = prepared(
        tmp_path
    )
    built = []
    with execution.machine_global_run_lock(global_root):
        with pytest.raises(execution.ExecutionFailure) as failure:
            execution.execute_authorized_45_unit_manifest(
                ROOT,
                global_root,
                local,
                candidate_path,
                approval_path,
                mapping_path,
                api_key="key",
                client_factory=lambda key: built.append(key),
                now=NOW,
                code_revision=REVISION,
                host_fingerprint_value=HOST,
                account_fingerprint_value=ACCOUNT,
                credential_fingerprint_value=CREDENTIAL,
            )
    assert failure.value.category == "machine_global_run_lock_unavailable"
    assert built == []


def test_blind_export_omits_controller_metadata_and_preserves_repeated_inputs(tmp_path):
    local, global_root, candidate_path, mapping_path, approval_path, _, _ = prepared(
        tmp_path
    )
    # Produce fixed missing terminals without dispatching, sufficient to exercise export shape.
    context = execution.verify_authority_v2(
        candidate_path,
        approval_path,
        mapping_path,
        REVISION,
        HOST,
        ACCOUNT,
        CREDENTIAL,
        NOW,
    )
    for unit in build_schedule(load_contract()[0]):
        execution.publish_unit_claim(global_root, local, unit, context, NOW)
        execution.publish_unit_terminal(
            global_root,
            local,
            unit,
            context,
            kind="failure",
            dispatch_invoked=False,
            server_acceptance="no",
            recorded_at=NOW,
            failure_category="preflight_rejected",
        )
    export = execution.build_blind_assessment_v2(
        ROOT,
        global_root,
        local,
        candidate_path,
        mapping_path,
        REVISION,
        HOST,
        ACCOUNT,
        CREDENTIAL,
        approval_path,
    )
    assert set(export) == {"version", "assessment_ready", "assessments"}
    assert len(export["assessments"]) == 45
    serialized = json.dumps(export)
    for forbidden in (
        "unit_id",
        "base_cell_id",
        "draw_index",
        "request_hash",
        "execution_order",
        "provider_request_id",
        "recorded_at",
    ):
        assert forbidden not in serialized
    assert {item["status"] for item in export["assessments"]} == {"failure"}
    assert all(
        item["criteria"] and item["adjudication_rules"]
        for item in export["assessments"]
    )


def test_annotation_lock_is_alias_only_no_overwrite_and_resolves_into_v2_scorer(
    tmp_path,
):
    (
        local,
        global_root,
        candidate_path,
        mapping_path,
        approval_path,
        candidate,
        context,
    ) = prepared(tmp_path)
    mapping = execution.verify_mapping_v2(mapping_path, candidate["candidate"])[
        "mapping"
    ]
    units = build_schedule(load_contract()[0])
    visible = execution._evidence_by_cell(ROOT)
    for index, unit in enumerate(units):
        execution.publish_unit_claim(global_root, local, unit, context, NOW)
        if index == 0:
            result = valid_transport_result()
            execution.publish_unit_terminal(
                global_root,
                local,
                unit,
                context,
                kind="success",
                dispatch_invoked=True,
                server_acceptance="yes",
                recorded_at=NOW,
                provider_visible_evidence=visible[unit.base_cell_id],
                structured_response=result.structured_response,
                raw_bytes=result.raw_bytes,
                usage=result.usage,
                actual_cost=result.actual_cost,
                conservative_cost_upper_bound=result.conservative_cost_upper_bound,
                response_metadata=result.response_metadata,
            )
        else:
            execution.publish_unit_terminal(
                global_root,
                local,
                unit,
                context,
                kind="failure",
                dispatch_invoked=False,
                server_acceptance="no",
                recorded_at=NOW,
                failure_category="preflight_rejected",
            )
    blind = execution.build_blind_assessment_v2(
        ROOT,
        global_root,
        local,
        candidate_path,
        mapping_path,
        REVISION,
        HOST,
        ACCOUNT,
        CREDENTIAL,
        approval_path,
    )
    successful = next(x for x in blind["assessments"] if x["status"] == "success")
    annotations = [
        {
            "assessment_alias": successful["assessment_alias"],
            "criteria": {x["criterion_id"]: "met" for x in successful["criteria"]},
            "critical_finding": False,
        }
    ]
    with pytest.raises(execution.ExecutionFailure):
        execution.publish_annotation_lock(
            local / "incomplete-lock.json", [], candidate["candidate"], mapping, blind
        )
    assert not (local / "incomplete-lock.json").exists()
    bad = [dict(annotations[0], criteria={})]
    with pytest.raises(execution.ExecutionFailure):
        execution.publish_annotation_lock(
            local / "bad-criteria-lock.json",
            bad,
            candidate["candidate"],
            mapping,
            blind,
        )
    substituted = dict(blind)
    substituted["assessments"] = list(blind["assessments"])
    substituted["assessments"][0] = dict(substituted["assessments"][0])
    substituted["assessments"][0]["task"] += " substituted"
    lock_path = local / "annotation-lock.json"
    lock = execution.publish_annotation_lock(
        lock_path, annotations, candidate["candidate"], mapping, blind
    )
    assert (
        execution.verify_annotation_lock(
            lock_path, candidate["candidate"], mapping, blind
        )["annotation_lock_digest"]
        == lock["annotation_lock_digest"]
    )
    with pytest.raises(execution.ExecutionFailure):
        execution.verify_annotation_lock(
            lock_path, candidate["candidate"], mapping, substituted
        )
    with pytest.raises(FileExistsError):
        execution.publish_annotation_lock(
            lock_path, annotations, candidate["candidate"], mapping, blind
        )
    resolved = execution.resolve_locked_annotations_v2(
        lock_path, mapping_path, candidate["candidate"], blind
    )
    assert len(resolved) == 1 and all(
        set(x)
        == {"unit_id", "criteria", "critical_finding", "locked", "evidence_coherent"}
        for x in resolved
    )


def test_orphan_finalization_requires_exact_external_claim_digest(tmp_path):
    local, global_root, _, _, _, _, context = prepared(tmp_path)
    unit = build_schedule(load_contract()[0])[0]
    claim = execution.publish_unit_claim(global_root, local, unit, context, NOW)
    auth_path = local / "orphan-authorization.json"
    payload = {
        "version": execution.ORPHAN_FINALIZATION_VERSION,
        "contract_lineage": execution.CONTRACT_LINEAGE,
        "authorization_id": context.authorization_id,
        "candidate_digest": context.candidate_digest,
        "approval_digest": context.approval_digest,
        "unit_id": unit.unit_id,
        "orphan_claim_digest": claim["claim_digest"],
        "owner_echoed_orphan_claim_digest_out_of_band": True,
        "owner_confirmed_no_process_remains": True,
        "confirmation_process": "Owner inspected the machine-global lock and process table out of band.",
        "owner_identity": context.owner_identity,
        "authorized_at": NOW,
        "operational_process_evidence_only": True,
    }
    execution.write_test_private_record(
        auth_path,
        {
            "finalization_authorization": payload,
            "finalization_authorization_digest": execution.sha256_canonical(payload),
        },
    )
    terminal = execution.finalize_orphan_unit_offline(
        global_root, local, unit, context, auth_path, NOW
    )
    assert terminal["terminal"]["kind"] == "invalid_ambiguous"


def test_default_cli_and_dry_run_are_offline_and_untainted(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-read")
    monkeypatch.setattr(
        execution.v1_execution,
        "_load_credential",
        lambda *args: pytest.fail("credential loader called by default CLI"),
    )
    monkeypatch.setattr(
        execution.v1_execution,
        "_create_live_client",
        lambda *args: pytest.fail("provider client created by default CLI"),
    )
    monkeypatch.setattr(
        execution,
        "_write_private_bytes_no_overwrite",
        lambda *args: pytest.fail("private state written by default CLI"),
    )
    monkeypatch.setattr(
        socket,
        "socket",
        lambda *args, **kwargs: pytest.fail("network socket opened by default CLI"),
    )
    assert execution.main([], _repo_root=ROOT) == 0
    output = capsys.readouterr().out
    assert (
        "network_attempts=0" in output and "credential_readiness=not_checked" in output
    )
    summary = execution.build_dry_run_summary(ROOT)
    assert summary["network_requests_authorized"] == 0
    serialized = json.dumps(summary)
    assert (
        "Task:" not in serialized
        and "Evidence:" not in serialized
        and "rubric" not in serialized
    )


def test_private_publication_is_atomic_and_rejects_symlink_ancestors_and_partial_finals(
    tmp_path, monkeypatch
):
    root = private(tmp_path / "private")
    final = root / "record.json"
    real_write = os.write
    calls = 0

    def crashing_write(fd, data):
        nonlocal calls
        calls += 1
        if calls == 1:
            real_write(fd, data[: max(1, len(data) // 2)])
            raise OSError("simulated crash")
        return real_write(fd, data)

    monkeypatch.setattr(os, "write", crashing_write)
    with pytest.raises(execution.ExecutionFailure):
        execution.write_test_private_record(final, {"complete": True})
    assert not final.exists()
    assert not list(root.glob(".task12b-v2-publish-*"))
    monkeypatch.setattr(os, "write", real_write)

    partial = root / "partial.json"
    partial.write_bytes(b'{"complete":')
    os.chmod(partial, 0o600)
    with pytest.raises(execution.ExecutionFailure):
        execution.read_private_record(partial)

    target = private(tmp_path / "target")
    ancestor = tmp_path / "ancestor"
    ancestor.symlink_to(target, target_is_directory=True)
    with pytest.raises(execution.ExecutionFailure):
        execution.write_test_private_record(ancestor / "bad.json", {"x": 1})


def test_authority_namespace_is_exact_lineage_only_and_cannot_be_reset(monkeypatch):
    expected = hashlib.sha256(execution.CONTRACT_LINEAGE.encode()).hexdigest()
    original = execution.default_authority_directory_v2()
    assert execution.AUTHORITY_NAMESPACE == expected
    assert original.parent.name == "task12b-authority"
    assert original.name == expected
    monkeypatch.setattr(execution, "READINESS_HASH", "sha256:" + "9" * 64)
    assert execution.default_authority_directory_v2() == original


def test_mapping_rejects_wrong_base_binding_mixed_and_duplicate_draw_blocks(tmp_path):
    *_, envelope, _ = prepared(tmp_path)
    mapping = execution.read_private_record(tmp_path / "local" / "mapping.json")[
        "mapping"
    ]

    wrong_base = json.loads(json.dumps(mapping))
    wrong_base["execution_blocks"][0][0]["base_cell_id"] = wrong_base[
        "execution_blocks"
    ][0][1]["base_cell_id"]
    with pytest.raises(execution.ExecutionFailure):
        execution._mapping_semantics(wrong_base, envelope["candidate"])

    mixed = json.loads(json.dumps(mapping))
    mixed["execution_blocks"][0][0], mixed["execution_blocks"][1][0] = (
        mixed["execution_blocks"][1][0],
        mixed["execution_blocks"][0][0],
    )
    mixed["execution_order"] = [
        x["unit_id"] for block in mixed["execution_blocks"] for x in block
    ]
    with pytest.raises(execution.ExecutionFailure):
        execution._mapping_semantics(mixed, envelope["candidate"])

    duplicate_draw = json.loads(json.dumps(mapping))
    duplicate_draw["execution_blocks"][1] = json.loads(
        json.dumps(duplicate_draw["execution_blocks"][0])
    )
    duplicate_draw["execution_order"] = [
        x["unit_id"] for block in duplicate_draw["execution_blocks"] for x in block
    ]
    with pytest.raises(execution.ExecutionFailure):
        execution._mapping_semantics(duplicate_draw, envelope["candidate"])


def test_candidate_binds_complete_frozen_requests_provider_pricing_and_exact_ceiling(
    tmp_path,
):
    *_, envelope, _ = prepared(tmp_path)
    candidate = envelope["candidate"]
    assert candidate["readiness_manifest_bytes_sha256"] == execution.READINESS_HASH
    assert (
        candidate["readiness_manifest"]["provider_configuration"]["model"]
        == execution.PINNED_MODEL
    )
    assert candidate["provider_settings"] == execution.PINNED_PROVIDER_SETTINGS
    assert candidate["prices"] == execution.PINNED_PRICES
    assert len(candidate["ordered_request_hashes"]) == 45
    assert len(set(candidate["ordered_request_hashes"])) == 9
    assert len(candidate["static_upper_bounds"]) == 45
    assert (
        format(sum(map(Decimal, candidate["static_upper_bounds"])), ".6f") == "1.470820"
    )
    assert candidate["calculated_static_ceiling"] == "1.470820"


@pytest.mark.parametrize(
    "field",
    [
        "readiness_manifest_hash",
        "readiness_manifest",
        "ordered_request_hashes",
        "static_upper_bounds",
        "provider_settings",
        "prices",
        "calculated_static_ceiling",
        "conservative_execution_ceiling",
        "no_retry_policy",
    ],
)
def test_every_frozen_candidate_fact_mutation_fails_even_after_rehash(tmp_path, field):
    _, _, candidate_path, _, _, envelope, _ = prepared(tmp_path)
    changed = json.loads(json.dumps(envelope))
    value = changed["candidate"][field]
    if isinstance(value, dict):
        value[sorted(value)[0]] = "mutated"
    elif isinstance(value, list):
        value[0] = "mutated"
    else:
        changed["candidate"][field] = "mutated"
    changed["candidate_sha256"] = execution.sha256_canonical(changed["candidate"])
    candidate_path.unlink()
    execution.write_test_private_record(candidate_path, changed)
    with pytest.raises(execution.ExecutionFailure):
        execution.verify_candidate_v2(candidate_path, REVISION)


def test_claim_mirror_publication_crash_consumes_global_authority_fail_closed(
    tmp_path, monkeypatch
):
    local, global_root, _, _, _, _, context = prepared(tmp_path)
    unit = build_schedule(load_contract()[0])[0]
    original = execution._write_private_bytes_no_overwrite
    count = 0

    def crash_local(path, data):
        nonlocal count
        count += 1
        if count == 2:
            raise execution.ExecutionFailure("simulated_local_claim_crash")
        return original(path, data)

    monkeypatch.setattr(execution, "_write_private_bytes_no_overwrite", crash_local)
    with pytest.raises(execution.ExecutionFailure):
        execution.publish_unit_claim(global_root, local, unit, context, NOW)
    monkeypatch.setattr(execution, "_write_private_bytes_no_overwrite", original)
    assert (global_root / "claims" / (unit.unit_id + ".json")).exists()
    with pytest.raises(execution.ExecutionFailure):
        execution.classify_unit_state(global_root, local, unit, context)


@pytest.mark.parametrize("publication", [1, 2, 3, 4])
def test_each_terminal_publication_crash_is_orphan_or_fail_closed(
    tmp_path, monkeypatch, publication
):
    local, global_root, _, _, _, _, context = prepared(tmp_path)
    unit = build_schedule(load_contract()[0])[0]
    execution.publish_unit_claim(global_root, local, unit, context, NOW)
    result = valid_transport_result()
    original = execution._write_private_bytes_no_overwrite
    count = 0

    def crash_at(path, data):
        nonlocal count
        count += 1
        if count == publication:
            raise execution.ExecutionFailure("simulated_publication_crash")
        return original(path, data)

    monkeypatch.setattr(execution, "_write_private_bytes_no_overwrite", crash_at)
    with pytest.raises(execution.ExecutionFailure):
        execution.publish_unit_terminal(
            global_root,
            local,
            unit,
            context,
            kind="success",
            dispatch_invoked=True,
            server_acceptance="yes",
            recorded_at=NOW,
            provider_visible_evidence=execution._evidence_by_cell(ROOT)[
                unit.base_cell_id
            ],
            structured_response=result.structured_response,
            raw_bytes=result.raw_bytes,
            usage=result.usage,
            actual_cost=result.actual_cost,
            conservative_cost_upper_bound=result.conservative_cost_upper_bound,
            response_metadata=result.response_metadata,
        )
    monkeypatch.setattr(execution, "_write_private_bytes_no_overwrite", original)
    if publication == 1:
        assert (
            execution.classify_unit_state(global_root, local, unit, context)
            == "blocked_orphan_claim"
        )
    else:
        with pytest.raises(execution.ExecutionFailure):
            execution.classify_unit_state(global_root, local, unit, context)


def test_terminal_index_is_required_and_index_only_fails_closed(tmp_path):
    local, global_root, _, _, _, _, context = prepared(tmp_path)
    unit, rogue = build_schedule(load_contract()[0])[:2]
    execution.publish_unit_claim(global_root, local, unit, context, NOW)
    execution.publish_unit_terminal(
        global_root,
        local,
        unit,
        context,
        kind="failure",
        dispatch_invoked=False,
        server_acceptance="no",
        recorded_at=NOW,
        failure_category="preflight_rejected",
    )
    index = global_root / "terminal-index" / (unit.unit_id + ".json")
    assert index.exists()
    index.unlink()
    with pytest.raises(execution.ExecutionFailure):
        execution.classify_unit_state(global_root, local, unit, context)
    execution.write_test_private_record(
        global_root / "terminal-index" / (rogue.unit_id + ".json"), {"rogue": True}
    )
    with pytest.raises(execution.ExecutionFailure):
        execution.classify_unit_state(global_root, local, rogue, context)


def test_orphan_finalization_rejects_weak_confirmation_and_chronology(tmp_path):
    local, global_root, _, _, _, _, context = prepared(tmp_path)
    unit = build_schedule(load_contract()[0])[0]
    claim = execution.publish_unit_claim(global_root, local, unit, context, NOW)
    path = local / "orphan.json"

    def write(confirmation, authorized_at):
        payload = {
            "version": execution.ORPHAN_FINALIZATION_VERSION,
            "contract_lineage": execution.CONTRACT_LINEAGE,
            "authorization_id": context.authorization_id,
            "candidate_digest": context.candidate_digest,
            "approval_digest": context.approval_digest,
            "unit_id": unit.unit_id,
            "orphan_claim_digest": claim["claim_digest"],
            "owner_echoed_orphan_claim_digest_out_of_band": True,
            "owner_confirmed_no_process_remains": True,
            "confirmation_process": confirmation,
            "owner_identity": context.owner_identity,
            "authorized_at": authorized_at,
            "operational_process_evidence_only": True,
        }
        if path.exists():
            path.unlink()
        execution.write_test_private_record(
            path,
            {
                "finalization_authorization": payload,
                "finalization_authorization_digest": execution.sha256_canonical(
                    payload
                ),
            },
        )

    write("checked", NOW)
    with pytest.raises(execution.ExecutionFailure):
        execution.finalize_orphan_unit_offline(
            global_root, local, unit, context, path, NOW
        )
    write(
        "Owner inspected the machine-global lock and process table out of band.",
        "2026-08-04T11:59:59.000000Z",
    )
    with pytest.raises(execution.ExecutionFailure):
        execution.finalize_orphan_unit_offline(
            global_root, local, unit, context, path, NOW
        )
