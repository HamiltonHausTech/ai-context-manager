import hashlib
import json
import logging
import os
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.adaptive_selection import context_sensitivity_execution as execution
from experiments.adaptive_selection.context_sensitivity_calibration import (
    EXECUTION_ORDER,
    canonical_bytes,
    load_contract,
    render_requests,
)

ROOT = Path(__file__).parents[2]
REVISION = "a" * 40
HOST = "sha256:" + "1" * 64
ACCOUNT = "sha256:" + "2" * 64
CREDENTIAL = "sha256:" + "3" * 64
NOW = "2026-08-04T12:00:00.000000Z"
REQUEST_HASHES = (
    "sha256:07d3a42b2267a187ba1234b80c22bd221a355f9905df51235c1b03f3e226462e",
    "sha256:a7b50fbffbbbaaa876852fe7f791b1568d1613a2919b3b3d136710d72be95c1d",
    "sha256:e2643c1988e3e7acb3f6fbeedee33c59db76ce3a3fd0f4ff0791a3c466dc1e19",
    "sha256:8f9e3584563e914d2110bd400bba9972c4a4e6419f86fc6897b3505dfc6bb91e",
    "sha256:a9ff62775f3b7026d69167a72ac27339fa1730233caa72c67ad879bff39b15fe",
    "sha256:490f667e064d0909bcef38cb7cf60e445eeceeec1a255ef462f5106c1336588b",
    "sha256:7be7ff6a80d4cdde19ae34ff052d26b9697e4cda29cd8f381ab98131d2368b35",
    "sha256:3b0f34553754aeb875c1cdf24d630ec5416e4ba84eee919f31d1c1673947fe8e",
    "sha256:fed0326309b2a3522c05b1d5baee1b4d0bfed5f62211bbfc08eda6b311c8f8c4",
)
VALID_STRUCTURED = {
    "diagnosis": "Bounded diagnosis.",
    "supporting_evidence_numbers": [1],
    "missing_evidence": [],
    "confidence": "medium",
    "next_safe_actions": ["Inspect the bounded state."],
    "actions_to_avoid": ["Do not make an irreversible change."],
}
SUCCESS_USAGE = {
    "input_tokens": 10,
    "cached_input_tokens": 2,
    "cache_write_input_tokens": 1,
    "output_tokens": 3,
    "total_tokens": 13,
}
SUCCESS_ACTUAL_COST = "0.0000529"
SUCCESS_UPPER_COST = "0.000061"


def observed_metadata(index=0, content_type="application/problem+json"):
    return {
        "http_status": 200,
        "content_type": content_type,
        "provider_request_id": f"req_observed_{index}",
        "response_id": f"resp_observed_{index}",
        "observed_model": execution.PINNED_MODEL,
    }


def success_raw(index=0, structured=None, usage=None, output_items=None):
    structured = VALID_STRUCTURED if structured is None else structured
    usage = SUCCESS_USAGE if usage is None else usage
    if output_items is None:
        text = json.dumps(structured, separators=(",", ":"))
        output_items = [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ]
    document = {
        "id": f"resp_observed_{index}",
        "model": execution.PINNED_MODEL,
        "status": "completed",
        "incomplete_details": None,
        "output": output_items,
        "usage": {
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "total_tokens": usage["total_tokens"],
            "input_tokens_details": {
                "cached_tokens": usage["cached_input_tokens"],
                "cache_write_tokens": usage["cache_write_input_tokens"],
            },
        },
    }
    return json.dumps(document, separators=(",", ":")).encode()


def private_root(tmp_path, name):
    root = tmp_path / name
    root.mkdir(mode=0o700, parents=True)
    os.chmod(root, 0o700)
    return root


def prepare(tmp_path, token_byte=7):
    local = private_root(tmp_path, "local")
    global_root = private_root(tmp_path, "global")
    candidate_path = local / "authorization-candidate.json"
    mapping_path = local / "blind-mapping.json"
    candidate = execution.prepare_non_authorizing_candidate(
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
        token_bytes=lambda count: bytes([token_byte]) * count,
        shuffle=lambda values: values.reverse(),
    )
    approval_path = local / "owner-approval.json"
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
    return (
        local,
        global_root,
        candidate_path,
        mapping_path,
        approval_path,
        candidate,
        approval,
    )


def authority_context(candidate, approval):
    c = candidate["candidate"]
    return execution.AuthorityContext(
        authorization_id=c["authorization_id"],
        candidate_digest=candidate["candidate_sha256"],
        approval_digest=approval["approval_digest"],
        code_revision=REVISION,
        host_fingerprint=HOST,
        account_fingerprint=ACCOUNT,
        credential_fingerprint=CREDENTIAL,
        blind_mapping_commitment=c["blind_mapping_commitment"],
        authorization_nonce=c["nonce"],
        owner_identity=c["owner_identity"],
        issued_at=c["issued_at"],
        approved_at=approval["approval"]["approved_at"],
        expires_at=c["expires_at"],
    )


def test_manifest_freezes_exact_new_ceiling_prices_versions_and_zero_authority():
    manifest, raw = execution.load_readiness_manifest(ROOT / execution.READINESS_PATH)
    assert hashlib.sha256(raw).hexdigest() == execution.PINNED_READINESS_MANIFEST_SHA256
    assert manifest["contract_lineage"]["identity"] == execution.CONTRACT_LINEAGE
    assert manifest["budget"] == {
        "conservative_input_token_bound": 29192,
        "conservative_output_token_bound": 18432,
        "currency": "USD",
        "short_context_rates_per_million": {
            "input": "2.00",
            "cached_input": "0.20",
            "cache_write_input": "2.50",
            "output": "12.00",
        },
        "conservative_execution_ceiling": "0.294164",
        "owner_cap": "1.00",
        "cap_semantics": "enforced projection/accepted-usage cap; not a guarantee against provider billing anomalies or unrelated activity",
        "pricing_frozen_on": "2026-08-04",
        "official_pricing_source_url": "https://developers.openai.com/api/docs/pricing",
        "historical_projection": {
            "amount": "0.279568",
            "semantics": "reproducibility-only; not a maximum or current execution ceiling",
        },
    }
    assert manifest["provider_configuration"]["openai_version"] == "2.46.0"
    assert manifest["provider_configuration"]["httpx_version"] == "0.28.1"
    assert manifest["execution"]["network_requests_authorized_by_manifest"] == 0
    assert raw.decode().count("0.279568") == 1


def test_candidate_schema_is_exact_and_candidate_alone_has_zero_authority(tmp_path):
    local, _, candidate_path, mapping_path, approval_path, candidate, _ = prepare(
        tmp_path
    )
    fields = set(candidate["candidate"])
    assert fields == execution.CANDIDATE_FIELDS
    assert candidate["candidate"]["candidate_grants_network_authority"] is False
    assert candidate["candidate"]["prices"] == execution.PINNED_PRICES
    assert candidate["candidate"]["conservative_execution_ceiling"] == "0.294164"
    assert candidate["candidate"]["owner_cap"] == "1.00"
    assert candidate["candidate"]["httpx_version"] == "0.28.1"
    assert (
        execution.verify_candidate(candidate_path, REVISION)["candidate_sha256"]
        == candidate["candidate_sha256"]
    )
    approval_path.unlink()
    with pytest.raises(execution.ExecutionFailure):
        execution.verify_authority(
            candidate_path,
            approval_path,
            mapping_path,
            REVISION,
            HOST,
            ACCOUNT,
            CREDENTIAL,
            NOW,
        )
    assert not any(path.name.endswith(".claim.json") for path in local.rglob("*"))


def test_separate_owner_approval_must_exactly_echo_candidate_and_has_unique_digest(
    tmp_path,
):
    _, _, candidate_path, mapping_path, approval_path, candidate, _ = prepare(tmp_path)
    value = json.loads(approval_path.read_bytes())
    value["approval"]["candidate_sha256"] = "sha256:" + "0" * 64
    value["approval_digest"] = execution.sha256_canonical(value["approval"])
    approval_path.unlink()
    execution.write_test_private_record(approval_path, value)
    with pytest.raises(execution.ExecutionFailure):
        execution.verify_authority(
            candidate_path,
            approval_path,
            mapping_path,
            REVISION,
            HOST,
            ACCOUNT,
            CREDENTIAL,
            NOW,
        )
    assert candidate["candidate"]["owner_identity"] == "repository-owner"


def test_mapping_is_private_precommitted_random_and_blind_order_differs(tmp_path):
    _, _, candidate_path, mapping_path, _, candidate, _ = prepare(tmp_path)
    mapping = execution.verify_mapping(mapping_path, candidate["candidate"])
    entries = mapping["mapping"]["entries"]
    assert [item["canonical_cell_id"] for item in entries] == list(EXECUTION_ORDER)
    assert len({item["assessment_id"] for item in entries}) == 9
    assert [
        item["canonical_cell_id"] for item in mapping["mapping"]["blind_order"]
    ] != list(EXECUTION_ORDER)
    assert (
        execution.sha256_canonical(mapping["mapping"])
        == candidate["candidate"]["blind_mapping_commitment"]
    )
    assert candidate["candidate"]["candidate_grants_network_authority"] is False


def test_private_primitives_reject_ancestor_and_final_symlink_hardlink_mode_and_rehash(
    tmp_path,
):
    good = private_root(tmp_path, "good")
    path = good / "record.json"
    execution.write_test_private_record(path, {"x": 1})
    execution.read_private_record(path)
    os.chmod(path, 0o644)
    with pytest.raises(execution.ExecutionFailure):
        execution.read_private_record(path)
    os.chmod(path, 0o600)
    hard = good / "hard.json"
    os.link(path, hard)
    with pytest.raises(execution.ExecutionFailure):
        execution.read_private_record(path)
    hard.unlink()
    final_link = good / "final-link.json"
    final_link.symlink_to(path)
    with pytest.raises(execution.ExecutionFailure):
        execution.read_private_record(final_link)
    actual = private_root(tmp_path, "actual")
    ancestor = tmp_path / "ancestor"
    ancestor.symlink_to(actual, target_is_directory=True)
    with pytest.raises(execution.ExecutionFailure):
        execution.write_test_private_record(ancestor / "bad.json", {"x": 1})
    payload = {"claim": {"authority_consumed": True}}
    envelope = {
        "claim": payload["claim"],
        "claim_digest": execution.sha256_canonical(payload["claim"]),
    }
    clean = private_root(tmp_path, "clean") / "rehashed.json"
    execution.write_test_private_record(clean, envelope)
    with pytest.raises(execution.ExecutionFailure):
        execution.verify_claim_file(
            clean,
            "cell-k4m2",
            REQUEST_HASHES[0],
            authority_context(*prepare(tmp_path / "other")[-2:]),
        )


def test_claim_semantics_bind_lineage_authority_approval_and_never_claim_dispatch(
    tmp_path,
):
    local, global_root, _, _, _, candidate, approval = prepare(tmp_path)
    context = authority_context(candidate, approval)
    claim = execution.publish_authority_claim(
        global_root, local, "cell-k4m2", REQUEST_HASHES[0], context, NOW
    )
    payload = claim["claim"]
    assert payload["authority_consumed"] is True
    assert "request_attempted" not in payload
    assert "dispatch_invoked" not in payload
    for field in (
        "contract_lineage",
        "authorization_id",
        "candidate_digest",
        "approval_digest",
        "code_revision",
        "host_fingerprint",
        "account_fingerprint",
        "credential_fingerprint",
    ):
        assert field in payload
    assert (global_root / "claims" / "cell-k4m2.claim.json").exists()
    assert (local / "mirrors" / "claims" / "cell-k4m2.claim.json").exists()


def test_state_machine_orphan_blocks_without_mutation_and_conflicts_abort(tmp_path):
    local, global_root, _, _, _, candidate, approval = prepare(tmp_path)
    context = authority_context(candidate, approval)
    assert (
        execution.classify_state(
            global_root, local, "cell-k4m2", REQUEST_HASHES[0], context
        )
        == "pending"
    )
    execution.publish_authority_claim(
        global_root, local, "cell-k4m2", REQUEST_HASHES[0], context, NOW
    )
    before = sorted(
        (str(p.relative_to(tmp_path)), p.stat().st_mtime_ns)
        for p in tmp_path.rglob("*")
        if p.is_file()
    )
    assert (
        execution.classify_state(
            global_root, local, "cell-k4m2", REQUEST_HASHES[0], context
        )
        == "blocked_orphan_claim"
    )
    after = sorted(
        (str(p.relative_to(tmp_path)), p.stat().st_mtime_ns)
        for p in tmp_path.rglob("*")
        if p.is_file()
    )
    assert before == after
    rogue = local / "mirrors" / "claims" / "cell-y9h4.claim.json"
    rogue.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    execution.write_test_private_record(rogue, {"x": 1})
    with pytest.raises(execution.ExecutionFailure):
        execution.classify_state(
            global_root, local, "cell-y9h4", REQUEST_HASHES[1], context
        )


def test_global_terminal_is_authoritative_raw_once_and_local_is_mirror(tmp_path):
    local, global_root, _, _, _, candidate, approval = prepare(tmp_path)
    context = authority_context(candidate, approval)
    claim = execution.publish_authority_claim(
        global_root, local, "cell-k4m2", REQUEST_HASHES[0], context, NOW
    )
    terminal = execution.publish_terminal(
        global_root,
        local,
        "cell-k4m2",
        REQUEST_HASHES[0],
        context,
        kind="success",
        dispatch_invoked=True,
        server_acceptance="yes",
        provider_visible_evidence={
            "task": "task",
            "timestamped_evidence": ["1. [time] evidence"],
        },
        structured_response=VALID_STRUCTURED,
        raw_bytes=success_raw(0),
        recorded_at=NOW,
        usage=SUCCESS_USAGE,
        actual_cost=SUCCESS_ACTUAL_COST,
        conservative_cost_upper_bound=SUCCESS_UPPER_COST,
        response_metadata=observed_metadata(),
    )
    assert terminal["terminal"]["authority_consumed"] is True
    assert terminal["terminal"]["claim_digest"] == claim["claim_digest"]
    assert terminal["terminal"]["dispatch_invoked"] is True
    assert terminal["terminal"]["server_acceptance"] == "yes"
    assert len(list((global_root / "raw").iterdir())) == 1
    assert (
        execution.classify_state(
            global_root, local, "cell-k4m2", REQUEST_HASHES[0], context
        )
        == "terminal"
    )
    assert (global_root / "terminal-index" / "cell-k4m2.json").exists()


def test_terminal_without_claim_raw_only_and_semantic_mismatch_abort(tmp_path):
    local, global_root, _, _, _, candidate, approval = prepare(tmp_path)
    context = authority_context(candidate, approval)
    raw_dir = global_root / "raw"
    raw_dir.mkdir(mode=0o700)
    execution.write_test_private_record(raw_dir / "cell-k4m2.raw", {"x": 1})
    with pytest.raises(execution.ExecutionFailure):
        execution.classify_state(
            global_root, local, "cell-k4m2", REQUEST_HASHES[0], context
        )


def test_orphan_finalization_requires_external_exact_claim_digest_and_owner_process_proof(
    tmp_path,
):
    local, global_root, _, _, _, candidate, approval = prepare(tmp_path)
    context = authority_context(candidate, approval)
    claim = execution.publish_authority_claim(
        global_root, local, "cell-k4m2", REQUEST_HASHES[0], context, NOW
    )
    recovery_path = local / "owner-orphan-finalization.json"
    recovery_payload = {
        "version": execution.ORPHAN_FINALIZATION_VERSION,
        "authorization_id": context.authorization_id,
        "candidate_digest": context.candidate_digest,
        "approval_digest": context.approval_digest,
        "contract_lineage": execution.CONTRACT_LINEAGE,
        "cell_id": "cell-k4m2",
        "request_hash": REQUEST_HASHES[0],
        "orphan_claim_digest": claim["claim_digest"],
        "owner_echoed_orphan_claim_digest_out_of_band": True,
        "owner_confirmed_no_process_remains": True,
        "confirmation_process": "Owner inspected the machine-global run lock and process table out of band.",
        "owner_identity": "repository-owner",
        "authorized_at": NOW,
        "operational_process_evidence_only": True,
    }
    recovery = {
        "finalization_authorization": recovery_payload,
        "finalization_authorization_digest": execution.sha256_canonical(
            recovery_payload
        ),
    }
    execution.write_test_private_record(recovery_path, recovery)
    terminal = execution.finalize_orphan_offline(
        global_root, local, "cell-k4m2", REQUEST_HASHES[0], context, recovery_path, NOW
    )
    assert terminal["terminal"]["kind"] == "invalid_ambiguous"
    assert terminal["terminal"]["dispatch_invoked"] == "unknown"
    assert terminal["terminal"]["server_acceptance"] == "unknown"
    assert (
        execution.classify_state(
            global_root, local, "cell-k4m2", REQUEST_HASHES[0], context
        )
        == "terminal"
    )


def test_machine_global_nonblocking_lock_allows_one_cross_checkout_winner(tmp_path):
    local1, global_root, _, _, _, candidate, approval = prepare(tmp_path / "one", 8)
    local2 = private_root(tmp_path, "two-local")
    context = authority_context(candidate, approval)
    barrier = __import__("threading").Barrier(2)

    def contender(local):
        barrier.wait(timeout=2)
        try:
            with execution.machine_global_run_lock(global_root):
                execution.publish_authority_claim(
                    global_root,
                    local,
                    "cell-k4m2",
                    REQUEST_HASHES[0],
                    context,
                    NOW,
                )
                return "winner"
        except (execution.ExecutionFailure, FileExistsError):
            return "loser"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(contender, (local1, local2)))
    assert outcomes.count("winner") == 1
    assert (
        execution.verify_claim_file(
            global_root / "claims" / "cell-k4m2.claim.json",
            "cell-k4m2",
            REQUEST_HASHES[0],
            context,
        )["claim"]["authority_consumed"]
        is True
    )


def _publish_nine_successes(local, global_root, context):
    _, visible_evidence = execution._validate_requests(ROOT)
    for index, (cell_id, request_hash) in enumerate(
        zip(EXECUTION_ORDER, REQUEST_HASHES)
    ):
        execution.publish_authority_claim(
            global_root, local, cell_id, request_hash, context, NOW
        )
        execution.publish_terminal(
            global_root,
            local,
            cell_id,
            request_hash,
            context,
            kind="success",
            dispatch_invoked=True,
            server_acceptance="yes",
            provider_visible_evidence=visible_evidence[index],
            structured_response=VALID_STRUCTURED,
            raw_bytes=success_raw(index),
            recorded_at=NOW,
            usage=SUCCESS_USAGE,
            actual_cost=SUCCESS_ACTUAL_COST,
            conservative_cost_upper_bound=SUCCESS_UPPER_COST,
            response_metadata=observed_metadata(index),
        )


def test_blind_export_uses_random_ids_order_full_assessment_material_and_no_leakage(
    tmp_path,
):
    local, global_root, _, mapping_path, _, candidate, approval = prepare(tmp_path)
    context = authority_context(candidate, approval)
    _publish_nine_successes(local, global_root, context)
    export = execution.build_blind_assessment(
        ROOT, global_root, local, mapping_path, context
    )
    assert export["assessment_ready"] is True
    assert len(export["assessments"]) == 9
    assert [x["assessment_id"] for x in export["assessments"]] != list(EXECUTION_ORDER)
    required = {
        "assessment_id",
        "task",
        "provider_visible_timestamped_evidence",
        "structured_response",
        "status",
        "criteria",
        "adjudication_rules",
        "critical_findings",
    }
    assert all(set(item) == required for item in export["assessments"])

    def nested_keys(value):
        if isinstance(value, dict):
            return set(value).union(
                *(nested_keys(item) for item in value.values()), set()
            )
        if isinstance(value, list):
            return set().union(*(nested_keys(item) for item in value), set())
        return set()

    keys = nested_keys(export)
    for forbidden_key in {
        "condition",
        "family",
        "scenario_id",
        "anchor",
        "source_role",
        "provenance",
        "request_hash",
        "code_revision",
        "provider_request_id",
        "canonical_cell_id",
        "execution_position",
    }:
        assert forbidden_key not in keys
    folded = json.dumps(export).casefold()
    for forbidden_value in list(EXECUTION_ORDER) + ["/raw/"]:
        assert forbidden_value.casefold() not in folded
    _, visible_evidence = execution._validate_requests(ROOT)
    mapping = execution.verify_mapping(
        mapping_path,
        {
            "authorization_id": context.authorization_id,
            "nonce": context.authorization_nonce,
            "blind_mapping_commitment": context.blind_mapping_commitment,
        },
    )["mapping"]
    by_assessment = {item["assessment_id"]: item for item in export["assessments"]}
    for item in mapping["blind_order"]:
        index = EXECUTION_ORDER.index(item["canonical_cell_id"])
        assessment = by_assessment[item["assessment_id"]]
        assert assessment["task"] == visible_evidence[index]["task"]
        assert (
            assessment["provider_visible_timestamped_evidence"]
            == visible_evidence[index]["timestamped_evidence"]
        )


def test_blind_export_failure_is_not_ready_and_has_no_score_or_verdict(tmp_path):
    local, global_root, _, mapping_path, _, candidate, approval = prepare(tmp_path)
    context = authority_context(candidate, approval)
    _, visible_evidence = execution._validate_requests(ROOT)
    for index, (cell_id, request_hash) in enumerate(
        zip(EXECUTION_ORDER, REQUEST_HASHES)
    ):
        execution.publish_authority_claim(
            global_root, local, cell_id, request_hash, context, NOW
        )
        execution.publish_terminal(
            global_root,
            local,
            cell_id,
            request_hash,
            context,
            kind="failure" if index == 0 else "success",
            dispatch_invoked=True,
            server_acceptance="unknown" if index == 0 else "yes",
            provider_visible_evidence=visible_evidence[index],
            structured_response=None if index == 0 else VALID_STRUCTURED,
            raw_bytes=None if index == 0 else success_raw(index),
            failure_category="transport_error" if index == 0 else None,
            recorded_at=NOW,
            usage=None if index == 0 else SUCCESS_USAGE,
            actual_cost=None if index == 0 else SUCCESS_ACTUAL_COST,
            conservative_cost_upper_bound=(
                "0.00001" if index == 0 else SUCCESS_UPPER_COST
            ),
            response_metadata=None if index == 0 else observed_metadata(index),
        )
    export = execution.build_blind_assessment(
        ROOT, global_root, local, mapping_path, context
    )
    assert export["assessment_ready"] is False
    assert "score" not in export and "verdict" not in export


def test_default_summary_is_offline_non_authorizing_and_uses_new_ceiling():
    summary = execution.build_dry_run_summary(ROOT)
    assert summary["network_requests_authorized"] == 0
    assert summary["conservative_execution_ceiling"] == "0.294164"
    assert summary["candidate_grants_network_authority"] is False


def _rewrite_private(path, value):
    path.unlink()
    execution.write_test_private_record(path, value)


@pytest.mark.parametrize(
    ("field", "mutation"),
    [
        ("version", "alternate"),
        ("authorization_id", "task12b-auth-" + "0" * 31),
        ("nonce", "0" * 47),
        ("issued_at", "2026-08-04T12:00:00Z"),
        ("expires_at", "2026-08-04T14:00:00.000000Z"),
        ("maximum_execution_window_seconds", True),
        ("contract_lineage", "alternate"),
        ("contract_hash", "sha256:" + "0" * 64),
        ("readiness_manifest_hash", "sha256:" + "0" * 64),
        ("execution_code_revision", "b" * 40),
        ("ordered_request_hashes", list(reversed(REQUEST_HASHES))),
        ("model", "alternate"),
        ("openai_version", "2.45.0"),
        ("httpx_version", "0.28.0"),
        (
            "provider_settings",
            {
                "reasoning_effort": "medium",
                "max_output_tokens": 2048,
                "max_retries": 0,
                "store": False,
                "stream": False,
                "timeout_seconds": "30.0",
                "tools": [],
            },
        ),
        ("prices", {**execution.PINNED_PRICES, "input_per_million": 2.0}),
        ("conservative_execution_ceiling", "0.294165"),
        ("owner_cap", "1.01"),
        ("no_retry_policy", "alternate"),
        ("approved_cell_count", True),
        ("scope", "alternate"),
        ("host_fingerprint", "sha256:" + "0" * 63),
        ("account_fingerprint", "sha256:" + "0" * 63),
        ("credential_fingerprint", "sha256:" + "0" * 63),
        ("blind_mapping_commitment", "sha256:" + "0" * 63),
        ("owner_identity", " repository-owner"),
        ("candidate_grants_network_authority", True),
    ],
)
def test_candidate_rejects_every_rehashed_security_or_scientific_mutation(
    tmp_path, field, mutation
):
    _, _, candidate_path, _, _, _, _ = prepare(tmp_path)
    envelope = json.loads(candidate_path.read_bytes())
    envelope["candidate"][field] = mutation
    envelope["candidate_sha256"] = execution.sha256_canonical(envelope["candidate"])
    _rewrite_private(candidate_path, envelope)
    with pytest.raises(execution.ExecutionFailure):
        execution.verify_candidate(candidate_path, REVISION)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m: m.update(version="alternate"),
        lambda m: m.update(nonce="0" * 48),
        lambda m: m.update(contract_lineage="alternate"),
        lambda m: m.update(private_until_annotations_locked=False),
        lambda m: m.update(extra=True),
        lambda m: m["entries"][0].update(extra=True),
        lambda m: m["entries"][0].update(
            assessment_id=m["entries"][1]["assessment_id"]
        ),
        lambda m: m["entries"][0].update(canonical_cell_id="cell-alternate"),
        lambda m: m.update(blind_order=list(m["entries"])),
    ],
)
def test_rehashed_alternate_mapping_cannot_verify_or_export(tmp_path, mutate):
    local, global_root, _, mapping_path, _, candidate, approval = prepare(tmp_path)
    context = authority_context(candidate, approval)
    envelope = json.loads(mapping_path.read_bytes())
    mutate(envelope["mapping"])
    envelope["mapping_digest"] = execution.sha256_canonical(envelope["mapping"])
    _rewrite_private(mapping_path, envelope)
    with pytest.raises(execution.ExecutionFailure):
        execution.verify_mapping(mapping_path, candidate["candidate"])
    with pytest.raises(execution.ExecutionFailure):
        execution.build_blind_assessment(
            ROOT, global_root, local, mapping_path, context
        )


def test_manifest_has_official_pricing_bounds_and_reproducibility_label():
    manifest, _ = execution.load_readiness_manifest(ROOT / execution.READINESS_PATH)
    assert manifest["budget"]["official_pricing_source_url"] == (
        "https://developers.openai.com/api/docs/pricing"
    )
    assert manifest["budget"]["conservative_input_token_bound"] == 29192
    assert manifest["budget"]["conservative_output_token_bound"] == 18432
    assert manifest["budget"]["historical_projection"]["semantics"] == (
        "reproducibility-only; not a maximum or current execution ceiling"
    )


def test_contract_and_request_bytes_regenerate_exact_frozen_hashes():
    contract_path = ROOT / execution.CONTRACT_PATH
    assert hashlib.sha256(contract_path.read_bytes()).hexdigest() == (
        "0bf61722680aca83432f8f82d29b9d309673efbf2e750720682fa2ff4b7b16d1"
    )
    contract, _ = load_contract(contract_path)
    assert (
        tuple(
            "sha256:" + hashlib.sha256(canonical_bytes(body)).hexdigest()
            for body in render_requests(contract)
        )
        == REQUEST_HASHES
    )


def test_private_directory_wrong_mode_and_owner_are_rejected(tmp_path, monkeypatch):
    root = private_root(tmp_path, "private")
    os.chmod(root, 0o755)
    with pytest.raises(execution.ExecutionFailure):
        execution.write_test_private_record(root / "x.json", {"x": 1})
    os.chmod(root, 0o700)
    real_lstat = Path.lstat

    def wrong_owner(path):
        result = real_lstat(path)
        if path == root:
            values = list(result)
            values[4] = result.st_uid + 1
            return os.stat_result(values)
        return result

    monkeypatch.setattr(Path, "lstat", wrong_owner)
    with pytest.raises(execution.ExecutionFailure):
        execution.write_test_private_record(root / "y.json", {"x": 1})


def test_private_record_existing_destination_is_never_overwritten(tmp_path):
    root = private_root(tmp_path, "private")
    path = root / "record.json"
    execution.write_test_private_record(path, {"winner": 1})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        execution.write_test_private_record(path, {"loser": 2})
    assert path.read_bytes() == before


def test_failure_with_http_raw_and_transport_failure_without_raw_are_verifiable(
    tmp_path,
):
    local, global_root, _, _, _, candidate, approval = prepare(tmp_path)
    context = authority_context(candidate, approval)
    usage = {
        "input_tokens": 10,
        "cached_input_tokens": 2,
        "cache_write_input_tokens": 1,
        "output_tokens": 3,
        "total_tokens": 13,
    }
    execution.publish_authority_claim(
        global_root, local, EXECUTION_ORDER[0], REQUEST_HASHES[0], context, NOW
    )
    http_failure = execution.publish_terminal(
        global_root,
        local,
        EXECUTION_ORDER[0],
        REQUEST_HASHES[0],
        context,
        kind="failure",
        dispatch_invoked=True,
        server_acceptance="yes",
        provider_visible_evidence=None,
        structured_response=None,
        raw_bytes=b'{"error":{"type":"rate_limit"}}',
        recorded_at=NOW,
        failure_category="http_error",
        usage=usage,
        actual_cost="0.0000529",
        conservative_cost_upper_bound="0.000061",
        response_metadata={
            "http_status": 429,
            "content_type": "application/json",
            "provider_request_id": "req_fixed",
            "response_id": None,
            "observed_model": None,
        },
    )
    assert http_failure["terminal"]["raw_response_sha256"] is not None
    assert http_failure["terminal"]["failure_category"] == "http_error"
    assert http_failure["terminal"]["usage"] == usage
    execution.verify_terminal(
        global_root, EXECUTION_ORDER[0], REQUEST_HASHES[0], context
    )

    execution.publish_authority_claim(
        global_root, local, EXECUTION_ORDER[1], REQUEST_HASHES[1], context, NOW
    )
    transport_failure = execution.publish_terminal(
        global_root,
        local,
        EXECUTION_ORDER[1],
        REQUEST_HASHES[1],
        context,
        kind="failure",
        dispatch_invoked=True,
        server_acceptance="no",
        provider_visible_evidence=None,
        structured_response=None,
        raw_bytes=None,
        recorded_at=NOW,
        failure_category="transport_error",
        conservative_cost_upper_bound="0.00001",
    )
    assert transport_failure["terminal"]["raw_response_sha256"] is None
    assert transport_failure["terminal"]["response_metadata"] is None
    assert "exception" not in json.dumps(transport_failure).casefold()


def test_rehashed_terminal_schema_tampering_is_rejected(tmp_path):
    local, global_root, _, _, _, candidate, approval = prepare(tmp_path)
    context = authority_context(candidate, approval)
    execution.publish_authority_claim(
        global_root, local, EXECUTION_ORDER[0], REQUEST_HASHES[0], context, NOW
    )
    execution.publish_terminal(
        global_root,
        local,
        EXECUTION_ORDER[0],
        REQUEST_HASHES[0],
        context,
        kind="failure",
        dispatch_invoked=True,
        server_acceptance="no",
        provider_visible_evidence=None,
        structured_response=None,
        raw_bytes=None,
        recorded_at=NOW,
        failure_category="transport_error",
        conservative_cost_upper_bound="0.00001",
    )
    terminal_path = global_root / "terminals" / f"{EXECUTION_ORDER[0]}.terminal.json"
    index_path = global_root / "terminal-index" / f"{EXECUTION_ORDER[0]}.json"
    value = json.loads(terminal_path.read_bytes())
    value["terminal"]["failure_category"] = "provider said secret text"
    value["terminal_digest"] = execution.sha256_canonical(value["terminal"])
    index = json.loads(index_path.read_bytes())
    index["terminal_index"]["terminal_digest"] = value["terminal_digest"]
    index["terminal_index_digest"] = execution.sha256_canonical(index["terminal_index"])
    _rewrite_private(terminal_path, value)
    _rewrite_private(index_path, index)
    with pytest.raises(execution.ExecutionFailure):
        execution.verify_terminal(
            global_root, EXECUTION_ORDER[0], REQUEST_HASHES[0], context
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda terminal: terminal.update(usage=None, actual_cost=None),
        lambda terminal: terminal.update(
            structured_response={"diagnosis": "incomplete"}
        ),
        lambda terminal: terminal["response_metadata"].update(observed_model="other"),
        lambda terminal: terminal.update(conservative_cost_upper_bound="0.999"),
    ],
)
def test_rehashed_success_evidence_and_cost_tampering_is_rejected(tmp_path, mutation):
    local, global_root, _, _, _, candidate, approval = prepare(tmp_path)
    context = authority_context(candidate, approval)
    cell_id = EXECUTION_ORDER[0]
    execution.publish_authority_claim(
        global_root, local, cell_id, REQUEST_HASHES[0], context, NOW
    )
    execution.publish_terminal(
        global_root,
        local,
        cell_id,
        REQUEST_HASHES[0],
        context,
        kind="success",
        dispatch_invoked=True,
        server_acceptance="yes",
        provider_visible_evidence={"task": "task", "timestamped_evidence": []},
        structured_response=VALID_STRUCTURED,
        raw_bytes=success_raw(0),
        recorded_at=NOW,
        usage=SUCCESS_USAGE,
        actual_cost=SUCCESS_ACTUAL_COST,
        conservative_cost_upper_bound=SUCCESS_UPPER_COST,
        response_metadata=observed_metadata(),
    )
    terminal_path = global_root / "terminals" / f"{cell_id}.terminal.json"
    index_path = global_root / "terminal-index" / f"{cell_id}.json"
    value = json.loads(terminal_path.read_bytes())
    mutation(value["terminal"])
    value["terminal_digest"] = execution.sha256_canonical(value["terminal"])
    index = json.loads(index_path.read_bytes())
    index["terminal_index"]["terminal_digest"] = value["terminal_digest"]
    index["terminal_index_digest"] = execution.sha256_canonical(index["terminal_index"])
    _rewrite_private(terminal_path, value)
    _rewrite_private(index_path, index)
    with pytest.raises(execution.ExecutionFailure):
        execution.verify_terminal(global_root, cell_id, REQUEST_HASHES[0], context)


@pytest.mark.parametrize(
    "raw_mutation",
    ["non_json", "failed_status", "altered_structure", "altered_usage"],
)
def test_rehashed_raw_semantic_tampering_is_rejected(tmp_path, raw_mutation):
    local, global_root, _, _, _, candidate, approval = prepare(tmp_path)
    context = authority_context(candidate, approval)
    cell_id = EXECUTION_ORDER[0]
    _, visible_evidence = execution._validate_requests(ROOT)
    execution.publish_authority_claim(
        global_root, local, cell_id, REQUEST_HASHES[0], context, NOW
    )
    execution.publish_terminal(
        global_root,
        local,
        cell_id,
        REQUEST_HASHES[0],
        context,
        kind="success",
        dispatch_invoked=True,
        server_acceptance="yes",
        provider_visible_evidence=visible_evidence[0],
        structured_response=VALID_STRUCTURED,
        raw_bytes=success_raw(0),
        recorded_at=NOW,
        usage=SUCCESS_USAGE,
        actual_cost=SUCCESS_ACTUAL_COST,
        conservative_cost_upper_bound=SUCCESS_UPPER_COST,
        response_metadata=observed_metadata(),
    )
    if raw_mutation == "non_json":
        changed = b"NOT JSON AND NOT A PROVIDER RESPONSE"
    else:
        document = json.loads(success_raw(0))
        if raw_mutation == "failed_status":
            document["status"] = "failed"
            document["model"] = "attacker-model"
        elif raw_mutation == "altered_structure":
            altered = dict(VALID_STRUCTURED, diagnosis="Different diagnosis.")
            document["output"][0]["content"][0]["text"] = json.dumps(
                altered, separators=(",", ":")
            )
        else:
            document["usage"]["input_tokens"] = 11
            document["usage"]["total_tokens"] = 14
        changed = json.dumps(document, separators=(",", ":")).encode()
    raw_path = global_root / "raw" / f"{cell_id}.raw"
    raw_path.unlink()
    execution._write_private_bytes_no_overwrite(raw_path, changed)
    terminal_path = global_root / "terminals" / f"{cell_id}.terminal.json"
    index_path = global_root / "terminal-index" / f"{cell_id}.json"
    value = json.loads(terminal_path.read_bytes())
    value["terminal"]["raw_response_sha256"] = execution._sha256_bytes(changed)
    value["terminal_digest"] = execution.sha256_canonical(value["terminal"])
    index = json.loads(index_path.read_bytes())
    index["terminal_index"]["terminal_digest"] = value["terminal_digest"]
    index["terminal_index_digest"] = execution.sha256_canonical(index["terminal_index"])
    _rewrite_private(terminal_path, value)
    _rewrite_private(index_path, index)
    with pytest.raises(execution.ExecutionFailure):
        execution.verify_terminal(
            global_root,
            cell_id,
            REQUEST_HASHES[0],
            context,
            visible_evidence[0],
        )


def test_rehashed_provider_evidence_leakage_is_rejected_before_blind_export(tmp_path):
    local, global_root, _, mapping_path, _, candidate, approval = prepare(tmp_path)
    context = authority_context(candidate, approval)
    _publish_nine_successes(local, global_root, context)
    cell_id = EXECUTION_ORDER[0]
    terminal_path = global_root / "terminals" / f"{cell_id}.terminal.json"
    index_path = global_root / "terminal-index" / f"{cell_id}.json"
    value = json.loads(terminal_path.read_bytes())
    value["terminal"]["provider_visible_evidence"]["condition"] = "correct"
    value["terminal"]["provider_visible_evidence"]["canonical_cell_id"] = cell_id
    value["terminal_digest"] = execution.sha256_canonical(value["terminal"])
    index = json.loads(index_path.read_bytes())
    index["terminal_index"]["terminal_digest"] = value["terminal_digest"]
    index["terminal_index_digest"] = execution.sha256_canonical(index["terminal_index"])
    _rewrite_private(terminal_path, value)
    _rewrite_private(index_path, index)
    with pytest.raises(execution.ExecutionFailure):
        execution.build_blind_assessment(
            ROOT, global_root, local, mapping_path, context
        )


def test_rehashed_dispatched_failure_cannot_drop_frozen_request_evidence(tmp_path):
    local, global_root, _, _, _, candidate, approval = prepare(tmp_path)
    context = authority_context(candidate, approval)
    cell_id = EXECUTION_ORDER[0]
    _, visible_evidence = execution._validate_requests(ROOT)
    execution.publish_authority_claim(
        global_root, local, cell_id, REQUEST_HASHES[0], context, NOW
    )
    execution.publish_terminal(
        global_root,
        local,
        cell_id,
        REQUEST_HASHES[0],
        context,
        kind="failure",
        dispatch_invoked=True,
        server_acceptance="unknown",
        provider_visible_evidence=visible_evidence[0],
        structured_response=None,
        raw_bytes=None,
        failure_category="transport_error",
        recorded_at=NOW,
        conservative_cost_upper_bound="0.00001",
    )
    terminal_path = global_root / "terminals" / f"{cell_id}.terminal.json"
    index_path = global_root / "terminal-index" / f"{cell_id}.json"
    value = json.loads(terminal_path.read_bytes())
    value["terminal"]["provider_visible_evidence"] = None
    value["terminal_digest"] = execution.sha256_canonical(value["terminal"])
    index = json.loads(index_path.read_bytes())
    index["terminal_index"]["terminal_digest"] = value["terminal_digest"]
    index["terminal_index_digest"] = execution.sha256_canonical(index["terminal_index"])
    _rewrite_private(terminal_path, value)
    _rewrite_private(index_path, index)
    with pytest.raises(execution.ExecutionFailure):
        execution.verify_terminal(
            global_root,
            cell_id,
            REQUEST_HASHES[0],
            context,
            visible_evidence[0],
        )


@pytest.mark.parametrize("publication_failure", ["terminals", "terminal-index"])
def test_terminal_publication_crash_windows_abort_and_never_overwrite(
    tmp_path, monkeypatch, publication_failure
):
    local, global_root, _, _, _, candidate, approval = prepare(tmp_path)
    context = authority_context(candidate, approval)
    cell_id = EXECUTION_ORDER[0]
    execution.publish_authority_claim(
        global_root, local, cell_id, REQUEST_HASHES[0], context, NOW
    )
    original = execution._write_private_bytes_no_overwrite

    def crash(path, data):
        if path.parent.name == publication_failure:
            raise execution.ExecutionFailure("simulated_publication_crash")
        return original(path, data)

    monkeypatch.setattr(execution, "_write_private_bytes_no_overwrite", crash)
    with pytest.raises(execution.ExecutionFailure):
        execution.publish_terminal(
            global_root,
            local,
            cell_id,
            REQUEST_HASHES[0],
            context,
            kind="success",
            dispatch_invoked=True,
            server_acceptance="yes",
            provider_visible_evidence={"task": "task", "timestamped_evidence": []},
            structured_response=VALID_STRUCTURED,
            raw_bytes=success_raw(0),
            recorded_at=NOW,
            usage=SUCCESS_USAGE,
            actual_cost=SUCCESS_ACTUAL_COST,
            conservative_cost_upper_bound=SUCCESS_UPPER_COST,
            response_metadata=observed_metadata(),
        )
    with pytest.raises(execution.ExecutionFailure):
        execution.classify_state(
            global_root, local, cell_id, REQUEST_HASHES[0], context
        )
    monkeypatch.setattr(execution, "_write_private_bytes_no_overwrite", original)
    with pytest.raises(FileExistsError):
        execution.publish_terminal(
            global_root,
            local,
            cell_id,
            REQUEST_HASHES[0],
            context,
            kind="success",
            dispatch_invoked=True,
            server_acceptance="yes",
            provider_visible_evidence={"task": "task", "timestamped_evidence": []},
            structured_response=VALID_STRUCTURED,
            raw_bytes=success_raw(0),
            recorded_at=NOW,
            usage=SUCCESS_USAGE,
            actual_cost=SUCCESS_ACTUAL_COST,
            conservative_cost_upper_bound=SUCCESS_UPPER_COST,
            response_metadata=observed_metadata(),
        )
    assert (global_root / "raw" / f"{cell_id}.raw").read_bytes() == success_raw(0)


def test_parent_revalidation_failure_preserves_winner_and_prevents_overwrite(
    tmp_path, monkeypatch
):
    root = private_root(tmp_path, "private")
    path = root / "record.json"
    monkeypatch.setattr(
        execution,
        "_revalidate_parent",
        lambda *_: (_ for _ in ()).throw(execution.ExecutionFailure("parent_replaced")),
    )
    with pytest.raises(execution.ExecutionFailure):
        execution.write_test_private_record(path, {"winner": 1})
    assert path.exists()
    with pytest.raises(FileExistsError):
        execution.write_test_private_record(path, {"loser": 2})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("owner_identity", "wrong-owner"),
        ("authorization_id", "task12b-auth-" + "0" * 32),
        ("candidate_digest", "sha256:" + "0" * 64),
        ("approval_digest", "sha256:" + "0" * 64),
        ("contract_lineage", "wrong-lineage"),
        ("cell_id", EXECUTION_ORDER[1]),
        ("request_hash", REQUEST_HASHES[1]),
        ("orphan_claim_digest", "sha256:" + "0" * 64),
        ("owner_echoed_orphan_claim_digest_out_of_band", False),
        ("operational_process_evidence_only", False),
        ("authorized_at", "2026-08-04T12:00:00Z"),
        ("authorized_at", "2026-08-04T13:00:00.000001Z"),
    ],
)
def test_orphan_finalization_rejects_wrong_owner_context_echo_or_time(
    tmp_path, field, value
):
    local, global_root, _, _, _, candidate, approval = prepare(tmp_path)
    context = authority_context(candidate, approval)
    claim = execution.publish_authority_claim(
        global_root, local, EXECUTION_ORDER[0], REQUEST_HASHES[0], context, NOW
    )
    payload = {
        "version": execution.ORPHAN_FINALIZATION_VERSION,
        "authorization_id": context.authorization_id,
        "candidate_digest": context.candidate_digest,
        "approval_digest": context.approval_digest,
        "contract_lineage": execution.CONTRACT_LINEAGE,
        "cell_id": EXECUTION_ORDER[0],
        "request_hash": REQUEST_HASHES[0],
        "orphan_claim_digest": claim["claim_digest"],
        "owner_echoed_orphan_claim_digest_out_of_band": True,
        "owner_confirmed_no_process_remains": True,
        "confirmation_process": "Owner inspected the process table and lock out of band.",
        "owner_identity": context.owner_identity,
        "authorized_at": NOW,
        "operational_process_evidence_only": True,
    }
    payload[field] = value
    envelope = {
        "finalization_authorization": payload,
        "finalization_authorization_digest": execution.sha256_canonical(payload),
    }
    path = local / "bad-finalization.json"
    execution.write_test_private_record(path, envelope)
    with pytest.raises(execution.ExecutionFailure):
        execution.finalize_orphan_offline(
            global_root,
            local,
            EXECUTION_ORDER[0],
            REQUEST_HASHES[0],
            context,
            path,
            NOW,
        )
    assert not (
        global_root / "terminals" / f"{EXECUTION_ORDER[0]}.terminal.json"
    ).exists()


def test_secret_canary_omits_unsafe_raw_but_keeps_exact_request_evidence(
    tmp_path, monkeypatch
):
    (
        local,
        global_root,
        candidate_path,
        mapping_path,
        approval_path,
        candidate,
        approval,
    ) = live_fixture(tmp_path, monkeypatch)
    api_key = "placeholder-key"
    raw, response = transport_pair()
    document = json.loads(raw)
    document["unsafe_echo"] = api_key
    client = SequenceClient(
        [TransportRaw(json.dumps(document, separators=(",", ":")).encode(), response)]
    )
    result = execution.execute_authorized_manifest(
        ROOT,
        global_root,
        local,
        candidate_path,
        approval_path,
        mapping_path,
        api_key=api_key,
        client_factory=lambda _: client,
        now=NOW,
    )
    assert result == {
        "status": "run_incomplete",
        "dispatches": 1,
        "successes": 0,
        "failures": 1,
        "pending": 8,
    }
    context = authority_context(candidate, approval)
    _, visible_evidence = execution._validate_requests(ROOT)
    terminal = execution.verify_terminal(
        global_root,
        EXECUTION_ORDER[0],
        REQUEST_HASHES[0],
        context,
        visible_evidence[0],
    )["terminal"]
    assert terminal["failure_category"] == "secret_detected"
    assert terminal["provider_visible_evidence"] == visible_evidence[0]
    assert terminal["raw_response_sha256"] is None
    assert not (global_root / "raw" / f"{EXECUTION_ORDER[0]}.raw").exists()
    assert len(client.calls) == 1
    assert client.closed is True


def transport_pair(**changes):
    text = json.dumps(VALID_STRUCTURED, separators=(",", ":"))
    document = {
        "id": "resp_task12b_1",
        "model": execution.PINNED_MODEL,
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
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
            "input_tokens_details": {"cached_tokens": 10, "cache_write_tokens": 5},
        },
    }
    document.update(changes)
    details = document.get("usage", {}).get("input_tokens_details", {})
    response = SimpleNamespace(
        id=document.get("id"),
        model=document.get("model"),
        status=document.get("status"),
        incomplete_details=document.get("incomplete_details"),
        output_text=text,
        output=(),
        usage=SimpleNamespace(
            input_tokens=document.get("usage", {}).get("input_tokens"),
            output_tokens=document.get("usage", {}).get("output_tokens"),
            total_tokens=document.get("usage", {}).get("total_tokens"),
            input_tokens_details=SimpleNamespace(
                cached_tokens=details.get("cached_tokens"),
                cache_write_tokens=details.get("cache_write_tokens"),
            ),
        ),
    )
    return json.dumps(document, separators=(",", ":")).encode(), response


class TransportRaw:
    def __init__(
        self,
        raw,
        response,
        *,
        status_code=200,
        content_type="application/problem+json; charset=utf-8",
    ):
        self.content = raw
        self.request_id = "req_task12b_1"
        self.status_code = status_code
        self.headers = {
            "content-type": content_type,
            "x-request-id": self.request_id,
        }
        self.response = response
        self.parse_calls = 0

    def parse(self):
        self.parse_calls += 1
        return self.response


class SequenceClient:
    def __init__(self, results, before=None):
        self.results = list(results)
        self.calls = []
        self.before = before
        self.closed = False
        self.responses = SimpleNamespace(
            with_raw_response=SimpleNamespace(create=self.create)
        )

    def create(self, **body):
        if self.before is not None:
            self.before(len(self.calls), body)
        self.calls.append(body)
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    def close(self):
        self.closed = True


def test_dispatch_once_preserves_exact_body_schema_usage_and_cost():
    requests, _ = execution._validate_requests(ROOT)
    raw, response = transport_pair()
    client = SequenceClient([TransportRaw(raw, response)])
    result = execution.dispatch_once(client, requests[0], REQUEST_HASHES[0])
    assert client.calls == [requests[0]]
    assert result.raw_bytes == raw
    assert result.usage == {
        "input_tokens": 100,
        "cached_input_tokens": 10,
        "cache_write_input_tokens": 5,
        "output_tokens": 20,
        "total_tokens": 120,
    }
    assert result.actual_cost == "0.0004245"
    assert result.conservative_cost_upper_bound == "0.00049"
    assert result.response_metadata == {
        "http_status": 200,
        "content_type": "application/problem+json; charset=utf-8",
        "provider_request_id": "req_task12b_1",
        "response_id": "resp_task12b_1",
        "observed_model": execution.PINNED_MODEL,
    }


def test_absent_cache_write_is_unknown_not_zero_and_upper_still_exact():
    raw, response = transport_pair()
    document = json.loads(raw)
    document["usage"]["input_tokens_details"].pop("cache_write_tokens")
    response.usage.input_tokens_details.cache_write_tokens = None
    usage = execution.validate_usage(document, response)
    actual, upper = execution.usage_costs(usage)
    assert usage["cache_write_input_tokens"] is None
    assert actual is None
    assert upper == "0.00049"


@pytest.mark.parametrize(
    ("raw_detail", "sdk_cached", "sdk_cache_write"),
    [
        (None, 99, None),
        ({"cached_tokens": 10}, 10, 0),
    ],
)
def test_raw_and_sdk_usage_detail_presence_contradictions_are_rejected(
    raw_detail, sdk_cached, sdk_cache_write
):
    raw, response = transport_pair()
    document = json.loads(raw)
    if raw_detail is None:
        document["usage"].pop("input_tokens_details")
    else:
        document["usage"]["input_tokens_details"] = raw_detail
    response.usage.input_tokens_details = SimpleNamespace(
        cached_tokens=sdk_cached, cache_write_tokens=sdk_cache_write
    )
    with pytest.raises(execution.ExecutionFailure):
        execution.validate_usage(document, response)


@pytest.mark.parametrize(
    "reasoning_item",
    [
        {"type": "reasoning", "tool_call": {"name": "exfiltrate"}},
        {"type": "reasoning", "action": "tool"},
        {"type": "reasoning", "arguments": {"secret": "x"}},
        {"type": "reasoning", "unknown": "unmodeled"},
    ],
)
def test_reasoning_items_reject_every_unmodeled_or_action_field(reasoning_item):
    raw, response = transport_pair()
    document = json.loads(raw)
    document["output"].insert(0, reasoning_item)
    response.output_text = document["output"][1]["content"][0]["text"]
    with pytest.raises(execution.ExecutionFailure) as caught:
        execution.dispatch_once(
            SequenceClient(
                [
                    TransportRaw(
                        json.dumps(document, separators=(",", ":")).encode(),
                        response,
                    )
                ]
            ),
            execution._validate_requests(ROOT)[0][0],
            REQUEST_HASHES[0],
        )
    assert caught.value.category == "malformed_response"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(extra=True),
        lambda value: value.update(confidence="certain"),
        lambda value: value.update(diagnosis=""),
        lambda value: value.update(supporting_evidence_numbers=[True]),
        lambda value: value.update(next_safe_actions=[]),
        lambda value: value.update(actions_to_avoid=["x\nsecret"]),
    ],
)
def test_strict_task12b_response_schema_rejects_mutations(mutation):
    value = json.loads(json.dumps(VALID_STRUCTURED))
    mutation(value)
    with pytest.raises(execution.ExecutionFailure) as caught:
        execution.validate_task12b_response(value)
    assert caught.value.category == "invalid_response"


@pytest.mark.parametrize(
    ("changes", "category"),
    [
        (
            {"status": "incomplete", "incomplete_details": {"reason": "limit"}},
            "invalid_response",
        ),
        ({"model": "other"}, "invalid_response"),
        (
            {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "refusal", "refusal": "no"}],
                    }
                ]
            },
            "provider_refusal",
        ),
    ],
)
def test_dispatch_once_rejects_incomplete_identity_and_refusal_with_raw(
    changes, category
):
    requests, _ = execution._validate_requests(ROOT)
    raw, response = transport_pair(**changes)
    response.status = changes.get("status", response.status)
    response.incomplete_details = changes.get(
        "incomplete_details", response.incomplete_details
    )
    response.model = changes.get("model", response.model)
    with pytest.raises(execution.ExecutionFailure) as caught:
        execution.dispatch_once(
            SequenceClient([TransportRaw(raw, response)]),
            requests[0],
            REQUEST_HASHES[0],
        )
    assert caught.value.category == category
    assert caught.value.raw_bytes == raw


def sdk_client(handler):
    import httpx
    import openai

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(
        transport=transport,
        trust_env=False,
        follow_redirects=False,
        timeout=30.0,
    )
    return openai.OpenAI(
        api_key="sk-test-placeholder-not-a-real-key",
        base_url="https://task12b.invalid/v1",
        max_retries=0,
        timeout=30.0,
        default_headers={},
        default_query={},
        http_client=http_client,
    )


def sdk_response_document(**changes):
    raw, _ = transport_pair(**changes)
    return json.loads(raw)


def test_real_sdk_mocktransport_success_is_one_stateless_literal_post():
    import httpx

    requests, _ = execution._validate_requests(ROOT)
    seen = []

    def handler(request):
        seen.append(request)
        assert request.method == "POST"
        assert request.url == httpx.URL("https://task12b.invalid/v1/responses")
        assert json.loads(request.content) == requests[0]
        assert "previous_response_id" not in json.loads(request.content)
        return httpx.Response(
            200,
            headers={"content-type": "application/json", "x-request-id": "req_sdk_1"},
            json=sdk_response_document(),
        )

    client = sdk_client(handler)
    try:
        result = execution.dispatch_once(client, requests[0], REQUEST_HASHES[0])
    finally:
        client.close()
    assert result.structured_response == VALID_STRUCTURED
    assert len(seen) == 1


def test_real_sdk_mocktransport_rejects_action_like_reasoning_item_once():
    import httpx

    requests, _ = execution._validate_requests(ROOT)
    document = sdk_response_document()
    document["output"].insert(
        0,
        {
            "type": "reasoning",
            "id": "rs_untrusted",
            "summary": [],
            "tool_call": {"name": "unmodeled_action"},
        },
    )
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "application/json", "x-request-id": "req_sdk_2"},
            json=document,
        )

    client = sdk_client(handler)
    try:
        with pytest.raises(execution.ExecutionFailure) as caught:
            execution.dispatch_once(client, requests[0], REQUEST_HASHES[0])
    finally:
        client.close()
    assert caught.value.category == "malformed_response"
    assert caught.value.raw_bytes is not None
    assert len(seen) == 1


@pytest.mark.parametrize("status", [307, 429, 500])
def test_real_sdk_mocktransport_http_status_has_one_handler_no_redirect_or_retry(
    status,
):
    import httpx

    requests, _ = execution._validate_requests(ROOT)
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(
            status,
            headers={
                "content-type": "application/json",
                "x-request-id": "req_sdk_error",
                "location": "https://redirect.invalid/v1/responses",
            },
            json={"error": {"type": "fixed_test_error"}},
        )

    client = sdk_client(handler)
    try:
        with pytest.raises(execution.ExecutionFailure) as caught:
            execution.dispatch_once(client, requests[0], REQUEST_HASHES[0])
    finally:
        client.close()
    assert caught.value.category == "http_error"
    assert caught.value.raw_bytes is not None
    assert caught.value.response_metadata["http_status"] == status
    assert len(seen) == 1


def test_real_sdk_mocktransport_malformed_200_and_connection_failure_are_single_call():
    import httpx

    requests, _ = execution._validate_requests(ROOT)
    for response_or_error, expected, has_raw in (
        (
            httpx.Response(
                200, headers={"x-request-id": "req_bad"}, content=b"not-json"
            ),
            "malformed_response",
            True,
        ),
        (httpx.ConnectError("secret test transport text"), "transport_error", False),
    ):
        count = [0]

        def handler(request, item=response_or_error):
            count[0] += 1
            if isinstance(item, BaseException):
                raise item
            return item

        client = sdk_client(handler)
        try:
            with pytest.raises(execution.ExecutionFailure) as caught:
                execution.dispatch_once(client, requests[0], REQUEST_HASHES[0])
        finally:
            client.close()
        assert caught.value.category == expected
        assert (caught.value.raw_bytes is not None) is has_raw
        assert count == [1]


@pytest.mark.parametrize(
    ("changes", "category"),
    [
        (
            {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "refusal", "refusal": "fixed refusal"}],
                    }
                ]
            },
            "provider_refusal",
        ),
        (
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
            },
            "invalid_response",
        ),
    ],
)
def test_real_sdk_mocktransport_refusal_and_incomplete_preserve_observed_evidence(
    changes, category
):
    import httpx

    requests, _ = execution._validate_requests(ROOT)
    seen = []
    document = sdk_response_document(**changes)

    def handler(request):
        seen.append(request)
        return httpx.Response(
            200,
            headers={
                "content-type": "application/problem+json; charset=utf-8",
                "x-request-id": "req_sdk_semantic",
            },
            json=document,
        )

    client = sdk_client(handler)
    try:
        with pytest.raises(execution.ExecutionFailure) as caught:
            execution.dispatch_once(client, requests[0], REQUEST_HASHES[0])
    finally:
        client.close()
    failure = caught.value
    assert failure.category == category
    assert failure.raw_bytes == json.dumps(document, separators=(",", ":")).encode()
    assert failure.response_metadata == {
        "http_status": 200,
        "content_type": "application/problem+json; charset=utf-8",
        "provider_request_id": "req_sdk_semantic",
        "response_id": "resp_task12b_1",
        "observed_model": execution.PINNED_MODEL,
    }
    assert failure.usage is not None
    assert failure.conservative_cost_upper_bound == "0.00049"
    assert len(seen) == 1


def test_paid_guard_removes_every_ambient_name_through_client_close(monkeypatch):
    hostile = {
        "OPENAI_BASE_URL": "https://hostile.invalid",
        "OPENAI_FUTURE_ROUTING": "hostile",
        "OPENAI_LOG": "debug",
        "HTTPS_PROXY": "https://proxy.invalid",
        "https_proxy": "https://proxy.invalid",
        "NO_PROXY": "*",
        "SSL_CERT_FILE": "/hostile/cert",
        "REQUESTS_CA_BUNDLE": "/hostile/bundle",
        "SSLKEYLOGFILE": "/hostile/keylog",
    }
    for name, value in hostile.items():
        monkeypatch.setenv(name, value)
    observed = []

    class Close:
        def close(self):
            observed.append(("close", {name: os.environ.get(name) for name in hostile}))

    with execution.paid_ambient_guard():
        observed.append(("body", {name: os.environ.get(name) for name in hostile}))
        Close().close()
        assert logging.root.manager.disable == logging.CRITICAL
    assert all(value is None for _, snapshot in observed for value in snapshot.values())
    assert {name: os.environ.get(name) for name in hostile} == hostile


def test_runtime_version_mismatch_fails_before_client(monkeypatch):
    real_import = execution.importlib.import_module

    def imported(name):
        if name == "openai":
            return SimpleNamespace(__version__="wrong")
        if name == "httpx":
            return SimpleNamespace(__version__=execution.PINNED_HTTPX_VERSION)
        return real_import(name)

    monkeypatch.setattr(execution.importlib, "import_module", imported)
    with pytest.raises(execution.ExecutionFailure):
        execution._load_live_modules()


def test_official_client_configuration_observed_with_fake_modules(monkeypatch):
    observations = {}

    class HTTPClient:
        def __init__(self, **kwargs):
            observations["http"] = kwargs

        def close(self):
            observations["http_closed"] = True

    class OpenAIClient:
        def __init__(self, **kwargs):
            observations["openai"] = kwargs

    fake_openai = SimpleNamespace(
        __version__=execution.PINNED_OPENAI_VERSION, OpenAI=OpenAIClient
    )
    fake_httpx = SimpleNamespace(
        __version__=execution.PINNED_HTTPX_VERSION, Client=HTTPClient
    )
    monkeypatch.setattr(
        execution, "_load_live_modules", lambda: (fake_openai, fake_httpx)
    )
    monkeypatch.setattr(
        execution.ssl,
        "create_default_context",
        lambda: SimpleNamespace(keylog_filename=None),
    )
    client = execution._create_live_client("sk-test-placeholder-not-a-real-key")
    assert isinstance(client, OpenAIClient)
    assert observations["http"]["trust_env"] is False
    assert observations["http"]["follow_redirects"] is False
    assert observations["http"]["timeout"] == 30.0
    assert observations["openai"]["base_url"] == "https://api.openai.com/v1"
    assert observations["openai"]["max_retries"] == 0
    assert observations["openai"]["default_headers"] == {}
    assert observations["openai"]["default_query"] == {}


def test_default_cli_never_reads_credential_writes_private_or_imports_sdk(
    tmp_path, monkeypatch, capsys
):
    private = tmp_path / "must-not-exist"
    monkeypatch.setattr(
        execution, "_load_credential", lambda *_: pytest.fail("credential read")
    )
    monkeypatch.setattr(
        execution, "_create_live_client", lambda *_: pytest.fail("client")
    )
    before = set(sys.modules)
    assert (
        execution.main([], _repo_root=ROOT, _local_root=private, _global_root=private)
        == 0
    )
    assert not private.exists()
    assert "openai" not in set(sys.modules) - before
    output = capsys.readouterr().out
    assert "Task:" not in output and "Evidence:" not in output
    assert "network_attempts=0" in output


def test_argparse_exposes_no_production_path_override(capsys):
    with pytest.raises(SystemExit):
        execution.main(["--output-dir", "/tmp/escape"], _repo_root=ROOT)
    assert "--output-dir" not in capsys.readouterr().out


def initialize_preflight_repo(path):
    path.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main"], cwd=path, check=True, stdout=subprocess.DEVNULL
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"], cwd=path, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    (path / "tracked").write_text("clean")
    subprocess.run(["git", "add", "tracked"], cwd=path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "test"], cwd=path, check=True, stdout=subprocess.DEVNULL
    )
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()
    subprocess.run(
        ["git", "update-ref", "refs/remotes/origin/main", head], cwd=path, check=True
    )
    return head


def test_repository_preflight_branch_dirty_untracked_and_ref_mismatch(tmp_path):
    repo = tmp_path / "repo"
    head = initialize_preflight_repo(repo)
    assert execution.repository_preflight(repo, head) == head
    (repo / "untracked").write_text("dirty")
    with pytest.raises(execution.ExecutionFailure):
        execution.repository_preflight(repo, head)
    (repo / "untracked").unlink()
    subprocess.run(
        ["git", "checkout", "-b", "other"],
        cwd=repo,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    with pytest.raises(execution.ExecutionFailure):
        execution.repository_preflight(repo, head)
    subprocess.run(
        ["git", "checkout", "main"], cwd=repo, check=True, stdout=subprocess.DEVNULL
    )
    subprocess.run(
        ["git", "update-ref", "refs/remotes/origin/main", "0" * 40],
        cwd=repo,
        check=False,
    )
    with pytest.raises(execution.ExecutionFailure):
        execution.repository_preflight(repo, head)


def live_fixture(tmp_path, monkeypatch):
    prepared = prepare(tmp_path)
    monkeypatch.setattr(execution, "repository_preflight", lambda *_: REVISION)
    monkeypatch.setattr(execution, "host_fingerprint", lambda: HOST)
    monkeypatch.setattr(execution, "account_fingerprint", lambda: ACCOUNT)
    monkeypatch.setattr(execution, "credential_fingerprint", lambda _: CREDENTIAL)
    return prepared


def nine_transport_results():
    results = []
    for index in range(9):
        raw, response = transport_pair(id="resp_task12b_%d" % index)
        results.append(TransportRaw(raw, response))
    return results


def test_fake_client_nine_successes_claim_before_exact_stateless_dispatch_and_close(
    tmp_path, monkeypatch
):
    local, global_root, candidate_path, mapping_path, approval_path, _, _ = (
        live_fixture(tmp_path, monkeypatch)
    )
    requests, _ = execution._validate_requests(ROOT)

    def before(index, body):
        cell_id = EXECUTION_ORDER[index]
        assert (global_root / "claims" / f"{cell_id}.claim.json").exists()
        assert (local / "mirrors" / "claims" / f"{cell_id}.claim.json").exists()
        assert body == requests[index]
        assert "previous_response_id" not in body
        assert "conversation" not in body
        assert body["tools"] == []

    client = SequenceClient(nine_transport_results(), before=before)
    result = execution.execute_authorized_manifest(
        ROOT,
        global_root,
        local,
        candidate_path,
        approval_path,
        mapping_path,
        api_key="sk-test-placeholder-not-a-real-key",
        client_factory=lambda _: client,
        now=NOW,
    )
    assert result == {
        "status": "run_complete",
        "dispatches": 9,
        "successes": 9,
        "failures": 0,
        "pending": 0,
    }
    assert client.calls == requests
    assert client.closed is True


def test_authorization_is_rechecked_before_every_claim_and_stops_after_expiry(
    tmp_path, monkeypatch
):
    (
        local,
        global_root,
        candidate_path,
        mapping_path,
        approval_path,
        candidate,
        approval,
    ) = live_fixture(tmp_path, monkeypatch)
    client = SequenceClient(nine_transport_results())
    observations = iter(
        [
            "2026-08-04T12:59:59.000000Z",
            "2026-08-04T13:00:00.000000Z",
            "2026-08-04T13:00:00.000001Z",
        ]
    )
    with pytest.raises(execution.ExecutionFailure) as caught:
        execution.execute_authorized_manifest(
            ROOT,
            global_root,
            local,
            candidate_path,
            approval_path,
            mapping_path,
            api_key="«redacted:sk-…»",
            client_factory=lambda _: client,
            now=NOW,
            _clock=lambda: next(observations),
        )
    assert caught.value.category == "authorization_not_live_no_network_attempt"
    assert len(client.calls) == 1
    assert client.closed is True
    context = authority_context(candidate, approval)
    _, visible_evidence = execution._validate_requests(ROOT)
    execution.verify_terminal(
        global_root,
        EXECUTION_ORDER[0],
        REQUEST_HASHES[0],
        context,
        visible_evidence[0],
    )
    assert not (global_root / "claims" / f"{EXECUTION_ORDER[1]}.claim.json").exists()


def test_fake_client_provider_failure_continues_then_all_terminal_resume_builds_no_client(
    tmp_path, monkeypatch
):
    (
        local,
        global_root,
        candidate_path,
        mapping_path,
        approval_path,
        candidate,
        approval,
    ) = live_fixture(tmp_path, monkeypatch)
    client = SequenceClient(
        [ConnectionError("provider text must stay private")]
        + nine_transport_results()[:8]
    )
    first = execution.execute_authorized_manifest(
        ROOT,
        global_root,
        local,
        candidate_path,
        approval_path,
        mapping_path,
        api_key="sk-test-placeholder-not-a-real-key",
        client_factory=lambda _: client,
        now=NOW,
    )
    assert first["dispatches"] == 9
    assert len(client.calls) == 9
    context = authority_context(candidate, approval)
    first_terminal = execution.verify_terminal(
        global_root, EXECUTION_ORDER[0], REQUEST_HASHES[0], context
    )["terminal"]
    assert first_terminal["failure_category"] == "transport_error"
    factories = []
    resumed = execution.execute_authorized_manifest(
        ROOT,
        global_root,
        local,
        candidate_path,
        approval_path,
        mapping_path,
        api_key="sk-test-placeholder-not-a-real-key",
        client_factory=lambda _: factories.append(True),
        now=NOW,
    )
    assert resumed == {
        "status": "all_terminal",
        "dispatches": 0,
        "successes": 8,
        "failures": 1,
        "pending": 0,
    }
    assert factories == []


def test_fake_client_budget_input_bound_is_terminal_and_stops_later_dispatch(
    tmp_path, monkeypatch
):
    (
        local,
        global_root,
        candidate_path,
        mapping_path,
        approval_path,
        candidate,
        approval,
    ) = live_fixture(tmp_path, monkeypatch)
    raw, response = transport_pair()
    document = json.loads(raw)
    document["usage"] = {
        "input_tokens": 999999,
        "output_tokens": 1,
        "total_tokens": 1000000,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
    }
    response.usage = SimpleNamespace(
        input_tokens=999999,
        output_tokens=1,
        total_tokens=1000000,
        input_tokens_details=SimpleNamespace(cached_tokens=0, cache_write_tokens=0),
    )
    client = SequenceClient(
        [TransportRaw(json.dumps(document, separators=(",", ":")).encode(), response)]
        + nine_transport_results()[:8]
    )
    result = execution.execute_authorized_manifest(
        ROOT,
        global_root,
        local,
        candidate_path,
        approval_path,
        mapping_path,
        api_key="sk-test-placeholder-not-a-real-key",
        client_factory=lambda _: client,
        now=NOW,
    )
    assert result["dispatches"] == 1
    assert len(client.calls) == 1
    terminal = execution.verify_terminal(
        global_root,
        EXECUTION_ORDER[0],
        REQUEST_HASHES[0],
        authority_context(candidate, approval),
    )["terminal"]
    assert terminal["failure_category"] == "budget_bound_violation"


def test_fake_client_local_claim_failure_consumes_authority_and_dispatches_zero(
    tmp_path, monkeypatch
):
    local, global_root, candidate_path, mapping_path, approval_path, _, _ = (
        live_fixture(tmp_path, monkeypatch)
    )
    original = execution._write_private_bytes_no_overwrite

    def fail_local_claim(path, data):
        if path.parent == local / "mirrors" / "claims":
            raise execution.ExecutionFailure("local_claim_failure")
        return original(path, data)

    monkeypatch.setattr(
        execution, "_write_private_bytes_no_overwrite", fail_local_claim
    )
    client = SequenceClient(nine_transport_results())
    with pytest.raises(execution.ExecutionFailure):
        execution.execute_authorized_manifest(
            ROOT,
            global_root,
            local,
            candidate_path,
            approval_path,
            mapping_path,
            api_key="sk-test-placeholder-not-a-real-key",
            client_factory=lambda _: client,
            now=NOW,
        )
    assert client.calls == []
    assert client.closed is True
    assert (global_root / "claims" / f"{EXECUTION_ORDER[0]}.claim.json").exists()


@pytest.mark.parametrize(
    "publication_failure", ["raw", "terminals", "terminal-index", "local-terminal"]
)
def test_execute_evidence_publication_failure_escapes_without_second_terminal_or_dispatch(
    tmp_path, monkeypatch, publication_failure
):
    local, global_root, candidate_path, mapping_path, approval_path, _, _ = (
        live_fixture(tmp_path, monkeypatch)
    )
    client = SequenceClient(nine_transport_results())
    original_write = execution._write_private_bytes_no_overwrite
    original_publish = execution.publish_terminal
    publications = []

    def fail_evidence(path, data):
        selected = path.parent.name == publication_failure or (
            publication_failure == "local-terminal"
            and path.parent == local / "mirrors" / "terminals"
        )
        if selected:
            raise execution.ExecutionFailure("simulated_publication_failure")
        return original_write(path, data)

    def counted_publish(*args, **kwargs):
        publications.append(args[2])
        return original_publish(*args, **kwargs)

    monkeypatch.setattr(execution, "_write_private_bytes_no_overwrite", fail_evidence)
    monkeypatch.setattr(execution, "publish_terminal", counted_publish)
    with pytest.raises(execution.ExecutionFailure):
        execution.execute_authorized_manifest(
            ROOT,
            global_root,
            local,
            candidate_path,
            approval_path,
            mapping_path,
            api_key="«redacted:sk-…»",
            client_factory=lambda _: client,
            now=NOW,
        )
    assert len(client.calls) == 1
    assert publications == [EXECUTION_ORDER[0]]
    assert not (global_root / "claims" / f"{EXECUTION_ORDER[1]}.claim.json").exists()


def test_connection_failure_gets_static_dispatch_upper_and_resume_counts_it(
    tmp_path, monkeypatch
):
    (
        local,
        global_root,
        candidate_path,
        mapping_path,
        approval_path,
        candidate,
        approval,
    ) = live_fixture(tmp_path, monkeypatch)
    client = SequenceClient(
        [ConnectionError("fixed transport failure")] + nine_transport_results()[:8]
    )
    result = execution.execute_authorized_manifest(
        ROOT,
        global_root,
        local,
        candidate_path,
        approval_path,
        mapping_path,
        api_key="«redacted:sk-…»",
        client_factory=lambda _: client,
        now=NOW,
    )
    context = authority_context(candidate, approval)
    terminal = execution.verify_terminal(
        global_root, EXECUTION_ORDER[0], REQUEST_HASHES[0], context
    )["terminal"]
    bounds = execution._projected_input_bounds(execution._validate_requests(ROOT)[0])
    assert terminal["actual_cost"] is None
    assert terminal[
        "conservative_cost_upper_bound"
    ] == execution._static_dispatch_upper(bounds[0])
    assert execution._accepted_upper(global_root, context) == sum(
        (
            __import__("decimal").Decimal(
                execution.verify_terminal(global_root, cell, request_hash, context)[
                    "terminal"
                ]["conservative_cost_upper_bound"]
            )
            for cell, request_hash in zip(EXECUTION_ORDER, REQUEST_HASHES)
        ),
        __import__("decimal").Decimal(0),
    )
    assert result["status"] == "run_incomplete"
    assert result["failures"] == 1


def test_raw_publication_requires_explicit_observed_metadata(tmp_path):
    local, global_root, _, _, _, candidate, approval = prepare(tmp_path)
    context = authority_context(candidate, approval)
    execution.publish_authority_claim(
        global_root, local, EXECUTION_ORDER[0], REQUEST_HASHES[0], context, NOW
    )
    with pytest.raises(execution.ExecutionFailure):
        execution.publish_terminal(
            global_root,
            local,
            EXECUTION_ORDER[0],
            REQUEST_HASHES[0],
            context,
            kind="success",
            dispatch_invoked=True,
            server_acceptance="yes",
            provider_visible_evidence={"task": "task", "timestamped_evidence": []},
            structured_response={"diagnosis": "bounded"},
            raw_bytes=b"raw",
            recorded_at=NOW,
            conservative_cost_upper_bound="0.00001",
        )


def _patch_historical_cli_identity(monkeypatch):
    monkeypatch.setattr(execution, "repository_preflight", lambda *args: REVISION)
    monkeypatch.setattr(execution, "host_fingerprint", lambda: HOST)
    monkeypatch.setattr(execution, "account_fingerprint", lambda: ACCOUNT)
    monkeypatch.setattr(execution, "credential_fingerprint", lambda _: CREDENTIAL)
    monkeypatch.setattr(execution, "_scan_committed_secret", lambda *_: None)


def test_offline_export_cli_succeeds_after_expiry_and_never_overwrites(
    tmp_path, monkeypatch
):
    local, global_root, _, mapping_path, _, candidate, approval = prepare(tmp_path)
    context = authority_context(candidate, approval)
    _publish_nine_successes(local, global_root, context)
    _patch_historical_cli_identity(monkeypatch)
    client_calls = []
    args = ["--export-blind-assessment"]
    assert (
        execution.main(
            args,
            _repo_root=ROOT,
            _local_root=local,
            _global_root=global_root,
            _api_key="placeholder-key",
            _client_factory=lambda _: client_calls.append(True),
            _now="2026-08-05T12:00:00.000000Z",
        )
        == 0
    )
    packet_path = local / "blind-assessment.json"
    packet = json.loads(packet_path.read_bytes())
    assert packet["assessment_ready"] is True
    assert len(packet["assessments"]) == 9
    assert stat.S_IMODE(packet_path.stat().st_mode) == 0o600
    assert client_calls == []
    assert (
        execution.main(
            args,
            _repo_root=ROOT,
            _local_root=local,
            _global_root=global_root,
            _api_key="placeholder-key",
            _client_factory=lambda _: client_calls.append(True),
            _now="2026-08-05T12:00:00.000000Z",
        )
        == 2
    )
    assert client_calls == []


def test_offline_orphan_finalization_cli_succeeds_after_expiry(tmp_path, monkeypatch):
    local, global_root, _, _, _, candidate, approval = prepare(tmp_path)
    context = authority_context(candidate, approval)
    cell_id = EXECUTION_ORDER[0]
    claim = execution.publish_authority_claim(
        global_root, local, cell_id, REQUEST_HASHES[0], context, NOW
    )
    payload = {
        "version": execution.ORPHAN_FINALIZATION_VERSION,
        "authorization_id": context.authorization_id,
        "candidate_digest": context.candidate_digest,
        "approval_digest": context.approval_digest,
        "contract_lineage": execution.CONTRACT_LINEAGE,
        "cell_id": cell_id,
        "request_hash": REQUEST_HASHES[0],
        "orphan_claim_digest": claim["claim_digest"],
        "owner_echoed_orphan_claim_digest_out_of_band": True,
        "owner_confirmed_no_process_remains": True,
        "confirmation_process": "Owner inspected the process table and lock out of band.",
        "owner_identity": context.owner_identity,
        "authorized_at": "2026-08-05T11:00:00.000000Z",
        "operational_process_evidence_only": True,
    }
    execution.write_test_private_record(
        local / "owner-orphan-finalization.json",
        {
            "finalization_authorization": payload,
            "finalization_authorization_digest": execution.sha256_canonical(payload),
        },
    )
    _patch_historical_cli_identity(monkeypatch)
    assert (
        execution.main(
            ["--finalize-authorized-orphan", "--orphan-cell", cell_id],
            _repo_root=ROOT,
            _local_root=local,
            _global_root=global_root,
            _api_key="placeholder-key",
            _client_factory=lambda _: pytest.fail("client constructed"),
            _now="2026-08-05T12:00:00.000000Z",
        )
        == 0
    )
    assert (
        execution.verify_terminal(global_root, cell_id, REQUEST_HASHES[0], context)[
            "terminal"
        ]["kind"]
        == "invalid_ambiguous"
    )


@pytest.mark.parametrize(
    "args",
    [
        ["--export-blind-assessment", "--owner-identity", "owner"],
        ["--execute-authorized-nine-cell-manifest", "--expires-at", "later"],
        ["--export-blind-assessment", "--orphan-cell", EXECUTION_ORDER[0]],
        ["--finalize-authorized-orphan"],
    ],
)
def test_cli_rejects_irrelevant_flag_combinations(args, tmp_path):
    assert (
        execution.main(
            args,
            _repo_root=ROOT,
            _local_root=tmp_path / "local",
            _global_root=tmp_path / "global",
            _api_key="placeholder-key",
        )
        == 2
    )
