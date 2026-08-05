"""Offline-first Task 12b v2 execution-readiness and evidence controller.

The default CLI is credential-free and non-authorizing.  The only provider
transport used by the explicit execution path is the reviewed v1
``dispatch_once`` function; this module owns all v2 authority identities and
never invokes v1 candidate, claim, terminal, or global-root helpers.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import pwd
import re
import secrets
import stat
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    MutableSequence,
    Optional,
    Sequence,
    Tuple,
    cast,
)

from . import context_sensitivity_execution as v1_execution
from .context_sensitivity_execution import ExecutionFailure as V1TransportFailure
from .context_sensitivity_execution import (
    TransportResult,
    dispatch_once,
)
from .context_sensitivity_replication_v2 import (
    CONTRACT_PATH,
    PINNED_CONTRACT_SHA256,
    ScheduledUnit,
    build_schedule,
    canonical_bytes,
    load_contract,
    render_unit_requests,
)

READINESS_PATH = Path(
    "experiments/adaptive_selection/controls/task12b_execution_readiness_v2.json"
)
LOCAL_OUTPUT_PATH = Path(".local/adaptive-selection-task12b-v2")
PINNED_READINESS_MANIFEST_SHA256 = (
    "d0b86651b0d33282222738821a816dc01b218f5b3e7e87f6d9a44a8b836e7afc"
)
CONTRACT_HASH = "sha256:" + PINNED_CONTRACT_SHA256
CONTRACT_LINEAGE = "task12b-context-sensitivity-repeated-draws-v2@" + CONTRACT_HASH
AUTHORITY_NAMESPACE = hashlib.sha256(CONTRACT_LINEAGE.encode()).hexdigest()
READINESS_HASH = "sha256:" + PINNED_READINESS_MANIFEST_SHA256

CANDIDATE_VERSION = "task12b-v2-authorization-candidate-v1"
MAPPING_VERSION = "task12b-v2-private-randomization-v1"
APPROVAL_VERSION = "task12b-v2-owner-approval-v1"
CLAIM_VERSION = "task12b-v2-unit-claim-v1"
TERMINAL_VERSION = "task12b-v2-unit-terminal-v1"
TERMINAL_INDEX_VERSION = "task12b-v2-unit-terminal-index-v1"
ANNOTATION_LOCK_VERSION = "task12b-v2-annotation-lock-v1"
ORPHAN_FINALIZATION_VERSION = "task12b-v2-orphan-finalization-v1"
BLIND_EXPORT_VERSION = "task12b-v2-blind-assessment-v1"
PINNED_MODEL = "gpt-5.6-terra"
PINNED_OPENAI_VERSION = "2.46.0"
PINNED_HTTPX_VERSION = "0.28.1"
PINNED_PROVIDER_SETTINGS = {
    "reasoning_effort": "medium",
    "max_output_tokens": 2048,
    "max_retries": 0,
    "store": False,
    "stream": False,
    "timeout_seconds": 30.0,
    "tools": [],
}
PINNED_PRICES = {
    "currency": "USD",
    "input_per_million": "2.00",
    "cached_input_per_million": "0.20",
    "cache_write_input_per_million": "2.50",
    "output_per_million": "12.00",
    "pricing_frozen_on": "2026-08-04",
    "official_pricing_source_url": "https://developers.openai.com/api/docs/pricing",
}
CONSERVATIVE_EXECUTION_CEILING = "1.470820"
OWNER_CAP = "2.00"
NO_RETRY_POLICY = "one claim and at most one dispatch per scheduled unit; max_retries=0; no retry, fallback, replacement, model substitution, premium rescue, rescheduling, tools, or adaptation"
FAILURE_CATEGORIES = {
    "http_error",
    "provider_refusal",
    "malformed_response",
    "invalid_response",
    "transport_error",
    "budget_bound_violation",
    "secret_detected",
    "preflight_rejected",
    "invalid_ambiguous_orphan_claim",
}


class ExecutionFailure(RuntimeError):
    """Sanitized controller failure."""

    def __init__(self, category: str = "preflight_rejected_no_network_attempt") -> None:
        self.category = category
        self.raw_bytes: Optional[bytes] = None
        self.response_metadata: Optional[Dict[str, Any]] = None
        self.usage: Optional[Dict[str, Any]] = None
        self.actual_cost: Optional[str] = None
        self.conservative_cost_upper_bound: Optional[str] = None
        super().__init__(category)


def _fail(category: str = "preflight_rejected_no_network_attempt") -> None:
    raise ExecutionFailure(category) from None


def sha256_canonical(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _parse_time(value: Any) -> datetime:
    if (
        type(value) is not str
        or re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z", value) is None
    ):
        _fail()
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        _fail()
    raise AssertionError


def _fingerprint(value: Any) -> bool:
    return (
        type(value) is str and re.fullmatch(r"sha256:[0-9a-f]{64}", value) is not None
    )


def _revision(value: Any) -> bool:
    return (
        type(value) is str
        and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value) is not None
    )


def _check_ancestors(path: Path) -> None:
    for ancestor in reversed(path.absolute().parents):
        if ancestor == Path(ancestor.anchor):
            continue
        try:
            mode = ancestor.lstat().st_mode
        except FileNotFoundError:
            continue
        except OSError:
            _fail()
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            _fail()


def _exists(path: Path) -> bool:
    try:
        path.lstat()
        return True
    except FileNotFoundError:
        return False
    except OSError:
        _fail()
    raise AssertionError


def _validate_private_directory(path: Path) -> None:
    _check_ancestors(path)
    try:
        metadata = path.lstat()
    except OSError:
        _fail()
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        _fail()


def _ensure_private_directory(path: Path) -> None:
    _check_ancestors(path)
    missing: List[Path] = []
    cursor = path
    while not _exists(cursor):
        missing.append(cursor)
        cursor = cursor.parent
    if cursor != Path(cursor.anchor):
        metadata = cursor.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            _fail()
    for item in reversed(missing):
        try:
            item.mkdir(mode=0o700)
        except OSError:
            _fail()
    _validate_private_directory(path)


def _open_parent(path: Path) -> Tuple[int, os.stat_result]:
    _validate_private_directory(path.parent)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(path.parent), flags)
        metadata = os.fstat(descriptor)
    except OSError:
        _fail()
    if not stat.S_ISDIR(metadata.st_mode):
        os.close(descriptor)
        _fail()
    return descriptor, metadata


def _revalidate_parent(path: Path, expected: os.stat_result) -> None:
    try:
        current = path.parent.lstat()
    except OSError:
        _fail()
    if stat.S_ISLNK(current.st_mode) or (current.st_dev, current.st_ino) != (
        expected.st_dev,
        expected.st_ino,
    ):
        _fail()


def _write_private_bytes_no_overwrite(path: Path, data: bytes) -> None:
    """Publish a complete fsynced temp inode with a no-overwrite hard link."""
    _ensure_private_directory(path.parent)
    parent_fd, parent_metadata = _open_parent(path)
    descriptor: Optional[int] = None
    temp_name = ".task12b-v2-publish-" + secrets.token_hex(16)
    temp_exists = False
    published = False
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(temp_name, flags, 0o600, dir_fd=parent_fd)
        temp_exists = True
        os.fchmod(descriptor, 0o600)
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError
            view = view[written:]
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_nlink != 1
        ):
            raise OSError
        os.link(
            temp_name,
            path.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        published = True
        os.unlink(temp_name, dir_fd=parent_fd)
        temp_exists = False
        final = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            (final.st_dev, final.st_ino) != (metadata.st_dev, metadata.st_ino)
            or final.st_nlink != 1
            or stat.S_IMODE(final.st_mode) != 0o600
        ):
            raise OSError
        os.fsync(parent_fd)
        _revalidate_parent(path, parent_metadata)
    except FileExistsError:
        raise
    except OSError:
        _fail()
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temp_exists:
            try:
                os.unlink(temp_name, dir_fd=parent_fd)
            except OSError:
                pass
        os.close(parent_fd)


def _read_private_bytes(path: Path, maximum: int = 16 * 1024 * 1024) -> bytes:
    parent_fd, parent_metadata = _open_parent(path)
    descriptor: Optional[int] = None
    try:
        descriptor = os.open(
            path.name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd
        )
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_nlink != 1
            or not 0 < before.st_size <= maximum
        ):
            raise OSError
        chunks: List[bytes] = []
        size = 0
        while True:
            part = os.read(descriptor, min(65536, maximum + 1 - size))
            if not part:
                break
            chunks.append(part)
            size += len(part)
            if size > maximum:
                raise OSError
        after = os.fstat(descriptor)
        final = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if (before.st_dev, before.st_ino, before.st_size) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
        ) or (after.st_dev, after.st_ino) != (final.st_dev, final.st_ino):
            raise OSError
        _revalidate_parent(path, parent_metadata)
        return b"".join(chunks)
    except OSError:
        _fail()
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)


def write_test_private_record(path: Path, value: Mapping[str, Any]) -> None:
    _write_private_bytes_no_overwrite(path, canonical_bytes(value) + b"\n")


def read_private_record(path: Path) -> Dict[str, Any]:
    try:
        raw = _read_private_bytes(path)
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        _fail()
    if type(value) is not dict or raw != canonical_bytes(value) + b"\n":
        _fail()
    return cast(Dict[str, Any], value)


def load_readiness_manifest(
    path: Path = READINESS_PATH,
) -> Tuple[Dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        _fail()
    if (
        hashlib.sha256(raw).hexdigest() != PINNED_READINESS_MANIFEST_SHA256
        or type(value) is not dict
    ):
        _fail()
    try:
        if (
            value["contract_lineage"]["identity"] != CONTRACT_LINEAGE
            or value["execution"]
            != {
                "network_requests_authorized_by_manifest": 0,
                "approved_scheduled_unit_count": 45,
                "candidate_grants_network_authority": False,
                "owner_approval_written_by_software": False,
                "separate_exact_digest_owner_echo_required": True,
                "cli_owner_approval_writer_present": False,
                "cli_exact_head_clean_preflight_before_candidate_and_execution": True,
                "cli_offline_verification_export_lock_scoring_and_orphan_surfaces_present": True,
                "cli_explicit_authorized_45_unit_execution_surface_present": True,
                "one_claim_and_at_most_one_dispatch_per_unit": True,
                "retries_fallbacks_replacements_and_adaptation_forbidden": True,
            }
            or value["provider_configuration"]
            != {
                "model": PINNED_MODEL,
                "openai_version": PINNED_OPENAI_VERSION,
                "httpx_version": PINNED_HTTPX_VERSION,
                **PINNED_PROVIDER_SETTINGS,
            }
            or value["budget"]
            != {
                **PINNED_PRICES,
                "conservative_execution_ceiling": CONSERVATIVE_EXECUTION_CEILING,
                "hard_owner_cap": OWNER_CAP,
                "pricing_must_be_reverified_before_candidate_preparation": True,
            }
        ):
            _fail()
    except (KeyError, TypeError):
        _fail()
    return value, raw


def default_authority_directory_v2() -> Path:
    account = pwd.getpwuid(os.getuid())
    return (
        Path(account.pw_dir)
        / ".local/state/ai-context-manager/task12b-authority"
        / AUTHORITY_NAMESPACE
    )


def _shuffle_changed(
    values: MutableSequence[Any], shuffle: Callable[[MutableSequence[Any]], None]
) -> None:
    before = list(values)
    shuffle(values)
    if values == before and len(values) > 1:
        values[:] = values[1:] + values[:1]


def build_private_randomization(
    authorization_id: str,
    nonce: str,
    *,
    token_bytes: Callable[[int], bytes] = secrets.token_bytes,
    shuffle: Callable[[MutableSequence[Any]], None] = secrets.SystemRandom().shuffle,
) -> Dict[str, Any]:
    contract, _ = load_contract()
    units = build_schedule(contract)
    blocks: List[List[Dict[str, Any]]] = []
    for draw in range(1, 6):
        block = [
            {"unit_id": u.unit_id, "base_cell_id": u.base_cell_id}
            for u in units
            if u.draw_index == draw
        ]
        _shuffle_changed(block, shuffle)
        blocks.append(block)
    _shuffle_changed(cast(MutableSequence[Any], blocks), shuffle)
    execution_order = [entry["unit_id"] for block in blocks for entry in block]
    if execution_order == [u.unit_id for u in units]:
        execution_order = execution_order[1:] + execution_order[:1]
    entries = []
    aliases = set()
    for index, unit in enumerate(units):
        # Domain-separate even deterministic test entropy.
        alias = (
            "assessment-"
            + hashlib.sha256(
                token_bytes(32) + nonce.encode() + index.to_bytes(2, "big")
            ).hexdigest()[:32]
        )
        if alias in aliases:
            _fail()
        aliases.add(alias)
        entries.append({"unit_id": unit.unit_id, "assessment_alias": alias})
    execution_alias_order = [
        next(e["assessment_alias"] for e in entries if e["unit_id"] == unit_id)
        for unit_id in execution_order
    ]
    assessment_order = list(aliases)
    _shuffle_changed(cast(MutableSequence[Any], assessment_order), shuffle)
    if assessment_order == execution_alias_order:
        assessment_order = assessment_order[1:] + assessment_order[:1]
    return {
        "version": MAPPING_VERSION,
        "authorization_id": authorization_id,
        "nonce": nonce,
        "contract_lineage": CONTRACT_LINEAGE,
        "execution_blocks": blocks,
        "execution_order": execution_order,
        "entries": entries,
        "assessment_order": assessment_order,
        "private_until_annotation_lock": True,
    }


def _mapping_semantics(
    mapping: Mapping[str, Any], candidate: Optional[Mapping[str, Any]] = None
) -> None:
    units = build_schedule(load_contract()[0])
    unit_lookup = {u.unit_id: u for u in units}
    unit_ids = set(unit_lookup)
    base_ids = {u.base_cell_id for u in units}
    required = {
        "version",
        "authorization_id",
        "nonce",
        "contract_lineage",
        "execution_blocks",
        "execution_order",
        "entries",
        "assessment_order",
        "private_until_annotation_lock",
    }
    if (
        set(mapping) != required
        or mapping["version"] != MAPPING_VERSION
        or mapping["contract_lineage"] != CONTRACT_LINEAGE
        or mapping["private_until_annotation_lock"] is not True
    ):
        _fail()
    if candidate is not None and (mapping["authorization_id"], mapping["nonce"]) != (
        candidate["authorization_id"],
        candidate["nonce"],
    ):
        _fail()
    blocks = mapping["execution_blocks"]
    if type(blocks) is not list or len(blocks) != 5:
        _fail()
    seen = []
    seen_draws = set()
    for block in blocks:
        if (
            type(block) is not list
            or len(block) != 9
            or any(type(x) is not dict for x in block)
            or {x.get("base_cell_id") for x in block} != base_ids
            or any(set(x) != {"unit_id", "base_cell_id"} for x in block)
        ):
            _fail()
        try:
            canonical = [unit_lookup[x["unit_id"]] for x in block]
        except (KeyError, TypeError):
            _fail()
        draws = {unit.draw_index for unit in canonical}
        if (
            len(draws) != 1
            or next(iter(draws)) in seen_draws
            or any(
                item["base_cell_id"] != unit.base_cell_id
                for item, unit in zip(block, canonical)
            )
        ):
            _fail()
        seen_draws.update(draws)
        seen.extend(x["unit_id"] for x in block)
    if (
        seen_draws != {1, 2, 3, 4, 5}
        or len(seen) != 45
        or len(set(seen)) != 45
        or set(seen) != unit_ids
        or mapping["execution_order"] != seen
    ):
        _fail()
    entries = mapping["entries"]
    if (
        type(entries) is not list
        or len(entries) != 45
        or {x.get("unit_id") for x in entries if type(x) is dict} != unit_ids
        or any(set(x) != {"unit_id", "assessment_alias"} for x in entries)
    ):
        _fail()
    aliases = [x["assessment_alias"] for x in entries]
    if len(set(aliases)) != 45 or any(
        type(x) is not str or re.fullmatch(r"assessment-[0-9a-f]{32}", x) is None
        for x in aliases
    ):
        _fail()
    order = mapping["assessment_order"]
    execution_aliases = [
        next(x["assessment_alias"] for x in entries if x["unit_id"] == unit)
        for unit in seen
    ]
    if (
        type(order) is not list
        or len(order) != 45
        or set(order) != set(aliases)
        or order == execution_aliases
    ):
        _fail()


def prepare_non_authorizing_candidate_v2(
    candidate_path: Path,
    mapping_path: Path,
    *,
    code_revision: str,
    owner_identity: str,
    host_fingerprint: str,
    account_fingerprint: str,
    credential_fingerprint: str,
    issued_at: str,
    expires_at: str,
    maximum_execution_window_seconds: int,
    token_bytes: Callable[[int], bytes] = secrets.token_bytes,
    shuffle: Callable[[MutableSequence[Any]], None] = secrets.SystemRandom().shuffle,
) -> Dict[str, Any]:
    readiness, readiness_raw = load_readiness_manifest(
        (Path(__file__).resolve().parents[2]) / READINESS_PATH
    )
    if (
        not _revision(code_revision)
        or any(
            not _fingerprint(x)
            for x in (host_fingerprint, account_fingerprint, credential_fingerprint)
        )
        or type(owner_identity) is not str
        or not owner_identity.strip()
    ):
        _fail()
    issued, expires = _parse_time(issued_at), _parse_time(expires_at)
    if (
        type(maximum_execution_window_seconds) is not int
        or not 0 < maximum_execution_window_seconds <= 86400
        or not issued < expires
        or (expires - issued).total_seconds() > maximum_execution_window_seconds
    ):
        _fail()
    nonce = token_bytes(24).hex()
    authorization_id = (
        "task12b-v2-auth-"
        + hashlib.sha256(token_bytes(16) + nonce.encode()).hexdigest()[:32]
    )
    mapping = build_private_randomization(
        authorization_id, nonce, token_bytes=token_bytes, shuffle=shuffle
    )
    _mapping_semantics(mapping)
    mapping_envelope = {"mapping": mapping, "mapping_digest": sha256_canonical(mapping)}
    contract = load_contract()[0]
    scheduled_units = build_schedule(contract)
    request_bodies = render_unit_requests(contract, scheduled_units)
    request_hashes = ["sha256:" + unit.request_sha256 for unit in scheduled_units]
    static_upper_bounds = [format(_static_upper(body), "f") for body in request_bodies]
    calculated_ceiling = sum(
        (Decimal(value) for value in static_upper_bounds), Decimal(0)
    )
    if format(calculated_ceiling, ".6f") != CONSERVATIVE_EXECUTION_CEILING:
        _fail()
    candidate = {
        "version": CANDIDATE_VERSION,
        "authorization_id": authorization_id,
        "nonce": nonce,
        "issued_at": issued_at,
        "expires_at": expires_at,
        "maximum_execution_window_seconds": maximum_execution_window_seconds,
        "contract_lineage": CONTRACT_LINEAGE,
        "contract_hash": CONTRACT_HASH,
        "readiness_manifest_hash": READINESS_HASH,
        "readiness_manifest": readiness,
        "readiness_manifest_bytes_sha256": _sha256_bytes(readiness_raw),
        "execution_code_revision": code_revision,
        "scheduled_units": [u.unit_id for u in scheduled_units],
        "ordered_request_hashes": request_hashes,
        "static_upper_bounds": static_upper_bounds,
        "execution_order_commitment": sha256_canonical(
            {
                "execution_blocks": mapping["execution_blocks"],
                "execution_order": mapping["execution_order"],
            }
        ),
        "assessment_mapping_commitment": sha256_canonical(
            {
                "entries": mapping["entries"],
                "assessment_order": mapping["assessment_order"],
            }
        ),
        "private_mapping_commitment": mapping_envelope["mapping_digest"],
        "model": PINNED_MODEL,
        "openai_version": PINNED_OPENAI_VERSION,
        "httpx_version": PINNED_HTTPX_VERSION,
        "provider_settings": PINNED_PROVIDER_SETTINGS,
        "prices": PINNED_PRICES,
        "calculated_static_ceiling": format(calculated_ceiling, ".6f"),
        "conservative_execution_ceiling": CONSERVATIVE_EXECUTION_CEILING,
        "hard_owner_cap": OWNER_CAP,
        "no_retry_policy": NO_RETRY_POLICY,
        "approved_scheduled_unit_count": 45,
        "host_fingerprint": host_fingerprint,
        "account_fingerprint": account_fingerprint,
        "credential_fingerprint": credential_fingerprint,
        "owner_identity": owner_identity,
        "candidate_grants_network_authority": False,
    }
    envelope = {"candidate": candidate, "candidate_sha256": sha256_canonical(candidate)}
    write_test_private_record(mapping_path, mapping_envelope)
    write_test_private_record(candidate_path, envelope)
    return envelope


def verify_candidate_v2(path: Path, code_revision: str) -> Dict[str, Any]:
    envelope = read_private_record(path)
    if (
        set(envelope) != {"candidate", "candidate_sha256"}
        or type(envelope["candidate"]) is not dict
        or envelope["candidate_sha256"] != sha256_canonical(envelope["candidate"])
    ):
        _fail()
    candidate = envelope["candidate"]
    readiness, readiness_raw = load_readiness_manifest(
        (Path(__file__).resolve().parents[2]) / READINESS_PATH
    )
    contract = load_contract()[0]
    units = build_schedule(contract)
    bodies = render_unit_requests(contract, units)
    static_upper_bounds = [format(_static_upper(body), "f") for body in bodies]
    calculated_ceiling = sum(
        (Decimal(value) for value in static_upper_bounds), Decimal(0)
    )
    expected = {
        "version": CANDIDATE_VERSION,
        "contract_lineage": CONTRACT_LINEAGE,
        "contract_hash": CONTRACT_HASH,
        "readiness_manifest_hash": READINESS_HASH,
        "readiness_manifest": readiness,
        "readiness_manifest_bytes_sha256": _sha256_bytes(readiness_raw),
        "execution_code_revision": code_revision,
        "model": PINNED_MODEL,
        "openai_version": PINNED_OPENAI_VERSION,
        "httpx_version": PINNED_HTTPX_VERSION,
        "provider_settings": PINNED_PROVIDER_SETTINGS,
        "prices": PINNED_PRICES,
        "ordered_request_hashes": ["sha256:" + unit.request_sha256 for unit in units],
        "static_upper_bounds": static_upper_bounds,
        "calculated_static_ceiling": format(calculated_ceiling, ".6f"),
        "conservative_execution_ceiling": CONSERVATIVE_EXECUTION_CEILING,
        "hard_owner_cap": OWNER_CAP,
        "no_retry_policy": NO_RETRY_POLICY,
        "approved_scheduled_unit_count": 45,
        "candidate_grants_network_authority": False,
    }
    if (
        any(candidate.get(k) != v for k, v in expected.items())
        or candidate.get("scheduled_units") != [u.unit_id for u in units]
        or format(calculated_ceiling, ".6f") != CONSERVATIVE_EXECUTION_CEILING
    ):
        _fail()
    required = set(expected) | {
        "authorization_id",
        "nonce",
        "issued_at",
        "expires_at",
        "maximum_execution_window_seconds",
        "scheduled_units",
        "execution_order_commitment",
        "assessment_mapping_commitment",
        "private_mapping_commitment",
        "host_fingerprint",
        "account_fingerprint",
        "credential_fingerprint",
        "owner_identity",
    }
    if (
        set(candidate) != required
        or not _revision(code_revision)
        or any(
            not _fingerprint(candidate[k])
            for k in (
                "execution_order_commitment",
                "assessment_mapping_commitment",
                "private_mapping_commitment",
                "host_fingerprint",
                "account_fingerprint",
                "credential_fingerprint",
            )
        )
    ):
        _fail()
    issued, expires = _parse_time(candidate["issued_at"]), _parse_time(
        candidate["expires_at"]
    )
    if (
        not issued < expires
        or type(candidate["maximum_execution_window_seconds"]) is not int
        or (expires - issued).total_seconds()
        > candidate["maximum_execution_window_seconds"]
    ):
        _fail()
    return envelope


def verify_mapping_v2(path: Path, candidate: Mapping[str, Any]) -> Dict[str, Any]:
    envelope = read_private_record(path)
    if (
        set(envelope) != {"mapping", "mapping_digest"}
        or type(envelope["mapping"]) is not dict
        or envelope["mapping_digest"] != sha256_canonical(envelope["mapping"])
        or envelope["mapping_digest"] != candidate["private_mapping_commitment"]
    ):
        _fail()
    mapping = envelope["mapping"]
    _mapping_semantics(mapping, candidate)
    if candidate["execution_order_commitment"] != sha256_canonical(
        {
            "execution_blocks": mapping["execution_blocks"],
            "execution_order": mapping["execution_order"],
        }
    ) or candidate["assessment_mapping_commitment"] != sha256_canonical(
        {"entries": mapping["entries"], "assessment_order": mapping["assessment_order"]}
    ):
        _fail()
    return envelope


@dataclass(frozen=True)
class AuthorityContext:
    authorization_id: str
    candidate_digest: str
    approval_digest: str
    code_revision: str
    host_fingerprint: str
    account_fingerprint: str
    credential_fingerprint: str
    private_mapping_commitment: str
    owner_identity: str
    issued_at: str
    approved_at: str
    expires_at: str


def verify_authority_v2(
    candidate_path: Path,
    approval_path: Path,
    mapping_path: Path,
    code_revision: str,
    host_fingerprint: str,
    account_fingerprint: str,
    credential_fingerprint: str,
    now: str,
    *,
    require_live: bool = True,
) -> AuthorityContext:
    envelope = verify_candidate_v2(candidate_path, code_revision)
    candidate = envelope["candidate"]
    verify_mapping_v2(mapping_path, candidate)
    approval_envelope = read_private_record(approval_path)
    if (
        set(approval_envelope) != {"approval", "approval_digest"}
        or type(approval_envelope["approval"]) is not dict
        or approval_envelope["approval_digest"]
        != sha256_canonical(approval_envelope["approval"])
    ):
        _fail()
    approval = approval_envelope["approval"]
    expected = {
        "version": APPROVAL_VERSION,
        "authorization_id": candidate["authorization_id"],
        "candidate_sha256": envelope["candidate_sha256"],
        "owner_echoed_candidate_sha256_out_of_band": True,
        "owner_identity": candidate["owner_identity"],
        "approved_at": approval.get("approved_at"),
        "operational_process_evidence_only": True,
    }
    if approval != expected or (
        host_fingerprint,
        account_fingerprint,
        credential_fingerprint,
    ) != (
        candidate["host_fingerprint"],
        candidate["account_fingerprint"],
        candidate["credential_fingerprint"],
    ):
        _fail()
    issued, approved, expires, observed = map(
        _parse_time,
        (candidate["issued_at"], approval["approved_at"], candidate["expires_at"], now),
    )
    if not issued <= approved <= expires or (
        require_live and not approved <= observed <= expires
    ):
        _fail("authorization_not_live_no_network_attempt")
    return AuthorityContext(
        candidate["authorization_id"],
        envelope["candidate_sha256"],
        approval_envelope["approval_digest"],
        code_revision,
        host_fingerprint,
        account_fingerprint,
        credential_fingerprint,
        candidate["private_mapping_commitment"],
        candidate["owner_identity"],
        candidate["issued_at"],
        approval["approved_at"],
        candidate["expires_at"],
    )


def _identity(unit: ScheduledUnit, context: AuthorityContext) -> Dict[str, Any]:
    return {
        "contract_lineage": CONTRACT_LINEAGE,
        "authorization_id": context.authorization_id,
        "candidate_digest": context.candidate_digest,
        "approval_digest": context.approval_digest,
        "code_revision": context.code_revision,
        "host_fingerprint": context.host_fingerprint,
        "account_fingerprint": context.account_fingerprint,
        "credential_fingerprint": context.credential_fingerprint,
        "private_mapping_commitment": context.private_mapping_commitment,
        "unit_id": unit.unit_id,
        "base_cell_id": unit.base_cell_id,
        "draw_index": unit.draw_index,
        "request_hash": "sha256:" + unit.request_sha256,
    }


def _paths(root: Path, kind: str, unit: ScheduledUnit) -> Path:
    return root / kind / (unit.unit_id + (".raw" if kind == "raw" else ".json"))


def _publish_mirrored(
    global_path: Path, local_path: Path, value: Mapping[str, Any]
) -> None:
    data = canonical_bytes(value) + b"\n"
    _write_private_bytes_no_overwrite(global_path, data)
    _write_private_bytes_no_overwrite(local_path, data)


def publish_unit_claim(
    global_root: Path,
    local_root: Path,
    unit: ScheduledUnit,
    context: AuthorityContext,
    claimed_at: str,
) -> Dict[str, Any]:
    _parse_time(claimed_at)
    claim = {
        "version": CLAIM_VERSION,
        **_identity(unit, context),
        "claimed_at": claimed_at,
        "authority_consumed": True,
    }
    envelope = {"claim": claim, "claim_digest": sha256_canonical(claim)}
    _publish_mirrored(
        _paths(global_root, "claims", unit),
        _paths(local_root / "mirrors", "claims", unit),
        envelope,
    )
    return envelope


def verify_unit_claim(
    path: Path, unit: ScheduledUnit, context: AuthorityContext
) -> Dict[str, Any]:
    value = read_private_record(path)
    claim = value.get("claim")
    if (
        set(value) != {"claim", "claim_digest"}
        or type(claim) is not dict
        or value["claim_digest"] != sha256_canonical(claim)
        or claim
        != {
            "version": CLAIM_VERSION,
            **_identity(unit, context),
            "claimed_at": claim.get("claimed_at"),
            "authority_consumed": True,
        }
    ):
        _fail()
    _parse_time(claim["claimed_at"])
    return value


def _request_metadata(request_hash: str) -> Dict[str, Any]:
    return {
        "method": "POST",
        "scheme": "https",
        "host": "api.openai.com",
        "path": "/v1/responses",
        "request_body_sha256": request_hash,
        "model": PINNED_MODEL,
    }


def _validate_terminal_semantics(
    terminal: Mapping[str, Any],
    *,
    raw_bytes: Optional[bytes],
    expected_provider_visible_evidence: Optional[Mapping[str, Any]] = None,
) -> None:
    kind = terminal["kind"]
    dispatch_invoked = terminal["dispatch_invoked"]
    acceptance = terminal["server_acceptance"]
    metadata = terminal["response_metadata"]
    usage = terminal["usage"]
    actual = terminal["actual_cost"]
    upper = terminal["conservative_cost_upper_bound"]
    evidence = terminal["provider_visible_evidence"]
    structured = terminal["structured_response"]
    has_raw = terminal["raw_sha256"] is not None
    if (
        kind not in {"success", "failure", "invalid_ambiguous"}
        or not v1_execution._valid_dispatch(dispatch_invoked)
        or acceptance not in {"yes", "no", "unknown"}
        or (
            metadata is not None and not v1_execution._valid_response_metadata(metadata)
        )
        or (usage is not None and not v1_execution._valid_usage(usage))
        or any(
            value is not None and not v1_execution._decimal_string(value)
            for value in (actual, upper)
        )
    ):
        _fail()
    if has_raw != (raw_bytes is not None) or (
        has_raw and _sha256_bytes(raw_bytes) != terminal["raw_sha256"]
    ):
        _fail()
    if (has_raw and metadata is None) or (acceptance == "yes" and metadata is None):
        _fail()
    if dispatch_invoked is True and (upper is None or type(evidence) is not dict):
        _fail()
    if (
        expected_provider_visible_evidence is not None
        and dispatch_invoked is True
        and evidence != dict(expected_provider_visible_evidence)
    ):
        _fail()
    if usage is not None:
        expected_actual, expected_upper = v1_execution.usage_costs(usage)
        if (actual, upper) != (expected_actual, expected_upper):
            _fail()
    if kind == "success":
        if (
            dispatch_invoked is not True
            or acceptance != "yes"
            or not has_raw
            or type(structured) is not dict
            or terminal["failure_category"] is not None
            or usage is None
            or type(evidence) is not dict
            or not v1_execution._valid_success_response_metadata(metadata)
        ):
            _fail()
        projection = v1_execution._replay_raw_success(cast(bytes, raw_bytes))
        if (
            structured != projection["structured_response"]
            or usage != projection["usage"]
            or actual != projection["actual_cost"]
            or upper != projection["conservative_cost_upper_bound"]
            or metadata["response_id"] != projection["response_id"]
            or metadata["observed_model"] != projection["observed_model"]
        ):
            _fail()
        return
    if terminal["failure_category"] not in FAILURE_CATEGORIES or structured is not None:
        _fail()
    if kind == "invalid_ambiguous":
        if (
            any(
                value is not None
                for value in (metadata, usage, actual, evidence, raw_bytes)
            )
            or not v1_execution._decimal_string(upper)
            or Decimal(cast(str, upper)) <= 0
            or dispatch_invoked != "unknown"
            or acceptance != "unknown"
        ):
            _fail()
        return
    if has_raw and (dispatch_invoked is not True or acceptance != "yes"):
        _fail()
    if (
        not has_raw
        and metadata is None
        and terminal["failure_category"]
        not in {
            "transport_error",
            "secret_detected",
            "preflight_rejected",
        }
    ):
        _fail()
    if dispatch_invoked is False and any(
        value is not None
        for value in (metadata, usage, actual, upper, evidence, raw_bytes)
    ):
        _fail()


def publish_unit_terminal(
    global_root: Path,
    local_root: Path,
    unit: ScheduledUnit,
    context: AuthorityContext,
    *,
    kind: str,
    dispatch_invoked: Any,
    server_acceptance: str,
    recorded_at: str,
    failure_category: Optional[str] = None,
    provider_visible_evidence: Optional[Mapping[str, Any]] = None,
    structured_response: Optional[Mapping[str, Any]] = None,
    raw_bytes: Optional[bytes] = None,
    usage: Optional[Mapping[str, Any]] = None,
    actual_cost: Optional[str] = None,
    conservative_cost_upper_bound: Optional[str] = None,
    response_metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    claim = verify_unit_claim(_paths(global_root, "claims", unit), unit, context)
    _parse_time(recorded_at)
    raw_hash = _sha256_bytes(raw_bytes) if raw_bytes is not None else None
    terminal = {
        "version": TERMINAL_VERSION,
        **_identity(unit, context),
        "claim_digest": claim["claim_digest"],
        "authority_consumed": True,
        "kind": kind,
        "dispatch_invoked": dispatch_invoked,
        "server_acceptance": server_acceptance,
        "failure_category": failure_category,
        "request_metadata": _request_metadata("sha256:" + unit.request_sha256),
        "response_metadata": (
            dict(response_metadata) if response_metadata is not None else None
        ),
        "provider_visible_evidence": (
            dict(provider_visible_evidence)
            if provider_visible_evidence is not None
            else None
        ),
        "structured_response": (
            dict(structured_response) if structured_response is not None else None
        ),
        "raw_sha256": raw_hash,
        "usage": dict(usage) if usage is not None else None,
        "actual_cost": actual_cost,
        "conservative_cost_upper_bound": conservative_cost_upper_bound,
        "recorded_at": recorded_at,
    }
    _validate_terminal_semantics(terminal, raw_bytes=raw_bytes)
    if _parse_time(recorded_at) < _parse_time(claim["claim"]["claimed_at"]):
        _fail()
    if raw_bytes is not None:
        _write_private_bytes_no_overwrite(_paths(global_root, "raw", unit), raw_bytes)
    envelope = {"terminal": terminal, "terminal_digest": sha256_canonical(terminal)}
    terminal_data = canonical_bytes(envelope) + b"\n"
    _write_private_bytes_no_overwrite(
        _paths(global_root, "terminals", unit), terminal_data
    )
    index_payload = {
        "version": TERMINAL_INDEX_VERSION,
        "contract_lineage": CONTRACT_LINEAGE,
        "unit_id": unit.unit_id,
        "terminal_digest": envelope["terminal_digest"],
        "terminal_record": "terminals/" + unit.unit_id + ".json",
    }
    index = {
        "terminal_index": index_payload,
        "terminal_index_digest": sha256_canonical(index_payload),
    }
    _write_private_bytes_no_overwrite(
        _paths(global_root, "terminal-index", unit),
        canonical_bytes(index) + b"\n",
    )
    _write_private_bytes_no_overwrite(
        _paths(local_root / "mirrors", "terminals", unit), terminal_data
    )
    return envelope


def verify_unit_terminal(
    global_root: Path,
    unit: ScheduledUnit,
    context: AuthorityContext,
    expected_provider_visible_evidence: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    value = read_private_record(_paths(global_root, "terminals", unit))
    terminal = value.get("terminal")
    if (
        set(value) != {"terminal", "terminal_digest"}
        or type(terminal) is not dict
        or value["terminal_digest"] != sha256_canonical(terminal)
    ):
        _fail()
    identity = _identity(unit, context)
    required = {
        "version",
        *identity,
        "claim_digest",
        "authority_consumed",
        "kind",
        "dispatch_invoked",
        "server_acceptance",
        "failure_category",
        "request_metadata",
        "response_metadata",
        "provider_visible_evidence",
        "structured_response",
        "raw_sha256",
        "usage",
        "actual_cost",
        "conservative_cost_upper_bound",
        "recorded_at",
    }
    if (
        set(terminal) != required
        or terminal["version"] != TERMINAL_VERSION
        or terminal["authority_consumed"] is not True
        or any(terminal.get(k) != v for k, v in identity.items())
        or terminal["request_metadata"] != _request_metadata(identity["request_hash"])
    ):
        _fail()
    claim = verify_unit_claim(_paths(global_root, "claims", unit), unit, context)
    if terminal.get("claim_digest") != claim["claim_digest"]:
        _fail()
    raw_path = _paths(global_root, "raw", unit)
    raw_exists = _exists(raw_path)
    raw_bytes = _read_private_bytes(raw_path) if raw_exists else None
    _validate_terminal_semantics(
        terminal,
        raw_bytes=raw_bytes,
        expected_provider_visible_evidence=expected_provider_visible_evidence,
    )
    if _parse_time(terminal["recorded_at"]) < _parse_time(claim["claim"]["claimed_at"]):
        _fail()
    index_payload = {
        "version": TERMINAL_INDEX_VERSION,
        "contract_lineage": CONTRACT_LINEAGE,
        "unit_id": unit.unit_id,
        "terminal_digest": value["terminal_digest"],
        "terminal_record": "terminals/" + unit.unit_id + ".json",
    }
    if read_private_record(_paths(global_root, "terminal-index", unit)) != {
        "terminal_index": index_payload,
        "terminal_index_digest": sha256_canonical(index_payload),
    }:
        _fail()
    return value


def classify_unit_state(
    global_root: Path, local_root: Path, unit: ScheduledUnit, context: AuthorityContext
) -> str:
    gc, lc = _paths(global_root, "claims", unit), _paths(
        local_root / "mirrors", "claims", unit
    )
    gt, lt = _paths(global_root, "terminals", unit), _paths(
        local_root / "mirrors", "terminals", unit
    )
    index = _paths(global_root, "terminal-index", unit)
    raw = _paths(global_root, "raw", unit)
    flags = tuple(_exists(x) for x in (gc, lc, gt, lt, index, raw))
    if flags == (False, False, False, False, False, False):
        return "pending"
    if flags == (True, True, False, False, False, False):
        if _read_private_bytes(gc) != _read_private_bytes(lc):
            _fail()
        verify_unit_claim(gc, unit, context)
        return "blocked_orphan_claim"
    if flags[:5] == (True, True, True, True, True):
        if _read_private_bytes(gc) != _read_private_bytes(lc) or _read_private_bytes(
            gt
        ) != _read_private_bytes(lt):
            _fail()
        terminal = verify_unit_terminal(global_root, unit, context)["terminal"]
        if flags[5] != (terminal["raw_sha256"] is not None):
            _fail()
        return "terminal"
    _fail("conflicting_evidence_no_network_attempt")
    raise AssertionError


@contextmanager
def machine_global_run_lock(global_root: Path) -> Iterator[None]:
    _ensure_private_directory(global_root)
    path = global_root / "run.lock"
    if not _exists(path):
        try:
            _write_private_bytes_no_overwrite(
                path, b"task12b-v2 machine-global run lock\n"
            )
        except FileExistsError:
            pass
    parent_fd, parent_metadata = _open_parent(path)
    descriptor: Optional[int] = None
    try:
        descriptor = os.open(
            path.name, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd
        )
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_nlink != 1
        ):
            _fail()
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            _fail("machine_global_run_lock_unavailable")
        _revalidate_parent(path, parent_metadata)
        yield
    finally:
        if descriptor is not None:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
        os.close(parent_fd)


def _evidence_by_cell(repo_root: Path) -> Dict[str, Dict[str, Any]]:
    contract, _ = load_contract(repo_root / CONTRACT_PATH)
    from .context_sensitivity_calibration import load_contract as load_v1_contract

    predecessor, _ = load_v1_contract(
        repo_root / contract["lineage"]["predecessor_contract_path"]
    )
    return {
        cell["cell_id"]: {
            "task": scenario["task_prompt"],
            "timestamped_evidence": [
                "%d. [%s] %s" % (i, item["observed_at"], item["content"])
                for i, item in enumerate(cell["evidence"], 1)
            ],
        }
        for scenario in predecessor["scenarios"]
        for cell in scenario["cells"]
    }


def _static_upper(body: Mapping[str, Any]) -> Decimal:
    return (
        Decimal(len(canonical_bytes(body)) + 1024) * Decimal("2.50")
        + Decimal(2048) * Decimal("12.00")
    ) / Decimal(1_000_000)


def execute_authorized_45_unit_manifest(
    repo_root: Path,
    global_root: Path,
    local_root: Path,
    candidate_path: Path,
    approval_path: Path,
    mapping_path: Path,
    *,
    api_key: str,
    client_factory: Callable[[str], Any],
    now: str,
    code_revision: str,
    host_fingerprint_value: str,
    account_fingerprint_value: str,
    credential_fingerprint_value: str,
    dispatch: Callable[[Any, Mapping[str, Any], str], TransportResult] = dispatch_once,
    clock: Optional[Callable[[], str]] = None,
) -> Dict[str, Any]:
    context = verify_authority_v2(
        candidate_path,
        approval_path,
        mapping_path,
        code_revision,
        host_fingerprint_value,
        account_fingerprint_value,
        credential_fingerprint_value,
        now,
    )
    candidate = verify_candidate_v2(candidate_path, code_revision)["candidate"]
    mapping = verify_mapping_v2(mapping_path, candidate)["mapping"]
    units_by_id = {
        u.unit_id: u
        for u in build_schedule(load_contract(repo_root / CONTRACT_PATH)[0])
    }
    units = [units_by_id[x] for x in mapping["execution_order"]]
    requests = dict(
        zip(
            (
                u.unit_id
                for u in build_schedule(load_contract(repo_root / CONTRACT_PATH)[0])
            ),
            render_unit_requests(load_contract(repo_root / CONTRACT_PATH)[0]),
        )
    )
    evidence = _evidence_by_cell(repo_root)
    current_clock = clock or (lambda: now)
    with machine_global_run_lock(global_root):
        states = [
            classify_unit_state(global_root, local_root, unit, context)
            for unit in units
        ]
        if "blocked_orphan_claim" in states:
            _fail("blocked_orphan_claim_no_network_attempt")
        pending = [u for u, state in zip(units, states) if state == "pending"]
        if not pending:
            return _run_summary(
                global_root, local_root, units, context, 0, all_terminal=True
            )
        accepted = sum(
            (
                Decimal(
                    verify_unit_terminal(global_root, u, context)["terminal"].get(
                        "conservative_cost_upper_bound"
                    )
                    or "0"
                )
                for u, state in zip(units, states)
                if state == "terminal"
            ),
            Decimal(0),
        )
        remaining = sum(
            (_static_upper(requests[u.unit_id]) for u in pending), Decimal(0)
        )
        if accepted + remaining > Decimal(OWNER_CAP):
            _fail("budget_bound_violation_no_network_attempt")
        client = client_factory(api_key)
        dispatches = 0
        try:
            for index, unit in enumerate(pending):
                observed = current_clock()
                if (
                    not _parse_time(context.approved_at)
                    <= _parse_time(observed)
                    <= _parse_time(context.expires_at)
                ):
                    break
                remaining_before_claim = sum(
                    (_static_upper(requests[x.unit_id]) for x in pending[index:]),
                    Decimal(0),
                )
                if accepted + remaining_before_claim > Decimal(OWNER_CAP):
                    break
                body = requests[unit.unit_id]
                request_hash = "sha256:" + unit.request_sha256
                if _sha256_bytes(canonical_bytes(body)) != request_hash:
                    _fail()
                publish_unit_claim(global_root, local_root, unit, context, observed)
                dispatches += 1
                try:
                    result = dispatch(client, body, request_hash)
                    terminal_at = current_clock()
                    if api_key.encode() in result.raw_bytes:
                        raise ExecutionFailure("secret_detected")
                    upper = result.conservative_cost_upper_bound
                    publish_unit_terminal(
                        global_root,
                        local_root,
                        unit,
                        context,
                        kind="success",
                        dispatch_invoked=True,
                        server_acceptance="yes",
                        recorded_at=terminal_at,
                        provider_visible_evidence=evidence[unit.base_cell_id],
                        structured_response=result.structured_response,
                        raw_bytes=result.raw_bytes,
                        usage=result.usage,
                        actual_cost=result.actual_cost,
                        conservative_cost_upper_bound=upper,
                        response_metadata=result.response_metadata,
                    )
                except (ExecutionFailure, V1TransportFailure) as failure:
                    terminal_at = current_clock()
                    raw_bytes = getattr(failure, "raw_bytes", None)
                    if raw_bytes is not None and api_key.encode() in raw_bytes:
                        raw_bytes = None
                        category = "secret_detected"
                    else:
                        category = getattr(failure, "category", "transport_error")
                    upper = getattr(
                        failure, "conservative_cost_upper_bound", None
                    ) or format(_static_upper(body), "f")
                    publish_unit_terminal(
                        global_root,
                        local_root,
                        unit,
                        context,
                        kind="failure",
                        dispatch_invoked=True,
                        server_acceptance=(
                            "yes"
                            if getattr(failure, "response_metadata", None)
                            else "unknown"
                        ),
                        recorded_at=terminal_at,
                        failure_category=category,
                        provider_visible_evidence=evidence[unit.base_cell_id],
                        raw_bytes=raw_bytes,
                        usage=getattr(failure, "usage", None),
                        actual_cost=getattr(failure, "actual_cost", None),
                        conservative_cost_upper_bound=upper,
                        response_metadata=getattr(failure, "response_metadata", None),
                    )
                accepted += Decimal(upper)
                remaining = sum(
                    (_static_upper(requests[x.unit_id]) for x in pending[index + 1 :]),
                    Decimal(0),
                )
                if accepted + remaining > Decimal(OWNER_CAP):
                    break
        finally:
            try:
                client.close()
            except Exception:
                _fail("client_close_failure")
        return _run_summary(global_root, local_root, units, context, dispatches)


def _run_summary(
    global_root: Path,
    local_root: Path,
    units: Sequence[ScheduledUnit],
    context: AuthorityContext,
    dispatches: int,
    *,
    all_terminal: bool = False,
) -> Dict[str, Any]:
    success = failure = pending = 0
    for unit in units:
        state = classify_unit_state(global_root, local_root, unit, context)
        if state == "pending":
            pending += 1
        elif state != "terminal":
            _fail()
        elif (
            verify_unit_terminal(global_root, unit, context)["terminal"]["kind"]
            == "success"
        ):
            success += 1
        else:
            failure += 1
    return {
        "status": (
            "all_terminal"
            if all_terminal
            else ("run_complete" if pending == 0 else "run_incomplete")
        ),
        "dispatches": dispatches,
        "successes": success,
        "failures": failure,
        "pending": pending,
    }


def build_blind_assessment_v2(
    repo_root: Path,
    global_root: Path,
    local_root: Path,
    candidate_path: Path,
    mapping_path: Path,
    code_revision: str,
    host_fingerprint: str,
    account_fingerprint: str,
    credential_fingerprint: str,
    approval_path: Optional[Path] = None,
) -> Dict[str, Any]:
    if approval_path is None:
        _fail()
    context = verify_authority_v2(
        candidate_path,
        approval_path,
        mapping_path,
        code_revision,
        host_fingerprint,
        account_fingerprint,
        credential_fingerprint,
        "1970-01-01T00:00:00.000000Z",
        require_live=False,
    )
    candidate = verify_candidate_v2(candidate_path, code_revision)["candidate"]
    mapping = verify_mapping_v2(mapping_path, candidate)["mapping"]
    aliases = {x["unit_id"]: x["assessment_alias"] for x in mapping["entries"]}
    units = {
        u.unit_id: u
        for u in build_schedule(load_contract(repo_root / CONTRACT_PATH)[0])
    }
    bodies = dict(
        zip(units, render_unit_requests(load_contract(repo_root / CONTRACT_PATH)[0]))
    )
    material = _contract_assessment_material_v2(repo_root)
    expected_evidence = _evidence_by_cell(repo_root)
    items_by_alias = {}
    for unit_id, unit in units.items():
        if classify_unit_state(global_root, local_root, unit, context) != "terminal":
            _fail()
        terminal_value = verify_unit_terminal(
            global_root, unit, context, expected_evidence[unit.base_cell_id]
        )
        terminal = terminal_value["terminal"]

        alias = aliases[unit_id]
        evidence = terminal["provider_visible_evidence"] or {
            "task": material[unit.base_cell_id]["task"],
            "timestamped_evidence": [],
        }
        items_by_alias[alias] = {
            "assessment_alias": alias,
            "task": evidence["task"],
            "provider_visible_timestamped_evidence": evidence["timestamped_evidence"],
            "structured_response": terminal["structured_response"],
            "status": terminal["kind"],
            "criteria": material[unit.base_cell_id]["criteria"],
            "adjudication_rules": material[unit.base_cell_id]["adjudication_rules"],
            "critical_findings": material[unit.base_cell_id]["critical_findings"],
        }
        if (
            bodies[unit_id]["input"]
            != "Task:\n%s\n\nEvidence:\n%s"
            % (evidence["task"], "\n".join(evidence["timestamped_evidence"]))
            and terminal["dispatch_invoked"] is True
        ):
            _fail()
    return {
        "version": BLIND_EXPORT_VERSION,
        "assessment_ready": all(
            items_by_alias[x]["status"] == "success"
            for x in mapping["assessment_order"]
        ),
        "assessments": [items_by_alias[x] for x in mapping["assessment_order"]],
    }


def _contract_assessment_material_v2(repo_root: Path) -> Dict[str, Dict[str, Any]]:
    contract, _ = load_contract(repo_root / CONTRACT_PATH)
    from .context_sensitivity_calibration import load_contract as load_v1_contract

    predecessor, _ = load_v1_contract(
        repo_root / contract["lineage"]["predecessor_contract_path"]
    )
    return {
        cell["cell_id"]: {
            "task": scenario["task_prompt"],
            "criteria": scenario["rubric"]["criteria"],
            "adjudication_rules": scenario["rubric"]["adjudication_rules"],
            "critical_findings": scenario["rubric"]["critical_findings"],
        }
        for scenario in predecessor["scenarios"]
        for cell in scenario["cells"]
    }


def publish_annotation_lock(
    path: Path,
    annotations: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
    mapping: Mapping[str, Any],
    blind_assessment: Mapping[str, Any],
) -> Dict[str, Any]:
    _mapping_semantics(mapping, candidate)
    if (
        set(blind_assessment) != {"version", "assessment_ready", "assessments"}
        or blind_assessment.get("version") != BLIND_EXPORT_VERSION
        or type(blind_assessment.get("assessments")) is not list
    ):
        _fail()
    assessments = blind_assessment["assessments"]
    exact_assessment_fields = {
        "assessment_alias",
        "task",
        "provider_visible_timestamped_evidence",
        "structured_response",
        "status",
        "criteria",
        "adjudication_rules",
        "critical_findings",
    }
    if len(assessments) != 45 or any(
        type(x) is not dict or set(x) != exact_assessment_fields for x in assessments
    ):
        _fail()
    expected_order = mapping["assessment_order"]
    if [x["assessment_alias"] for x in assessments] != expected_order:
        _fail()
    success_by_alias = {
        x["assessment_alias"]: x for x in assessments if x["status"] == "success"
    }
    if any(
        x["status"] not in {"success", "failure", "invalid_ambiguous"}
        for x in assessments
    ):
        _fail()
    status_values = set(load_contract()[0]["assessment"]["criterion_status_scores"])
    if type(annotations) is not list or any(
        type(x) is not dict
        or set(x) != {"assessment_alias", "criteria", "critical_finding"}
        or re.fullmatch(r"assessment-[0-9a-f]{32}", x["assessment_alias"]) is None
        or type(x["criteria"]) is not dict
        or type(x["critical_finding"]) is not bool
        for x in annotations
    ):
        _fail()
    aliases = [x["assessment_alias"] for x in annotations]
    if len(aliases) != len(set(aliases)) or set(aliases) != set(success_by_alias):
        _fail()
    for annotation in annotations:
        assessment = success_by_alias[annotation["assessment_alias"]]
        criterion_ids = {x["criterion_id"] for x in assessment["criteria"]}
        if set(annotation["criteria"]) != criterion_ids or any(
            type(status) is not str or status not in status_values
            for status in annotation["criteria"].values()
        ):
            _fail()
    payload = {
        "version": ANNOTATION_LOCK_VERSION,
        "contract_lineage": CONTRACT_LINEAGE,
        "candidate_digest": sha256_canonical(candidate),
        "private_mapping_commitment": candidate["private_mapping_commitment"],
        "blind_assessment_digest": sha256_canonical(blind_assessment),
        "annotations": [dict(x) for x in annotations],
        "immutable_no_overwrite": True,
    }
    envelope = {
        "annotation_lock": payload,
        "annotation_lock_digest": sha256_canonical(payload),
    }
    write_test_private_record(path, envelope)
    return envelope


def verify_annotation_lock(
    path: Path,
    candidate: Mapping[str, Any],
    mapping: Mapping[str, Any],
    blind_assessment: Mapping[str, Any],
) -> Dict[str, Any]:
    value = read_private_record(path)
    payload = value.get("annotation_lock")
    if (
        set(value) != {"annotation_lock", "annotation_lock_digest"}
        or type(payload) is not dict
        or value["annotation_lock_digest"] != sha256_canonical(payload)
        or payload.get("candidate_digest") != sha256_canonical(candidate)
        or payload.get("private_mapping_commitment")
        != candidate["private_mapping_commitment"]
        or payload.get("blind_assessment_digest") != sha256_canonical(blind_assessment)
        or payload.get("immutable_no_overwrite") is not True
    ):
        _fail()
    _validate_annotation_payload(
        payload.get("annotations"), candidate, mapping, blind_assessment
    )
    return value


def _validate_annotation_payload(
    annotations: Any,
    candidate: Mapping[str, Any],
    mapping: Mapping[str, Any],
    blind_assessment: Mapping[str, Any],
) -> None:
    # Reuse the publication validator against a disposable path without writing.
    assessments = blind_assessment.get("assessments")
    if type(assessments) is not list:
        _fail()
    success = {
        x["assessment_alias"]: x
        for x in assessments
        if type(x) is dict and x.get("status") == "success"
    }
    statuses = set(load_contract()[0]["assessment"]["criterion_status_scores"])
    if (
        type(annotations) is not list
        or {x.get("assessment_alias") for x in annotations if type(x) is dict}
        != set(success)
        or len(annotations) != len(success)
    ):
        _fail()
    for item in annotations:
        if (
            type(item) is not dict
            or set(item) != {"assessment_alias", "criteria", "critical_finding"}
            or type(item["criteria"]) is not dict
            or type(item["critical_finding"]) is not bool
        ):
            _fail()
        criterion_ids = {
            x["criterion_id"] for x in success[item["assessment_alias"]]["criteria"]
        }
        if set(item["criteria"]) != criterion_ids or any(
            type(x) is not str or x not in statuses for x in item["criteria"].values()
        ):
            _fail()


def resolve_locked_annotations_v2(
    lock_path: Path,
    mapping_path: Path,
    candidate: Mapping[str, Any],
    blind_assessment: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    mapping = verify_mapping_v2(mapping_path, candidate)["mapping"]
    lock = verify_annotation_lock(lock_path, candidate, mapping, blind_assessment)[
        "annotation_lock"
    ]
    alias_to_unit = {
        item["assessment_alias"]: item["unit_id"] for item in mapping["entries"]
    }
    if any(
        item["assessment_alias"] not in alias_to_unit for item in lock["annotations"]
    ):
        _fail()
    return [
        {
            "unit_id": alias_to_unit[item["assessment_alias"]],
            "criteria": item["criteria"],
            "critical_finding": item["critical_finding"],
            "locked": True,
            "evidence_coherent": True,
        }
        for item in lock["annotations"]
    ]


def score_locked_annotations_v2(
    repo_root: Path,
    global_root: Path,
    local_root: Path,
    candidate_path: Path,
    approval_path: Path,
    mapping_path: Path,
    annotation_lock_path: Path,
    context: AuthorityContext,
) -> Dict[str, Any]:
    """Verify complete terminal/lock evidence, resolve aliases, and invoke the v2 scorer."""
    from .context_sensitivity_replication_v2 import score_replication

    candidate_envelope = verify_candidate_v2(candidate_path, context.code_revision)
    candidate = candidate_envelope["candidate"]
    if candidate_envelope["candidate_sha256"] != context.candidate_digest:
        _fail()
    mapping = verify_mapping_v2(mapping_path, candidate)["mapping"]
    blind_assessment = build_blind_assessment_v2(
        repo_root,
        global_root,
        local_root,
        candidate_path,
        mapping_path,
        context.code_revision,
        context.host_fingerprint,
        context.account_fingerprint,
        context.credential_fingerprint,
        approval_path,
    )
    aliases = {item["unit_id"]: item["assessment_alias"] for item in mapping["entries"]}
    units = build_schedule(load_contract()[0])
    outcomes = []
    valid_aliases = set()
    for unit in units:
        if classify_unit_state(global_root, local_root, unit, context) != "terminal":
            _fail("incomplete_execution_cannot_score")
        terminal = verify_unit_terminal(global_root, unit, context)["terminal"]
        schema_valid = terminal["kind"] == "success"
        outcomes.append({"unit_id": unit.unit_id, "schema_valid": schema_valid})
        if schema_valid:
            valid_aliases.add(aliases[unit.unit_id])
    lock = verify_annotation_lock(
        annotation_lock_path, candidate, mapping, blind_assessment
    )["annotation_lock"]
    locked_aliases = {item["assessment_alias"] for item in lock["annotations"]}
    if locked_aliases != valid_aliases:
        _fail("incomplete_annotation_lock_cannot_score")
    resolved = resolve_locked_annotations_v2(
        annotation_lock_path, mapping_path, candidate, blind_assessment
    )
    return score_replication(
        load_contract()[0],
        outcomes,
        resolved,
        annotation_lock_verified=True,
    )


def finalize_orphan_unit_offline(
    global_root: Path,
    local_root: Path,
    unit: ScheduledUnit,
    context: AuthorityContext,
    authorization_path: Path,
    recorded_at: str,
    *,
    _lock_held: bool = False,
) -> Dict[str, Any]:
    if (
        classify_unit_state(global_root, local_root, unit, context)
        != "blocked_orphan_claim"
    ):
        _fail()
    claim = verify_unit_claim(_paths(global_root, "claims", unit), unit, context)
    envelope = read_private_record(authorization_path)
    payload = envelope.get("finalization_authorization")
    expected = {
        "version": ORPHAN_FINALIZATION_VERSION,
        "contract_lineage": CONTRACT_LINEAGE,
        "authorization_id": context.authorization_id,
        "candidate_digest": context.candidate_digest,
        "approval_digest": context.approval_digest,
        "unit_id": unit.unit_id,
        "orphan_claim_digest": claim["claim_digest"],
        "owner_echoed_orphan_claim_digest_out_of_band": True,
        "owner_confirmed_no_process_remains": True,
        "confirmation_process": (
            payload.get("confirmation_process") if type(payload) is dict else None
        ),
        "owner_identity": context.owner_identity,
        "authorized_at": (
            payload.get("authorized_at") if type(payload) is dict else None
        ),
        "operational_process_evidence_only": True,
    }
    if (
        set(envelope)
        != {"finalization_authorization", "finalization_authorization_digest"}
        or payload != expected
        or envelope["finalization_authorization_digest"] != sha256_canonical(payload)
        or type(payload["confirmation_process"]) is not str
        or len(payload["confirmation_process"].strip()) < 20
    ):
        _fail()
    authorized = _parse_time(payload["authorized_at"])
    recorded = _parse_time(recorded_at)
    if not (
        _parse_time(context.issued_at)
        <= _parse_time(context.approved_at)
        <= _parse_time(claim["claim"]["claimed_at"])
        <= authorized
        <= recorded
    ):
        _fail()
    body = render_unit_requests(load_contract()[0], [unit])[0]
    static_upper = format(_static_upper(body), "f")

    def publish() -> Dict[str, Any]:
        if (
            classify_unit_state(global_root, local_root, unit, context)
            != "blocked_orphan_claim"
        ):
            _fail()
        return publish_unit_terminal(
            global_root,
            local_root,
            unit,
            context,
            kind="invalid_ambiguous",
            dispatch_invoked="unknown",
            server_acceptance="unknown",
            recorded_at=recorded_at,
            failure_category="invalid_ambiguous_orphan_claim",
            conservative_cost_upper_bound=static_upper,
        )

    if _lock_held:
        return publish()
    with machine_global_run_lock(global_root):
        return publish()


def build_dry_run_summary(repo_root: Path = Path(".")) -> Dict[str, Any]:
    load_readiness_manifest(repo_root / READINESS_PATH)
    contract, _ = load_contract(repo_root / CONTRACT_PATH)
    units = build_schedule(contract)
    return {
        "mode": "offline_dry_run",
        "contract_lineage": CONTRACT_LINEAGE,
        "contract_sha256": CONTRACT_HASH,
        "readiness_manifest_sha256": READINESS_HASH,
        "network_requests_authorized": 0,
        "candidate_grants_network_authority": False,
        "scheduled_unit_count": len(units),
        "base_cell_count": len({u.base_cell_id for u in units}),
        "unique_request_hash_count": len({u.request_sha256 for u in units}),
        "conservative_execution_ceiling": CONSERVATIVE_EXECUTION_CEILING,
        "hard_owner_cap": OWNER_CAP,
    }


def _canonical_local_root(repo_root: Path) -> Path:
    return repo_root / LOCAL_OUTPUT_PATH


def _historical_context(
    candidate_path: Path,
    approval_path: Path,
    mapping_path: Path,
    revision: str,
    credential_digest: str,
) -> AuthorityContext:
    return verify_authority_v2(
        candidate_path,
        approval_path,
        mapping_path,
        revision,
        v1_execution.host_fingerprint(),
        v1_execution.account_fingerprint(),
        credential_digest,
        "1970-01-01T00:00:00.000000Z",
        require_live=False,
    )


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    _repo_root: Optional[Path] = None,
    _global_root: Optional[Path] = None,
    _local_root: Optional[Path] = None,
    _api_key: Optional[str] = None,
    _client_factory: Callable[[str], Any] = v1_execution._create_live_client,
    _now: Optional[str] = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Task 12b v2 offline-first execution readiness"
    )
    commands = parser.add_mutually_exclusive_group()
    commands.add_argument("--prepare-authorization-candidate", action="store_true")
    commands.add_argument("--verify-authority", action="store_true")
    commands.add_argument("--execute-authorized-45-unit-manifest", action="store_true")
    commands.add_argument("--export-blind-assessment", action="store_true")
    commands.add_argument("--lock-annotations", action="store_true")
    commands.add_argument("--score-locked-annotations", action="store_true")
    commands.add_argument("--finalize-authorized-orphan", action="store_true")
    parser.add_argument("--owner-identity")
    parser.add_argument("--expires-at")
    parser.add_argument("--credential-fingerprint")
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--orphan-unit")
    args = parser.parse_args(argv)
    repo_root = (_repo_root or Path.cwd()).resolve()
    local_root = _local_root or _canonical_local_root(repo_root)
    global_root = _global_root or default_authority_directory_v2()
    try:
        selected = any(
            (
                args.prepare_authorization_candidate,
                args.verify_authority,
                args.execute_authorized_45_unit_manifest,
                args.export_blind_assessment,
                args.lock_annotations,
                args.score_locked_annotations,
                args.finalize_authorized_orphan,
            )
        )
        if not selected:
            build_dry_run_summary(repo_root)
            print(
                "task12b-v2 status=dry-run network_attempts=0 credential_readiness=not_checked scheduled_units=45"
            )
            return 0
        now = _now or v1_execution._utc_now()
        candidate_path = local_root / "authorization-candidate.json"
        approval_path = local_root / "owner-approval.json"
        mapping_path = local_root / "private-randomization.json"
        revision = v1_execution.repository_preflight(repo_root)
        if args.prepare_authorization_candidate:
            if (
                not args.owner_identity
                or not args.expires_at
                or not _fingerprint(args.credential_fingerprint)
            ):
                _fail()
            issued = _parse_time(now)
            expires = _parse_time(args.expires_at)
            window = math.ceil((expires - issued).total_seconds())
            envelope = prepare_non_authorizing_candidate_v2(
                candidate_path,
                mapping_path,
                code_revision=revision,
                owner_identity=args.owner_identity,
                host_fingerprint=v1_execution.host_fingerprint(),
                account_fingerprint=v1_execution.account_fingerprint(),
                credential_fingerprint=cast(str, args.credential_fingerprint),
                issued_at=now,
                expires_at=args.expires_at,
                maximum_execution_window_seconds=window,
            )
            print(
                "task12b-v2 status=candidate-prepared network_authority=0 candidate_sha256=%s"
                % envelope["candidate_sha256"]
            )
            return 0
        candidate = read_private_record(candidate_path)["candidate"]
        revision = v1_execution.repository_preflight(
            repo_root, cast(str, candidate.get("execution_code_revision"))
        )
        if args.execute_authorized_45_unit_manifest:
            api_key = _api_key or v1_execution._load_credential(repo_root)
            v1_execution._scan_committed_secret(repo_root, api_key)
            with v1_execution.paid_ambient_guard():
                result = execute_authorized_45_unit_manifest(
                    repo_root,
                    global_root,
                    local_root,
                    candidate_path,
                    approval_path,
                    mapping_path,
                    api_key=api_key,
                    client_factory=_client_factory,
                    now=now,
                    code_revision=revision,
                    host_fingerprint_value=v1_execution.host_fingerprint(),
                    account_fingerprint_value=v1_execution.account_fingerprint(),
                    credential_fingerprint_value=v1_execution.credential_fingerprint(
                        api_key
                    ),
                    clock=(
                        (lambda: cast(str, _now))
                        if _now is not None
                        else v1_execution._utc_now
                    ),
                )
            print(
                "task12b-v2 status=%s dispatches=%d successes=%d failures=%d pending=%d"
                % (
                    result["status"],
                    result["dispatches"],
                    result["successes"],
                    result["failures"],
                    result["pending"],
                )
            )
            return 0
        credential_digest = args.credential_fingerprint
        if not _fingerprint(credential_digest):
            _fail("offline_command_requires_credential_fingerprint")
        context = _historical_context(
            candidate_path,
            approval_path,
            mapping_path,
            revision,
            cast(str, credential_digest),
        )
        if args.verify_authority:
            print(
                "task12b-v2 status=authority-verified network_attempts=0 authorization_id=%s"
                % context.authorization_id
            )
            return 0
        if args.export_blind_assessment:
            with machine_global_run_lock(global_root):
                packet = build_blind_assessment_v2(
                    repo_root,
                    global_root,
                    local_root,
                    candidate_path,
                    mapping_path,
                    revision,
                    context.host_fingerprint,
                    context.account_fingerprint,
                    context.credential_fingerprint,
                    approval_path,
                )
                _write_private_bytes_no_overwrite(
                    local_root / "blind-assessment.json",
                    canonical_bytes(packet) + b"\n",
                )
            print(
                "task12b-v2 status=blind-assessment-exported assessment_ready=%s network_attempts=0"
                % str(packet["assessment_ready"]).lower()
            )
            return 0
        blind = read_private_record(local_root / "blind-assessment.json")
        mapping = verify_mapping_v2(mapping_path, candidate)["mapping"]
        if args.lock_annotations:
            if args.annotations is None:
                _fail()
            try:
                annotations = json.loads(args.annotations.read_text("utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                _fail()
            publish_annotation_lock(
                local_root / "annotation-lock.json",
                annotations,
                candidate,
                mapping,
                blind,
            )
            print("task12b-v2 status=annotations-locked network_attempts=0")
            return 0
        if args.score_locked_annotations:
            result = score_locked_annotations_v2(
                repo_root,
                global_root,
                local_root,
                candidate_path,
                approval_path,
                mapping_path,
                local_root / "annotation-lock.json",
                context,
            )
            _write_private_bytes_no_overwrite(
                local_root / "score.json", canonical_bytes(result) + b"\n"
            )
            print(
                "task12b-v2 status=scored verdict=%s network_attempts=0"
                % result["verdict"]
            )
            return 0
        if args.finalize_authorized_orphan:
            units = {u.unit_id: u for u in build_schedule(load_contract()[0])}
            if args.orphan_unit not in units:
                _fail()
            with machine_global_run_lock(global_root):
                finalize_orphan_unit_offline(
                    global_root,
                    local_root,
                    units[args.orphan_unit],
                    context,
                    local_root / "owner-orphan-finalization.json",
                    now,
                    _lock_held=True,
                )
            print("task12b-v2 status=orphan-finalized network_attempts=0")
            return 0
        _fail()
    except Exception as exc:
        category = (
            exc.category if type(exc) is ExecutionFailure else "sanitized_failure"
        )
        print(
            "task12b-v2 status=blocked category=%s network_details=withheld" % category
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
