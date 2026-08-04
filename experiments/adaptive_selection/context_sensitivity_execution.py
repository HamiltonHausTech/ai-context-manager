"""Task 12b authority, replay evidence, and blind-assessment controls.

This module is offline by default.  A preparation function can create a private
candidate and blind mapping, but a candidate grants *zero* network authority.
Actual authority additionally requires a separately written owner-approval
record containing the exact candidate SHA-256 echoed out of band.  That record
is operational process evidence; it is not cryptographic proof against another
process running as the same OS user.

Machine-global claims and terminals are the authoritative cooperative replay
interlock.  Repository-local records are only verified mirrors.  Files are
atomic no-overwrite cooperative evidence, not immutable against the OS owner.
Public CLI paths are canonical and non-configurable; path injection exists only
in these offline primitives for tests.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import json
import logging
import os
import pwd
import re
import secrets
import ssl
import stat
import subprocess
import sys
from decimal import Decimal, InvalidOperation
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    MutableSequence,
    NoReturn,
    Optional,
    Sequence,
    Set,
    Tuple,
    cast,
)

from .context_sensitivity_calibration import (
    CONTRACT_PATH,
    EXECUTION_ORDER,
    PINNED_CONTRACT_SHA256,
    canonical_bytes,
    load_contract,
    render_requests,
)

READINESS_PATH = Path(
    "experiments/adaptive_selection/controls/task12b_execution_readiness_v1.json"
)
LOCAL_OUTPUT_PATH = Path(".local/adaptive-selection-task12b")
PINNED_READINESS_MANIFEST_SHA256 = (
    "09e08e249b914b096271f64fb39a212a70539e231b2ea7198805c4978ecaa4b4"
)
READINESS_HASH = "sha256:" + PINNED_READINESS_MANIFEST_SHA256
CONTRACT_HASH = "sha256:" + PINNED_CONTRACT_SHA256
CONTRACT_LINEAGE = "task12b-context-sensitivity-calibration-v1@" + CONTRACT_HASH
CONTRACT_LINEAGE_DIRECTORY = hashlib.sha256(
    CONTRACT_LINEAGE.encode("utf-8")
).hexdigest()

CANDIDATE_VERSION = "task12b-authorization-candidate-v2"
APPROVAL_VERSION = "task12b-owner-approval-v1"
MAPPING_VERSION = "task12b-private-blind-mapping-v1"
CLAIM_VERSION = "task12b-authority-consumption-claim-v2"
TERMINAL_VERSION = "task12b-machine-global-terminal-v2"
TERMINAL_INDEX_VERSION = "task12b-machine-global-terminal-index-v1"
ORPHAN_FINALIZATION_VERSION = "task12b-orphan-finalization-authorization-v1"
BLIND_EXPORT_VERSION = "task12b-blind-assessment-v2"
PINNED_MODEL = "gpt-5.6-terra"
PINNED_OPENAI_VERSION = "2.46.0"
PINNED_HTTPX_VERSION = "0.28.1"
PINNED_PRICES = {
    "currency": "USD",
    "input_per_million": "2.00",
    "cached_input_per_million": "0.20",
    "cache_write_input_per_million": "2.50",
    "output_per_million": "12.00",
}
PINNED_PROVIDER_SETTINGS = {
    "reasoning_effort": "medium",
    "max_output_tokens": 2048,
    "max_retries": 0,
    "store": False,
    "stream": False,
    "timeout_seconds": 30.0,
    "tools": [],
}
PINNED_SCOPE = (
    "single-host-single-account-single-credential-exact-contract-lineage-nine-cell-run"
)
CONSERVATIVE_EXECUTION_CEILING = "0.294164"
OWNER_CAP = "1.00"
NO_RETRY_POLICY = "one claim and at most one dispatch per contract-lineage cell; max_retries=0; no retry, fallback, replacement, model substitution, tools, or adaptation"
FAILURE_CATEGORIES = {
    "http_error",
    "provider_refusal",
    "malformed_response",
    "invalid_response",
    "transport_error",
    "budget_bound_violation",
    "secret_detected",
    "invalid_ambiguous_orphan_claim",
}
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
CANDIDATE_FIELDS: Set[str] = {
    "version",
    "authorization_id",
    "nonce",
    "issued_at",
    "expires_at",
    "maximum_execution_window_seconds",
    "contract_lineage",
    "contract_hash",
    "readiness_manifest_hash",
    "execution_code_revision",
    "ordered_request_hashes",
    "model",
    "openai_version",
    "httpx_version",
    "provider_settings",
    "prices",
    "conservative_execution_ceiling",
    "owner_cap",
    "no_retry_policy",
    "approved_cell_count",
    "scope",
    "host_fingerprint",
    "account_fingerprint",
    "credential_fingerprint",
    "blind_mapping_commitment",
    "owner_identity",
    "candidate_grants_network_authority",
}


class ExecutionFailure(RuntimeError):
    """Sanitized fixed failure; private/provider values are never included."""

    def __init__(self, category: str = "preflight_rejected_no_network_attempt") -> None:
        self.category = category
        self.raw_bytes: Optional[bytes] = None
        self.response_metadata: Optional[Dict[str, Any]] = None
        self.usage: Optional[Dict[str, Any]] = None
        self.actual_cost: Optional[str] = None
        self.conservative_cost_upper_bound: Optional[str] = None
        super().__init__(category)


def _fail(category: str = "preflight_rejected_no_network_attempt") -> NoReturn:
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
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        _fail()
    if parsed.tzinfo is None:
        _fail()
    return parsed


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _require_live_authority(context: "AuthorityContext", observed_at: str) -> None:
    observed = _parse_time(observed_at)
    if not (
        _parse_time(context.issued_at)
        <= _parse_time(context.approved_at)
        <= observed
        <= _parse_time(context.expires_at)
    ):
        _fail("authorization_not_live_no_network_attempt")


def _hex_form(value: Any, prefix: str, hex_length: int) -> bool:
    return (
        type(value) is str
        and value.startswith(prefix)
        and len(value) == len(prefix) + hex_length
        and all(c in "0123456789abcdef" for c in value[len(prefix) :])
    )


def _owner_identity(value: Any) -> bool:
    return (
        type(value) is str
        and 0 < len(value) <= 200
        and value == value.strip()
        and all(32 <= ord(character) <= 126 for character in value)
    )


def _exact_json(value: Any, expected: Any) -> bool:
    if type(value) is not type(expected):
        return False
    if type(expected) is dict:
        return set(value) == set(expected) and all(
            _exact_json(value[key], expected[key]) for key in expected
        )
    if type(expected) is list:
        return len(value) == len(expected) and all(
            _exact_json(item, expected_item)
            for item, expected_item in zip(value, expected)
        )
    return value == expected


def _fingerprint(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 71
        and value.startswith("sha256:")
        and all(c in "0123456789abcdef" for c in value[7:])
    )


def _revision(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) in {40, 64}
        and all(c in "0123456789abcdef" for c in value)
    )


def _lexists(path: Path) -> bool:
    try:
        path.lstat()
        return True
    except FileNotFoundError:
        return False
    except OSError:
        _fail()


def _check_no_symlink_ancestors(path: Path) -> None:
    absolute = path.absolute()
    for ancestor in reversed(absolute.parents):
        if ancestor == Path(ancestor.anchor):
            continue
        try:
            metadata = ancestor.lstat()
        except FileNotFoundError:
            continue
        except OSError:
            _fail()
        if stat.S_ISLNK(metadata.st_mode):
            _fail()
        if not stat.S_ISDIR(metadata.st_mode):
            _fail()


def _validate_private_directory(path: Path) -> None:
    _check_no_symlink_ancestors(path)
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
    """Create private descendants without traversing a symlink."""
    _check_no_symlink_ancestors(path)
    missing: List[Path] = []
    cursor = path
    while not _lexists(cursor):
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
    """Atomically publish via a private temp inode and no-overwrite hard link."""
    _ensure_private_directory(path.parent)
    parent_fd, parent_metadata = _open_parent(path)
    descriptor: Optional[int] = None
    temp_name = ".task12b-publish-" + secrets.token_hex(16)
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
        # A failure before the link leaves no final record.  A failure after the
        # link preserves the complete fsynced winner for later reconciliation.
        if not published:
            _fail()
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


def _read_private_bytes(path: Path, maximum: int = 4 * 1024 * 1024) -> bytes:
    parent_fd, parent_metadata = _open_parent(path)
    descriptor: Optional[int] = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path.name, flags, dir_fd=parent_fd)
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
            chunk = os.read(descriptor, min(65536, maximum + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
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
    """Test-only path-injected writer; production CLI never exposes this."""
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
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        _fail()
    if (
        hashlib.sha256(raw).hexdigest() != PINNED_READINESS_MANIFEST_SHA256
        or type(value) is not dict
    ):
        _fail()
    try:
        cells = value["execution"]["ordered_cells"]
        exact = (
            value["readiness_version"] == "task12b-execution-readiness-v2"
            and value["contract_lineage"]["identity"] == CONTRACT_LINEAGE
            and value["contract"]["sha256"] == CONTRACT_HASH
            and tuple(item["cell_id"] for item in cells) == EXECUTION_ORDER
            and tuple(item["request_sha256"] for item in cells) == REQUEST_HASHES
            and value["execution"]["network_requests_authorized_by_manifest"] == 0
            and value["budget"]["conservative_execution_ceiling"]
            == CONSERVATIVE_EXECUTION_CEILING
            and value["budget"]["owner_cap"] == OWNER_CAP
            and value["provider_configuration"]["openai_version"]
            == PINNED_OPENAI_VERSION
            and value["provider_configuration"]["httpx_version"] == PINNED_HTTPX_VERSION
            and value["provider_configuration"]["max_retries"] == 0
        )
    except (KeyError, TypeError):
        exact = False
    if not exact:
        _fail()
    return cast(Dict[str, Any], value), raw


def _mapping_payload(
    authorization_id: str,
    nonce: str,
    token_bytes: Callable[[int], bytes],
    shuffle: Callable[[MutableSequence[Dict[str, str]]], None],
) -> Dict[str, Any]:
    entries = [
        {
            "canonical_cell_id": cell_id,
            "assessment_id": "assessment-"
            + hashlib.sha256(token_bytes(16) + bytes([index])).hexdigest()[:32],
        }
        for index, cell_id in enumerate(EXECUTION_ORDER)
    ]
    blind_order = [dict(item) for item in entries]
    shuffle(blind_order)
    if [item["canonical_cell_id"] for item in blind_order] == list(EXECUTION_ORDER):
        blind_order = blind_order[1:] + blind_order[:1]
    return {
        "version": MAPPING_VERSION,
        "authorization_id": authorization_id,
        "nonce": nonce,
        "contract_lineage": CONTRACT_LINEAGE,
        "entries": entries,
        "blind_order": blind_order,
        "private_until_annotations_locked": True,
    }


def prepare_non_authorizing_candidate(
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
    shuffle: Callable[
        [MutableSequence[Dict[str, str]]], None
    ] = secrets.SystemRandom().shuffle,
) -> Dict[str, Any]:
    """Prepare candidate/mapping only; deliberately never writes approval."""
    if not _revision(code_revision) or any(
        not _fingerprint(v)
        for v in (host_fingerprint, account_fingerprint, credential_fingerprint)
    ):
        _fail()
    if not _owner_identity(owner_identity):
        _fail()
    issued = _parse_time(issued_at)
    expires = _parse_time(expires_at)
    if (
        type(maximum_execution_window_seconds) is not int
        or maximum_execution_window_seconds <= 0
        or maximum_execution_window_seconds > 86400
        or not issued < expires
        or (expires - issued).total_seconds() > maximum_execution_window_seconds
    ):
        _fail()
    nonce = token_bytes(24).hex()
    authorization_id = "task12b-auth-" + token_bytes(16).hex()
    mapping = _mapping_payload(authorization_id, nonce, token_bytes, shuffle)
    mapping_envelope = {"mapping": mapping, "mapping_digest": sha256_canonical(mapping)}
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
        "execution_code_revision": code_revision,
        "ordered_request_hashes": list(REQUEST_HASHES),
        "model": PINNED_MODEL,
        "openai_version": PINNED_OPENAI_VERSION,
        "httpx_version": PINNED_HTTPX_VERSION,
        "provider_settings": dict(PINNED_PROVIDER_SETTINGS),
        "prices": dict(PINNED_PRICES),
        "conservative_execution_ceiling": CONSERVATIVE_EXECUTION_CEILING,
        "owner_cap": OWNER_CAP,
        "no_retry_policy": NO_RETRY_POLICY,
        "approved_cell_count": 9,
        "scope": PINNED_SCOPE,
        "host_fingerprint": host_fingerprint,
        "account_fingerprint": account_fingerprint,
        "credential_fingerprint": credential_fingerprint,
        "blind_mapping_commitment": mapping_envelope["mapping_digest"],
        "owner_identity": owner_identity,
        "candidate_grants_network_authority": False,
    }
    envelope = {"candidate": candidate, "candidate_sha256": sha256_canonical(candidate)}
    write_test_private_record(mapping_path, mapping_envelope)
    try:
        write_test_private_record(candidate_path, envelope)
    except Exception:
        # Mapping remains explicitly non-authorizing even if candidate publication fails.
        raise
    return envelope


def verify_candidate(path: Path, code_revision: str) -> Dict[str, Any]:
    value = read_private_record(path)
    if (
        set(value) != {"candidate", "candidate_sha256"}
        or type(value["candidate"]) is not dict
    ):
        _fail()
    candidate = value["candidate"]
    if set(candidate) != CANDIDATE_FIELDS or value[
        "candidate_sha256"
    ] != sha256_canonical(candidate):
        _fail()
    expected = {
        "version": CANDIDATE_VERSION,
        "contract_lineage": CONTRACT_LINEAGE,
        "contract_hash": CONTRACT_HASH,
        "readiness_manifest_hash": READINESS_HASH,
        "execution_code_revision": code_revision,
        "ordered_request_hashes": list(REQUEST_HASHES),
        "model": PINNED_MODEL,
        "openai_version": PINNED_OPENAI_VERSION,
        "httpx_version": PINNED_HTTPX_VERSION,
        "provider_settings": PINNED_PROVIDER_SETTINGS,
        "prices": PINNED_PRICES,
        "conservative_execution_ceiling": CONSERVATIVE_EXECUTION_CEILING,
        "owner_cap": OWNER_CAP,
        "no_retry_policy": NO_RETRY_POLICY,
        "approved_cell_count": 9,
        "scope": PINNED_SCOPE,
        "candidate_grants_network_authority": False,
    }
    if any(
        candidate.get(name) != expected_value
        for name, expected_value in expected.items()
    ):
        _fail()
    if (
        not _revision(code_revision)
        or not _hex_form(candidate["authorization_id"], "task12b-auth-", 32)
        or not _hex_form(candidate["nonce"], "", 48)
        or not _fingerprint(candidate["blind_mapping_commitment"])
        or not _owner_identity(candidate["owner_identity"])
    ):
        _fail()
    if any(
        not _fingerprint(candidate[name])
        for name in (
            "host_fingerprint",
            "account_fingerprint",
            "credential_fingerprint",
        )
    ):
        _fail()
    issued = _parse_time(candidate["issued_at"])
    expires = _parse_time(candidate["expires_at"])
    window = candidate["maximum_execution_window_seconds"]
    if (
        type(window) is not int
        or not 0 < window <= 86400
        or not issued < expires
        or (expires - issued).total_seconds() > window
    ):
        _fail()
    return value


def verify_mapping(path: Path, candidate: Mapping[str, Any]) -> Dict[str, Any]:
    value = read_private_record(path)
    if (
        set(value) != {"mapping", "mapping_digest"}
        or type(value["mapping"]) is not dict
    ):
        _fail()
    mapping = value["mapping"]
    if (
        value["mapping_digest"] != sha256_canonical(mapping)
        or value["mapping_digest"] != candidate["blind_mapping_commitment"]
    ):
        _fail()
    mapping_fields = {
        "version",
        "authorization_id",
        "nonce",
        "contract_lineage",
        "entries",
        "blind_order",
        "private_until_annotations_locked",
    }
    if (
        set(mapping) != mapping_fields
        or mapping.get("version") != MAPPING_VERSION
        or mapping.get("authorization_id") != candidate["authorization_id"]
        or mapping.get("nonce") != candidate["nonce"]
        or mapping.get("contract_lineage") != CONTRACT_LINEAGE
        or mapping.get("private_until_annotations_locked") is not True
    ):
        _fail()
    entries = mapping.get("entries")
    order = mapping.get("blind_order")
    if (
        type(entries) is not list
        or type(order) is not list
        or len(entries) != 9
        or len(order) != 9
        or any(type(item) is not dict for item in entries + order)
        or any(
            set(item) != {"canonical_cell_id", "assessment_id"}
            for item in entries + order
        )
        or [item.get("canonical_cell_id") for item in entries] != list(EXECUTION_ORDER)
    ):
        _fail()
    ids = [item.get("assessment_id") for item in entries]
    if (
        len(set(ids)) != 9
        or any(not _hex_form(item, "assessment-", 32) for item in ids)
        or any(item in EXECUTION_ORDER for item in ids)
        or set(
            (item.get("canonical_cell_id"), item.get("assessment_id")) for item in order
        )
        != set(
            (item.get("canonical_cell_id"), item.get("assessment_id"))
            for item in entries
        )
        or [item["canonical_cell_id"] for item in order] == list(EXECUTION_ORDER)
    ):
        _fail()
    return value


@dataclass(frozen=True)
class AuthorityContext:
    authorization_id: str
    candidate_digest: str
    approval_digest: str
    code_revision: str
    host_fingerprint: str
    account_fingerprint: str
    credential_fingerprint: str
    blind_mapping_commitment: str
    authorization_nonce: str
    owner_identity: str
    issued_at: str
    approved_at: str
    expires_at: str


def verify_authority(
    candidate_path: Path,
    approval_path: Path,
    mapping_path: Path,
    code_revision: str,
    host_fingerprint: str,
    account_fingerprint: str,
    credential_fingerprint: str,
    now: str,
) -> AuthorityContext:
    context = verify_historical_authority(
        candidate_path,
        approval_path,
        mapping_path,
        code_revision,
        host_fingerprint,
        account_fingerprint,
        credential_fingerprint,
    )
    current = _parse_time(now)
    if (
        not _parse_time(context.approved_at)
        <= current
        <= _parse_time(context.expires_at)
    ):
        _fail()
    return context


def verify_historical_authority(
    candidate_path: Path,
    approval_path: Path,
    mapping_path: Path,
    code_revision: str,
    host_fingerprint: str,
    account_fingerprint: str,
    credential_fingerprint: str,
) -> AuthorityContext:
    """Verify exact authority identity without requiring its execution window current."""
    candidate_envelope = verify_candidate(candidate_path, code_revision)
    candidate = candidate_envelope["candidate"]
    verify_mapping(mapping_path, candidate)
    approval_envelope = read_private_record(approval_path)
    if (
        set(approval_envelope) != {"approval", "approval_digest"}
        or type(approval_envelope["approval"]) is not dict
    ):
        _fail()
    approval = approval_envelope["approval"]
    required = {
        "version",
        "authorization_id",
        "candidate_sha256",
        "owner_echoed_candidate_sha256_out_of_band",
        "owner_identity",
        "approved_at",
        "operational_process_evidence_only",
    }
    if set(approval) != required or approval_envelope[
        "approval_digest"
    ] != sha256_canonical(approval):
        _fail()
    if approval != {
        "version": APPROVAL_VERSION,
        "authorization_id": candidate["authorization_id"],
        "candidate_sha256": candidate_envelope["candidate_sha256"],
        "owner_echoed_candidate_sha256_out_of_band": True,
        "owner_identity": candidate["owner_identity"],
        "approved_at": approval["approved_at"],
        "operational_process_evidence_only": True,
    }:
        _fail()
    if (
        not _parse_time(candidate["issued_at"])
        <= _parse_time(approval["approved_at"])
        <= _parse_time(candidate["expires_at"])
    ):
        _fail()
    if (host_fingerprint, account_fingerprint, credential_fingerprint) != (
        candidate["host_fingerprint"],
        candidate["account_fingerprint"],
        candidate["credential_fingerprint"],
    ):
        _fail()
    return AuthorityContext(
        candidate["authorization_id"],
        candidate_envelope["candidate_sha256"],
        approval_envelope["approval_digest"],
        code_revision,
        host_fingerprint,
        account_fingerprint,
        credential_fingerprint,
        candidate["blind_mapping_commitment"],
        candidate["nonce"],
        candidate["owner_identity"],
        candidate["issued_at"],
        approval["approved_at"],
        candidate["expires_at"],
    )


def _identity(
    cell_id: str, request_hash: str, context: AuthorityContext
) -> Dict[str, Any]:
    return {
        "contract_lineage": CONTRACT_LINEAGE,
        "contract_hash": CONTRACT_HASH,
        "readiness_manifest_hash": READINESS_HASH,
        "cell_id": cell_id,
        "request_hash": request_hash,
        "authorization_id": context.authorization_id,
        "candidate_digest": context.candidate_digest,
        "approval_digest": context.approval_digest,
        "code_revision": context.code_revision,
        "host_fingerprint": context.host_fingerprint,
        "account_fingerprint": context.account_fingerprint,
        "credential_fingerprint": context.credential_fingerprint,
    }


def _claim_path(root: Path, cell_id: str) -> Path:
    return root / "claims" / (cell_id + ".claim.json")


def _terminal_path(root: Path, cell_id: str) -> Path:
    return root / "terminals" / (cell_id + ".terminal.json")


def _index_path(root: Path, cell_id: str) -> Path:
    return root / "terminal-index" / (cell_id + ".json")


def _raw_path(root: Path, cell_id: str) -> Path:
    return root / "raw" / (cell_id + ".raw")


def _local_claim(local: Path, cell_id: str) -> Path:
    return local / "mirrors" / "claims" / (cell_id + ".claim.json")


def _local_terminal(local: Path, cell_id: str) -> Path:
    return local / "mirrors" / "terminals" / (cell_id + ".terminal.json")


def publish_authority_claim(
    global_root: Path,
    local_root: Path,
    cell_id: str,
    request_hash: str,
    context: AuthorityContext,
    timestamp: str,
) -> Dict[str, Any]:
    """Consume authority globally first; existence does not assert dispatch."""
    _parse_time(timestamp)
    claim = {
        "version": CLAIM_VERSION,
        **_identity(cell_id, request_hash, context),
        "authority_consumed": True,
        "consumed_at": timestamp,
    }
    envelope = {"claim": claim, "claim_digest": sha256_canonical(claim)}
    data = canonical_bytes(envelope) + b"\n"
    _write_private_bytes_no_overwrite(_claim_path(global_root, cell_id), data)
    try:
        _write_private_bytes_no_overwrite(_local_claim(local_root, cell_id), data)
    except Exception:
        # Global authority remains consumed; no dispatch may follow.
        raise
    verify_claim_file(_claim_path(global_root, cell_id), cell_id, request_hash, context)
    verify_claim_file(_local_claim(local_root, cell_id), cell_id, request_hash, context)
    return envelope


def verify_claim_file(
    path: Path, cell_id: str, request_hash: str, context: AuthorityContext
) -> Dict[str, Any]:
    value = read_private_record(path)
    if set(value) != {"claim", "claim_digest"} or type(value["claim"]) is not dict:
        _fail()
    claim = value["claim"]
    expected_identity = _identity(cell_id, request_hash, context)
    expected_fields = {
        "version",
        *expected_identity,
        "authority_consumed",
        "consumed_at",
    }
    if (
        set(claim) != expected_fields
        or claim.get("version") != CLAIM_VERSION
        or claim.get("authority_consumed") is not True
        or any(claim.get(k) != v for k, v in expected_identity.items())
        or value["claim_digest"] != sha256_canonical(claim)
    ):
        _fail()
    _parse_time(claim["consumed_at"])
    return value


def _mirror_exact(source: Path, destination: Path) -> None:
    data = _read_private_bytes(source)
    if _lexists(destination):
        if _read_private_bytes(destination) != data:
            _fail()
        return
    _write_private_bytes_no_overwrite(destination, data)


def _valid_usage(value: Any) -> bool:
    fields = {
        "input_tokens",
        "cached_input_tokens",
        "cache_write_input_tokens",
        "output_tokens",
        "total_tokens",
    }
    if type(value) is not dict or set(value) != fields:
        return False
    integer_fields = fields - {"cache_write_input_tokens"}
    if any(
        type(value[field]) is not int or value[field] < 0 for field in integer_fields
    ):
        return False
    if value["cache_write_input_tokens"] is not None and (
        type(value["cache_write_input_tokens"]) is not int
        or value["cache_write_input_tokens"] < 0
    ):
        return False
    return (
        value["cached_input_tokens"] <= value["input_tokens"]
        and (
            value["cache_write_input_tokens"] is None
            or value["cached_input_tokens"] + value["cache_write_input_tokens"]
            <= value["input_tokens"]
        )
        and value["total_tokens"] == value["input_tokens"] + value["output_tokens"]
        and value["output_tokens"] <= 2048
    )


def _decimal_string(value: Any) -> bool:
    return (
        type(value) is str and re.fullmatch(r"(?:0|[1-9]\d*)\.\d+", value) is not None
    )


def _request_metadata(request_hash: str) -> Dict[str, Any]:
    return {
        "method": "POST",
        "scheme": "https",
        "host": "api.openai.com",
        "path": "/v1/responses",
        "request_body_sha256": request_hash,
        "model": PINNED_MODEL,
    }


def _valid_response_metadata(value: Any) -> bool:
    if type(value) is not dict or set(value) != {
        "http_status",
        "content_type",
        "provider_request_id",
        "response_id",
        "observed_model",
    }:
        return False
    return (
        type(value["http_status"]) is int
        and 100 <= value["http_status"] <= 599
        and (
            value["content_type"] is None
            or (
                type(value["content_type"]) is str
                and 0 < len(value["content_type"]) <= 200
                and all(32 <= ord(c) <= 126 for c in value["content_type"])
            )
        )
        and (
            value["provider_request_id"] is None
            or (
                type(value["provider_request_id"]) is str
                and 0 < len(value["provider_request_id"]) <= 200
                and all(32 <= ord(c) <= 126 for c in value["provider_request_id"])
            )
        )
        and all(
            item is None
            or (
                type(item) is str
                and 0 < len(item) <= 200
                and all(32 <= ord(c) <= 126 for c in item)
            )
            for item in (value["response_id"], value["observed_model"])
        )
    )


def _valid_success_transport_metadata(value: Any) -> bool:
    if not _valid_response_metadata(value):
        return False
    content_type = value["content_type"]
    media_type = content_type.split(";", 1)[0].strip().lower() if content_type else ""
    return bool(
        value["http_status"] == 200
        and media_type in {"application/json", "application/problem+json"}
        and value["provider_request_id"] is not None
        and value["response_id"] is not None
        and value["provider_request_id"] != value["response_id"]
    )


def _valid_success_response_metadata(value: Any) -> bool:
    return bool(
        _valid_success_transport_metadata(value)
        and value["observed_model"] == PINNED_MODEL
    )


def _valid_dispatch(value: Any) -> bool:
    return value is True or value is False or value == "unknown"


def publish_terminal(
    global_root: Path,
    local_root: Path,
    cell_id: str,
    request_hash: str,
    context: AuthorityContext,
    *,
    kind: str,
    dispatch_invoked: Any,
    server_acceptance: str,
    provider_visible_evidence: Optional[Mapping[str, Any]],
    structured_response: Optional[Mapping[str, Any]],
    raw_bytes: Optional[bytes],
    recorded_at: str,
    failure_category: Optional[str] = None,
    usage: Optional[Mapping[str, Any]] = None,
    actual_cost: Optional[str] = None,
    conservative_cost_upper_bound: Optional[str] = None,
    response_metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    _parse_time(recorded_at)
    claim = verify_claim_file(
        _claim_path(global_root, cell_id), cell_id, request_hash, context
    )
    if (
        not _valid_dispatch(dispatch_invoked)
        or server_acceptance not in {"yes", "no", "unknown"}
        or kind not in {"success", "failure", "invalid_ambiguous"}
    ):
        _fail()
    if kind == "success" and (
        dispatch_invoked is not True
        or server_acceptance != "yes"
        or raw_bytes is None
        or type(structured_response) is not dict
    ):
        _fail()
    if (kind == "success" and failure_category is not None) or (
        kind != "success" and failure_category not in FAILURE_CATEGORIES
    ):
        _fail()
    if usage is not None and not _valid_usage(usage):
        _fail()
    if any(
        value is not None and not _decimal_string(value)
        for value in (actual_cost, conservative_cost_upper_bound)
    ):
        _fail()
    if raw_bytes is not None and response_metadata is None:
        _fail()
    if response_metadata is not None and not _valid_response_metadata(
        response_metadata
    ):
        _fail()
    if server_acceptance == "yes" and response_metadata is None:
        _fail()
    if dispatch_invoked is True and conservative_cost_upper_bound is None:
        _fail()
    if usage is not None:
        expected_actual, expected_upper = usage_costs(usage)
        if (
            actual_cost != expected_actual
            or conservative_cost_upper_bound != expected_upper
        ):
            _fail()
    if kind == "success" and (
        usage is None
        or provider_visible_evidence is None
        or response_metadata is None
        or response_metadata["response_id"] is None
        or response_metadata["observed_model"] != PINNED_MODEL
    ):
        _fail()
    if kind == "success":
        try:
            validate_task12b_response(structured_response)
        except ExecutionFailure:
            _fail()
    if kind == "invalid_ambiguous" and (
        dispatch_invoked != "unknown"
        or server_acceptance != "unknown"
        or raw_bytes is not None
        or usage is not None
        or actual_cost is not None
        or conservative_cost_upper_bound is not None
        or provider_visible_evidence is not None
        or structured_response is not None
    ):
        _fail()
    if (
        kind == "failure"
        and raw_bytes is not None
        and (dispatch_invoked is not True or server_acceptance != "yes")
    ):
        _fail()
    if (
        kind == "failure"
        and raw_bytes is None
        and response_metadata is None
        and failure_category not in {"transport_error", "secret_detected"}
    ):
        _fail()
    raw_hash: Optional[str] = None
    if raw_bytes is not None:
        raw_hash = _sha256_bytes(raw_bytes)
        _write_private_bytes_no_overwrite(_raw_path(global_root, cell_id), raw_bytes)
    terminal = {
        "version": TERMINAL_VERSION,
        **_identity(cell_id, request_hash, context),
        "claim_digest": claim["claim_digest"],
        "authority_consumed": True,
        "kind": kind,
        "dispatch_invoked": dispatch_invoked,
        "server_acceptance": server_acceptance,
        "failure_category": failure_category,
        "request_metadata": _request_metadata(request_hash),
        "response_metadata": (
            dict(response_metadata) if response_metadata is not None else None
        ),
        "usage": dict(usage) if usage is not None else None,
        "actual_cost": actual_cost,
        "conservative_cost_upper_bound": conservative_cost_upper_bound,
        "provider_visible_evidence": (
            dict(provider_visible_evidence)
            if provider_visible_evidence is not None
            else None
        ),
        "structured_response": (
            dict(structured_response) if structured_response is not None else None
        ),
        "raw_response_sha256": raw_hash,
        "recorded_at": recorded_at,
    }
    envelope = {"terminal": terminal, "terminal_digest": sha256_canonical(terminal)}
    data = canonical_bytes(envelope) + b"\n"
    _write_private_bytes_no_overwrite(_terminal_path(global_root, cell_id), data)
    index = {
        "terminal_index": {
            "version": TERMINAL_INDEX_VERSION,
            "contract_lineage": CONTRACT_LINEAGE,
            "cell_id": cell_id,
            "terminal_digest": envelope["terminal_digest"],
            "terminal_record": "terminals/" + cell_id + ".terminal.json",
        }
    }
    index["terminal_index_digest"] = sha256_canonical(index["terminal_index"])
    _write_private_bytes_no_overwrite(
        _index_path(global_root, cell_id), canonical_bytes(index) + b"\n"
    )
    try:
        _write_private_bytes_no_overwrite(_local_terminal(local_root, cell_id), data)
    except Exception:
        # Canonical global terminal remains authoritative and can later be mirrored.
        raise
    verify_terminal(global_root, cell_id, request_hash, context)
    return envelope


def verify_terminal(
    global_root: Path,
    cell_id: str,
    request_hash: str,
    context: AuthorityContext,
    expected_provider_visible_evidence: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    claim = verify_claim_file(
        _claim_path(global_root, cell_id), cell_id, request_hash, context
    )
    value = read_private_record(_terminal_path(global_root, cell_id))
    index = read_private_record(_index_path(global_root, cell_id))
    if (
        set(value) != {"terminal", "terminal_digest"}
        or type(value["terminal"]) is not dict
    ):
        _fail()
    terminal = value["terminal"]
    identity = _identity(cell_id, request_hash, context)
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
        "usage",
        "actual_cost",
        "conservative_cost_upper_bound",
        "provider_visible_evidence",
        "structured_response",
        "raw_response_sha256",
        "recorded_at",
    }
    if (
        set(terminal) != required
        or terminal.get("version") != TERMINAL_VERSION
        or any(terminal.get(k) != v for k, v in identity.items())
        or terminal.get("claim_digest") != claim["claim_digest"]
        or terminal.get("authority_consumed") is not True
        or value["terminal_digest"] != sha256_canonical(terminal)
    ):
        _fail()
    if not _valid_dispatch(terminal["dispatch_invoked"]) or terminal[
        "server_acceptance"
    ] not in {"yes", "no", "unknown"}:
        _fail()
    if terminal["request_metadata"] != _request_metadata(request_hash):
        _fail()
    if terminal["usage"] is not None and not _valid_usage(terminal["usage"]):
        _fail()
    if any(
        value is not None and not _decimal_string(value)
        for value in (
            terminal["actual_cost"],
            terminal["conservative_cost_upper_bound"],
        )
    ):
        _fail()
    has_response = terminal["response_metadata"] is not None
    has_raw_hash = terminal["raw_response_sha256"] is not None
    if (
        (has_raw_hash and not has_response)
        or (
            has_response and not _valid_response_metadata(terminal["response_metadata"])
        )
        or (terminal["server_acceptance"] == "yes" and not has_response)
    ):
        _fail()
    if (
        terminal["dispatch_invoked"] is True
        and terminal["conservative_cost_upper_bound"] is None
    ):
        _fail()
    if terminal["usage"] is not None:
        expected_actual, expected_upper = usage_costs(terminal["usage"])
        if (
            terminal["actual_cost"] != expected_actual
            or terminal["conservative_cost_upper_bound"] != expected_upper
        ):
            _fail()
    if (
        expected_provider_visible_evidence is not None
        and terminal["dispatch_invoked"] is True
        and terminal["provider_visible_evidence"]
        != dict(expected_provider_visible_evidence)
    ):
        _fail()
    if _parse_time(terminal["recorded_at"]) < _parse_time(
        claim["claim"]["consumed_at"]
    ):
        _fail()
    expected_index = {
        "version": TERMINAL_INDEX_VERSION,
        "contract_lineage": CONTRACT_LINEAGE,
        "cell_id": cell_id,
        "terminal_digest": value["terminal_digest"],
        "terminal_record": "terminals/" + cell_id + ".terminal.json",
    }
    if index != {
        "terminal_index": expected_index,
        "terminal_index_digest": sha256_canonical(expected_index),
    }:
        _fail()
    if terminal["kind"] == "success":
        if (
            terminal["dispatch_invoked"] is not True
            or terminal["server_acceptance"] != "yes"
            or type(terminal["structured_response"]) is not dict
            or not _fingerprint(terminal["raw_response_sha256"])
            or terminal["failure_category"] is not None
            or terminal["usage"] is None
            or type(terminal["provider_visible_evidence"]) is not dict
            or not _valid_success_response_metadata(terminal["response_metadata"])
        ):
            _fail()
        raw_bytes = _read_private_bytes(_raw_path(global_root, cell_id))
        if _sha256_bytes(raw_bytes) != terminal["raw_response_sha256"]:
            _fail()
        projection = _replay_raw_success(raw_bytes)
        if (
            terminal["structured_response"] != projection["structured_response"]
            or terminal["usage"] != projection["usage"]
            or terminal["actual_cost"] != projection["actual_cost"]
            or terminal["conservative_cost_upper_bound"]
            != projection["conservative_cost_upper_bound"]
            or terminal["response_metadata"]["response_id"] != projection["response_id"]
            or terminal["response_metadata"]["observed_model"]
            != projection["observed_model"]
        ):
            _fail()
    elif terminal["kind"] in {"failure", "invalid_ambiguous"}:
        if terminal["failure_category"] not in FAILURE_CATEGORIES:
            _fail()
        if has_raw_hash:
            if (
                terminal["kind"] != "failure"
                or terminal["dispatch_invoked"] is not True
                or terminal["server_acceptance"] != "yes"
            ):
                _fail()
            if (
                _sha256_bytes(_read_private_bytes(_raw_path(global_root, cell_id)))
                != terminal["raw_response_sha256"]
            ):
                _fail()
        else:
            if _lexists(_raw_path(global_root, cell_id)):
                _fail()
            if (
                terminal["kind"] == "failure"
                and not has_response
                and terminal["failure_category"]
                not in {"transport_error", "secret_detected"}
            ):
                _fail()
            if terminal["kind"] == "invalid_ambiguous" and (
                terminal["dispatch_invoked"] != "unknown"
                or terminal["server_acceptance"] != "unknown"
                or terminal["usage"] is not None
                or terminal["actual_cost"] is not None
                or terminal["conservative_cost_upper_bound"] is not None
                or terminal["provider_visible_evidence"] is not None
                or terminal["structured_response"] is not None
            ):
                _fail()
    else:
        _fail()
    return value


def classify_state(
    global_root: Path,
    local_root: Path,
    cell_id: str,
    request_hash: str,
    context: AuthorityContext,
) -> str:
    global_claim = _claim_path(global_root, cell_id)
    local_claim = _local_claim(local_root, cell_id)
    global_terminal = _terminal_path(global_root, cell_id)
    local_terminal = _local_terminal(local_root, cell_id)
    index = _index_path(global_root, cell_id)
    raw = _raw_path(global_root, cell_id)
    gc, lc, gt, lt, ix, rw = (
        _lexists(p)
        for p in (
            global_claim,
            local_claim,
            global_terminal,
            local_terminal,
            index,
            raw,
        )
    )
    if lc and not gc:
        _fail()
    if (gt or ix or lt) and not gc:
        _fail()
    if rw and not gt:
        _fail()
    if gc:
        global_value = verify_claim_file(global_claim, cell_id, request_hash, context)
        if lc and read_private_record(local_claim) != global_value:
            _fail()
        if gt:
            terminal = verify_terminal(global_root, cell_id, request_hash, context)
            if not ix:
                _fail()
            if lt and read_private_record(local_terminal) != terminal:
                _fail()
            return "terminal"
        if ix or lt or rw:
            _fail()
        return "blocked_orphan_claim"
    if any((lc, gt, lt, ix, rw)):
        _fail()
    return "pending"


@contextmanager
def machine_global_run_lock(global_root: Path) -> Iterator[None]:
    """Hold a machine-global nonblocking lock across reconciliation/execution."""
    _ensure_private_directory(global_root)
    path = global_root / "run.lock"
    if not _lexists(path):
        try:
            _write_private_bytes_no_overwrite(
                path, b"task12b machine-global run lock\n"
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
            _fail("machine_global_run_lock_busy_no_network_attempt")
        _revalidate_parent(path, parent_metadata)
        yield
    finally:
        if descriptor is not None:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(descriptor)
        os.close(parent_fd)


def finalize_orphan_offline(
    global_root: Path,
    local_root: Path,
    cell_id: str,
    request_hash: str,
    context: AuthorityContext,
    finalization_authorization_path: Path,
    recorded_at: str,
    *,
    _lock_held: bool = False,
) -> Dict[str, Any]:
    """Zero-network owner-authorized orphan finalization; never dispatches."""
    if (
        classify_state(global_root, local_root, cell_id, request_hash, context)
        != "blocked_orphan_claim"
    ):
        _fail()
    claim = verify_claim_file(
        _claim_path(global_root, cell_id), cell_id, request_hash, context
    )
    record = read_private_record(finalization_authorization_path)
    if (
        set(record)
        != {"finalization_authorization", "finalization_authorization_digest"}
        or type(record["finalization_authorization"]) is not dict
    ):
        _fail()
    authorization = record["finalization_authorization"]
    required = {
        "version",
        "authorization_id",
        "candidate_digest",
        "approval_digest",
        "contract_lineage",
        "cell_id",
        "request_hash",
        "orphan_claim_digest",
        "owner_echoed_orphan_claim_digest_out_of_band",
        "owner_confirmed_no_process_remains",
        "confirmation_process",
        "owner_identity",
        "authorized_at",
        "operational_process_evidence_only",
    }
    if (
        set(authorization) != required
        or record["finalization_authorization_digest"]
        != sha256_canonical(authorization)
        or authorization["version"] != ORPHAN_FINALIZATION_VERSION
        or authorization["authorization_id"] != context.authorization_id
        or authorization["candidate_digest"] != context.candidate_digest
        or authorization["approval_digest"] != context.approval_digest
        or authorization["contract_lineage"] != CONTRACT_LINEAGE
        or authorization["cell_id"] != cell_id
        or authorization["request_hash"] != request_hash
        or authorization["orphan_claim_digest"] != claim["claim_digest"]
        or authorization["owner_echoed_orphan_claim_digest_out_of_band"] is not True
        or authorization["owner_confirmed_no_process_remains"] is not True
        or type(authorization["confirmation_process"]) is not str
        or len(authorization["confirmation_process"].strip()) < 20
        or authorization["owner_identity"] != context.owner_identity
        or authorization["operational_process_evidence_only"] is not True
    ):
        _fail()
    authorized = _parse_time(authorization["authorized_at"])
    recorded = _parse_time(recorded_at)
    if not (
        _parse_time(context.issued_at)
        <= _parse_time(context.approved_at)
        <= authorized
        <= recorded
    ):
        _fail()

    def publish() -> Dict[str, Any]:
        if (
            classify_state(global_root, local_root, cell_id, request_hash, context)
            != "blocked_orphan_claim"
        ):
            _fail()
        return publish_terminal(
            global_root,
            local_root,
            cell_id,
            request_hash,
            context,
            kind="invalid_ambiguous",
            dispatch_invoked="unknown",
            server_acceptance="unknown",
            provider_visible_evidence=None,
            structured_response=None,
            raw_bytes=None,
            failure_category="invalid_ambiguous_orphan_claim",
            recorded_at=recorded_at,
        )

    if _lock_held:
        return publish()
    with machine_global_run_lock(global_root):
        return publish()


def _contract_assessment_material(repo_root: Path) -> Dict[str, Dict[str, Any]]:
    contract, _ = load_contract(repo_root / CONTRACT_PATH)
    by_cell: Dict[str, Dict[str, Any]] = {}
    for scenario in contract["scenarios"]:
        rubric = scenario["rubric"]
        for cell in scenario["cells"]:
            by_cell[cell["cell_id"]] = {
                "task": scenario["task_prompt"],
                "criteria": rubric["criteria"],
                "adjudication_rules": rubric["adjudication_rules"],
                "critical_findings": rubric["critical_findings"],
            }
    return by_cell


def build_blind_assessment(
    repo_root: Path,
    global_root: Path,
    local_root: Path,
    mapping_path: Path,
    context: AuthorityContext,
) -> Dict[str, Any]:
    mapping_envelope = verify_mapping(
        mapping_path,
        {
            "authorization_id": context.authorization_id,
            "nonce": context.authorization_nonce,
            "blind_mapping_commitment": context.blind_mapping_commitment,
        },
    )
    mapping = mapping_envelope["mapping"]
    material = _contract_assessment_material(repo_root)
    _, expected_visible_evidence = _validate_requests(repo_root)
    expected_by_cell = dict(zip(EXECUTION_ORDER, expected_visible_evidence))
    terminal_by_cell: Dict[str, Dict[str, Any]] = {}
    for cell_id, request_hash in zip(EXECUTION_ORDER, REQUEST_HASHES):
        if (
            classify_state(global_root, local_root, cell_id, request_hash, context)
            != "terminal"
        ):
            _fail()
        terminal_by_cell[cell_id] = verify_terminal(
            global_root,
            cell_id,
            request_hash,
            context,
            expected_by_cell[cell_id],
        )["terminal"]
    assessments = []
    for item in mapping["blind_order"]:
        cell_id = item["canonical_cell_id"]
        terminal = terminal_by_cell[cell_id]
        evidence = terminal["provider_visible_evidence"] or {
            "task": material[cell_id]["task"],
            "timestamped_evidence": [],
        }
        assessments.append(
            {
                "assessment_id": item["assessment_id"],
                "task": evidence["task"],
                "provider_visible_timestamped_evidence": evidence[
                    "timestamped_evidence"
                ],
                "structured_response": terminal["structured_response"],
                "status": terminal["kind"],
                "criteria": material[cell_id]["criteria"],
                "adjudication_rules": material[cell_id]["adjudication_rules"],
                "critical_findings": material[cell_id]["critical_findings"],
            }
        )
    ready = (
        all(item["status"] == "success" for item in assessments)
        and len(assessments) == 9
    )
    return {
        "version": BLIND_EXPORT_VERSION,
        "assessment_ready": ready,
        "assessments": assessments,
    }


MAX_RAW_RESPONSE_BYTES = 4 * 1024 * 1024
OFFICIAL_OPENAI_BASE_URL = "https://api.openai.com/v1"
OFFICIAL_OPENAI_WEBSOCKET_URL = "wss://api.openai.com/v1"
PROXY_ENVIRONMENT_NAMES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
)
TLS_ENVIRONMENT_NAMES = (
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "SSLKEYLOGFILE",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
)
EXPLICIT_OPENAI_ENVIRONMENT_NAMES = (
    "OPENAI_API_KEY",
    "OPENAI_ADMIN_KEY",
    "OPENAI_ORG_ID",
    "OPENAI_ORGANIZATION",
    "OPENAI_PROJECT_ID",
    "OPENAI_PROJECT",
    "OPENAI_WEBHOOK_SECRET",
    "OPENAI_BASE_URL",
    "OPENAI_WEBSOCKET_BASE_URL",
    "OPENAI_CUSTOM_HEADERS",
    "OPENAI_LOG",
)


def _validate_requests(
    repo_root: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    load_readiness_manifest(repo_root / READINESS_PATH)
    contract, _ = load_contract(repo_root / CONTRACT_PATH)
    requests = render_requests(contract)
    hashes = tuple(_sha256_bytes(canonical_bytes(body)) for body in requests)
    if hashes != REQUEST_HASHES:
        _fail()
    evidence_by_cell: Dict[str, Dict[str, Any]] = {}
    for scenario in contract["scenarios"]:
        for cell in scenario["cells"]:
            evidence_by_cell[cell["cell_id"]] = {
                "task": scenario["task_prompt"],
                "timestamped_evidence": [
                    "%d. [%s] %s" % (index, item["observed_at"], item["content"])
                    for index, item in enumerate(cell["evidence"], 1)
                ],
            }
    evidence = [evidence_by_cell[cell_id] for cell_id in EXECUTION_ORDER]
    for body, visible in zip(requests, evidence):
        expected_input = "Task:\n%s\n\nEvidence:\n%s" % (
            visible["task"],
            "\n".join(visible["timestamped_evidence"]),
        )
        if body.get("input") != expected_input:
            _fail()
    return cast(List[Dict[str, Any]], requests), evidence


def build_dry_run_summary(repo_root: Path = Path(".")) -> Dict[str, Any]:
    """No credentials, provider imports, files, network, prompts, or authority."""
    _validate_requests(repo_root)
    return {
        "mode": "offline_dry_run",
        "contract_lineage": CONTRACT_LINEAGE,
        "contract_sha256": CONTRACT_HASH,
        "readiness_manifest_sha256": READINESS_HASH,
        "network_requests_authorized": 0,
        "candidate_grants_network_authority": False,
        "cell_count": 9,
        "ordered_request_hashes": list(REQUEST_HASHES),
        "conservative_execution_ceiling": CONSERVATIVE_EXECUTION_CEILING,
        "owner_cap": OWNER_CAP,
    }


def _run_git(repo_root: Path, arguments: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            ["git"] + list(arguments),
            cwd=str(repo_root),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        return completed.stdout.decode("utf-8", "strict").strip()
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError):
        _fail()


def repository_preflight(
    repo_root: Path, candidate_revision: Optional[str] = None
) -> str:
    """Require ordinary clean main at the already-fetched origin/main revision."""
    if _run_git(repo_root, ["symbolic-ref", "--quiet", "--short", "HEAD"]) != "main":
        _fail()
    if _run_git(repo_root, ["status", "--porcelain=v1", "--untracked-files=all"]):
        _fail()
    head = _run_git(repo_root, ["rev-parse", "HEAD"])
    origin = _run_git(repo_root, ["rev-parse", "origin/main"])
    if not _revision(head) or head != origin:
        _fail()
    if candidate_revision is not None and head != candidate_revision:
        _fail()
    return head


def _stable_fingerprint(parts: Sequence[str]) -> str:
    if not parts or any(type(item) is not str or not item for item in parts):
        _fail()
    return _sha256_bytes("\0".join(parts).encode("utf-8"))


def host_fingerprint() -> str:
    try:
        facts = os.uname()
        return _stable_fingerprint((facts.sysname, facts.nodename, facts.machine))
    except OSError:
        _fail()


def account_fingerprint() -> str:
    try:
        account = pwd.getpwuid(os.getuid())
        return _stable_fingerprint((str(os.getuid()), account.pw_name, account.pw_dir))
    except (KeyError, OSError):
        _fail()


def credential_fingerprint(api_key: str) -> str:
    if type(api_key) is not str or not api_key:
        _fail()
    return _sha256_bytes(api_key.encode("utf-8"))


def _default_authority_directory() -> Path:
    try:
        home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    except (KeyError, OSError):
        _fail()
    if not home.is_absolute():
        _fail()
    return (
        home
        / ".local/state/ai-context-manager/task12b-authority"
        / CONTRACT_LINEAGE_DIRECTORY
    )


def _canonical_local_root(repo_root: Path) -> Path:
    return repo_root / LOCAL_OUTPUT_PATH


def _load_credential(repo_root: Path) -> str:
    """Reuse Task 12a's ignored 0600 loader without importing the SDK."""
    try:
        probe = importlib.import_module(
            "experiments.adaptive_selection.openai_manifest_probe"
        )
        return cast(str, probe._load_ignored_api_key(repo_root))
    except Exception:
        _fail()


def _scan_committed_secret(repo_root: Path, api_key: str) -> None:
    try:
        probe = importlib.import_module(
            "experiments.adaptive_selection.openai_manifest_probe"
        )
        probe._assert_api_key_absent_from_git(repo_root, api_key)
    except Exception:
        _fail()


def _paid_environment_names() -> Tuple[str, ...]:
    dynamic = tuple(name for name in os.environ if name.upper().startswith("OPENAI_"))
    return tuple(
        sorted(
            set(dynamic)
            | set(EXPLICIT_OPENAI_ENVIRONMENT_NAMES)
            | set(PROXY_ENVIRONMENT_NAMES)
            | set(TLS_ENVIRONMENT_NAMES)
        )
    )


@contextmanager
def paid_ambient_guard() -> Iterator[None]:
    """Neutralize routing, TLS, proxy, and logging state for the whole paid section."""
    names = _paid_environment_names()
    saved = {name: os.environ.get(name) for name in names}
    previous = logging.root.manager.disable
    for name in names:
        os.environ.pop(name, None)
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        logging.disable(previous)


def _load_live_modules() -> Tuple[Any, Any]:
    try:
        openai = importlib.import_module("openai")
        httpx = importlib.import_module("httpx")
    except Exception:
        _fail()
    if (
        getattr(openai, "__version__", None) != PINNED_OPENAI_VERSION
        or getattr(httpx, "__version__", None) != PINNED_HTTPX_VERSION
    ):
        _fail()
    return openai, httpx


def _create_live_client(api_key: str) -> Any:
    """Build only the official HTTPS Responses client; caller holds ambient guard."""
    openai, httpx = _load_live_modules()
    http_client: Any = None
    try:
        ssl_context = ssl.create_default_context()
        if getattr(ssl_context, "keylog_filename", None) is not None:
            _fail()
        http_client = httpx.Client(
            verify=ssl_context,
            timeout=30.0,
            trust_env=False,
            follow_redirects=False,
        )
        return openai.OpenAI(
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
        if http_client is not None:
            try:
                http_client.close()
            except Exception:
                pass
        _fail()


def _member(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _safe_request_id(value: Any) -> Optional[str]:
    if (
        type(value) is str
        and value != ""
        and len(value) <= 200
        and re.fullmatch(r"[A-Za-z0-9_-]+", value) is not None
    ):
        return value
    return None


def _safe_observed_text(value: Any) -> Optional[str]:
    if (
        type(value) is str
        and 0 < len(value) <= 200
        and all(32 <= ord(character) <= 126 for character in value)
    ):
        return value
    return None


def _response_metadata(
    response: Any,
    *,
    response_id: Any = None,
    observed_model: Any = None,
) -> Optional[Dict[str, Any]]:
    status = getattr(response, "status_code", None)
    if type(status) is not int or not 100 <= status <= 599:
        return None
    headers = getattr(response, "headers", None)
    content_type = headers.get("content-type") if headers is not None else None
    request_id = getattr(response, "request_id", None)
    if request_id is None and headers is not None:
        request_id = headers.get("x-request-id")
    return {
        "http_status": status,
        "content_type": _safe_observed_text(content_type),
        "provider_request_id": _safe_request_id(request_id),
        "response_id": _safe_request_id(response_id),
        "observed_model": _safe_observed_text(observed_model),
    }


def _http_evidence_from_exception(
    exc: BaseException,
) -> Tuple[Optional[bytes], Optional[Dict[str, Any]]]:
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    content = getattr(response, "content", None)
    if type(status) is not int or type(content) is not bytes:
        return None, None
    metadata = _response_metadata(response)
    if metadata is None:
        return None, None
    if not content or len(content) > MAX_RAW_RESPONSE_BYTES:
        return None, metadata
    return content, metadata


def _raw_text(document: Mapping[str, Any]) -> str:
    output = document.get("output")
    if type(output) is not list or not output:
        _fail("malformed_response")
    messages = []
    for item in output:
        if type(item) is not dict:
            _fail("malformed_response")
        if item.get("type") == "reasoning":
            if not set(item).issubset(
                {"id", "type", "summary", "encrypted_content", "status"}
            ):
                _fail("malformed_response")
            if "id" in item and _safe_request_id(item["id"]) is None:
                _fail("malformed_response")
            if "summary" in item:
                summary = item["summary"]
                if type(summary) is not list:
                    _fail("malformed_response")
                for entry in summary:
                    if (
                        type(entry) is not dict
                        or set(entry) != {"type", "text"}
                        or entry["type"] != "summary_text"
                        or type(entry["text"]) is not str
                        or len(entry["text"]) > MAX_RAW_RESPONSE_BYTES
                    ):
                        _fail("malformed_response")
            if "encrypted_content" in item and not (
                item["encrypted_content"] is None
                or type(item["encrypted_content"]) is str
            ):
                _fail("malformed_response")
            if "status" in item and item["status"] not in {None, "completed"}:
                _fail("malformed_response")
            continue
        if item.get("type") != "message":
            _fail("malformed_response")
        if not set(item).issubset({"id", "type", "status", "role", "content", "phase"}):
            _fail("malformed_response")
        if "id" in item and _safe_request_id(item["id"]) is None:
            _fail("malformed_response")
        if "status" in item and item["status"] not in {None, "completed"}:
            _fail("malformed_response")
        if "phase" in item and item["phase"] not in {None, "final_answer"}:
            _fail("malformed_response")
        messages.append(item)
    if len(messages) != 1 or messages[0].get("role") != "assistant":
        _fail("malformed_response")
    content = messages[0].get("content")
    if type(content) is not list or len(content) != 1 or type(content[0]) is not dict:
        _fail("malformed_response")
    if content[0].get("type") == "refusal" or content[0].get("refusal"):
        _fail("provider_refusal")
    if not set(content[0]).issubset({"type", "text", "annotations", "logprobs"}):
        _fail("malformed_response")
    if "annotations" in content[0] and content[0]["annotations"] != []:
        _fail("malformed_response")
    if "logprobs" in content[0] and content[0]["logprobs"] not in (None, []):
        _fail("malformed_response")
    if (
        content[0].get("type") != "output_text"
        or type(content[0].get("text")) is not str
    ):
        _fail("malformed_response")
    return cast(str, content[0]["text"])


def _printable_bounded(value: Any, maximum: int) -> bool:
    return (
        type(value) is str
        and 1 <= len(value) <= maximum
        and all(32 <= ord(character) <= 126 for character in value)
    )


def validate_task12b_response(value: Any) -> Dict[str, Any]:
    fields = {
        "diagnosis",
        "supporting_evidence_numbers",
        "missing_evidence",
        "confidence",
        "next_safe_actions",
        "actions_to_avoid",
    }
    if type(value) is not dict or set(value) != fields:
        _fail("invalid_response")
    if not _printable_bounded(value["diagnosis"], 400):
        _fail("invalid_response")
    numbers = value["supporting_evidence_numbers"]
    if (
        type(numbers) is not list
        or len(numbers) > 5
        or any(type(item) is not int or not 1 <= item <= 5 for item in numbers)
    ):
        _fail("invalid_response")
    for field, maximum_items, maximum_length, minimum_items in (
        ("missing_evidence", 3, 120, 0),
        ("next_safe_actions", 3, 160, 1),
        ("actions_to_avoid", 3, 160, 1),
    ):
        items = value[field]
        if (
            type(items) is not list
            or not minimum_items <= len(items) <= maximum_items
            or any(not _printable_bounded(item, maximum_length) for item in items)
        ):
            _fail("invalid_response")
    if value["confidence"] not in {"low", "medium", "high"}:
        _fail("invalid_response")
    return cast(Dict[str, Any], value)


def _exact_nonnegative(value: Any) -> int:
    if type(value) is not int or value < 0 or value > 2**63 - 1:
        _fail("invalid_response")
    return cast(int, value)


def _raw_usage(document: Mapping[str, Any]) -> Dict[str, Any]:
    raw = document.get("usage")
    if type(raw) is not dict:
        _fail("invalid_response")
    values = {
        name: _exact_nonnegative(raw.get(name))
        for name in ("input_tokens", "output_tokens", "total_tokens")
    }
    if values["total_tokens"] != values["input_tokens"] + values["output_tokens"]:
        _fail("invalid_response")
    details = raw.get("input_tokens_details")
    if details is not None and type(details) is not dict:
        _fail("invalid_response")
    cached = (
        _exact_nonnegative(details["cached_tokens"])
        if details is not None and "cached_tokens" in details
        else 0
    )
    cache_write: Optional[int] = None
    if details is not None and "cache_write_tokens" in details:
        cache_write = _exact_nonnegative(details["cache_write_tokens"])
    if cached > values["input_tokens"] or (
        cache_write is not None and cached + cache_write > values["input_tokens"]
    ):
        _fail("invalid_response")
    return {
        "input_tokens": values["input_tokens"],
        "cached_input_tokens": cached,
        "cache_write_input_tokens": cache_write,
        "output_tokens": values["output_tokens"],
        "total_tokens": values["total_tokens"],
    }


def _member_presence(value: Any, name: str) -> Tuple[bool, Any]:
    if isinstance(value, Mapping):
        return name in value, value.get(name)
    if value is None:
        return False, None
    fields_set = getattr(value, "model_fields_set", None)
    if not isinstance(fields_set, (set, frozenset)):
        fields_set = getattr(value, "__fields_set__", None)
    if isinstance(fields_set, (set, frozenset)):
        return name in fields_set, getattr(value, name, None)
    return hasattr(value, name), getattr(value, name, None)


def validate_usage(document: Mapping[str, Any], response: Any) -> Dict[str, Any]:
    usage = _raw_usage(document)
    raw = cast(Mapping[str, Any], document["usage"])
    sdk = getattr(response, "usage", None)
    if sdk is None:
        _fail("invalid_response")
    for name in ("input_tokens", "output_tokens", "total_tokens"):
        if type(_member(sdk, name)) is not int or _member(sdk, name) != usage[name]:
            _fail("invalid_response")
    raw_details_present = "input_tokens_details" in raw
    sdk_details_present, sdk_details = _member_presence(sdk, "input_tokens_details")
    if raw_details_present != sdk_details_present:
        _fail("invalid_response")
    details = raw.get("input_tokens_details")
    if details is None:
        if raw_details_present and sdk_details is not None:
            _fail("invalid_response")
        return usage
    if sdk_details is None:
        _fail("invalid_response")
    for raw_name, sdk_name, projected_name in (
        ("cached_tokens", "cached_tokens", "cached_input_tokens"),
        ("cache_write_tokens", "cache_write_tokens", "cache_write_input_tokens"),
    ):
        raw_present = raw_name in details
        sdk_present, sdk_value = _member_presence(sdk_details, sdk_name)
        if raw_present != sdk_present:
            _fail("invalid_response")
        if raw_present and sdk_value != usage[projected_name]:
            _fail("invalid_response")
    return usage


def _exact_json_object(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _load_raw_document(raw_bytes: bytes) -> Dict[str, Any]:
    if (
        type(raw_bytes) is not bytes
        or not raw_bytes
        or len(raw_bytes) > MAX_RAW_RESPONSE_BYTES
    ):
        _fail("malformed_response")

    try:
        document = json.loads(raw_bytes, object_pairs_hook=_exact_json_object)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
        _fail("malformed_response")
    if type(document) is not dict:
        _fail("malformed_response")
    return cast(Dict[str, Any], document)


def _replay_raw_success(raw_bytes: bytes) -> Dict[str, Any]:
    document = _load_raw_document(raw_bytes)
    response_id = document.get("id")
    if _safe_request_id(response_id) is None:
        _fail("malformed_response")
    if document.get("model") != PINNED_MODEL:
        _fail("invalid_response")
    if (
        document.get("status") == "incomplete"
        or document.get("incomplete_details") is not None
    ):
        _fail("invalid_response")
    if document.get("status") != "completed":
        _fail("invalid_response")
    text = _raw_text(document)
    try:
        structured = json.loads(text, object_pairs_hook=_exact_json_object)
    except (TypeError, ValueError, json.JSONDecodeError):
        _fail("invalid_response")
    structured = validate_task12b_response(structured)
    usage = _raw_usage(document)
    if usage["output_tokens"] > 2048:
        _fail("invalid_response")
    actual, upper = usage_costs(usage)
    return {
        "response_id": response_id,
        "observed_model": document["model"],
        "structured_response": structured,
        "usage": usage,
        "actual_cost": actual,
        "conservative_cost_upper_bound": upper,
        "output_text": text,
    }


def usage_costs(usage: Mapping[str, Any]) -> Tuple[Optional[str], str]:
    try:
        input_tokens = Decimal(usage["input_tokens"])
        cached = Decimal(usage["cached_input_tokens"])
        output = Decimal(usage["output_tokens"])
        raw_write = usage["cache_write_input_tokens"]
        upper = (input_tokens * Decimal("2.50") + output * Decimal("12.00")) / Decimal(
            1_000_000
        )
        actual: Optional[Decimal] = None
        if raw_write is not None:
            cache_write = Decimal(raw_write)
            ordinary = input_tokens - cached - cache_write
            if ordinary < 0:
                _fail("invalid_response")
            actual = (
                ordinary * Decimal("2.00")
                + cached * Decimal("0.20")
                + cache_write * Decimal("2.50")
                + output * Decimal("12.00")
            ) / Decimal(1_000_000)
    except (InvalidOperation, KeyError, TypeError, ValueError):
        _fail("invalid_response")
    return (format(actual, "f") if actual is not None else None, format(upper, "f"))


@dataclass(frozen=True)
class TransportResult:
    raw_bytes: bytes
    response_metadata: Dict[str, Any]
    structured_response: Dict[str, Any]
    usage: Dict[str, Any]
    actual_cost: Optional[str]
    conservative_cost_upper_bound: str


def dispatch_once(
    client: Any, body: Mapping[str, Any], request_hash: str
) -> TransportResult:
    """Perform exactly one injected-client call and strictly validate raw and SDK views."""
    literal = json.loads(canonical_bytes(body).decode("utf-8"))
    if _sha256_bytes(canonical_bytes(literal)) != request_hash:
        _fail()
    try:
        raw = client.responses.with_raw_response.create(**literal)
    except Exception as exc:
        raw_bytes, metadata = _http_evidence_from_exception(exc)
        failure = ExecutionFailure(
            "http_error" if metadata is not None else "transport_error"
        )
        failure.raw_bytes = raw_bytes
        failure.response_metadata = metadata
        raise failure from None

    raw_bytes: Optional[bytes] = None
    metadata = _response_metadata(raw)
    document: Any = None
    response: Any = None
    try:
        content = raw.content
        if type(content) is not bytes or not content:
            _fail("malformed_response")
        if len(content) > MAX_RAW_RESPONSE_BYTES:
            _fail("malformed_response")
        raw_bytes = content
        document = _load_raw_document(raw_bytes)
        response = raw.parse()
        response_id = getattr(response, "id", None)
        observed_model = getattr(response, "model", None)
        metadata = _response_metadata(
            raw, response_id=response_id, observed_model=observed_model
        )
        if (
            metadata is None
            or not _valid_success_transport_metadata(metadata)
            or document.get("id") != response_id
        ):
            _fail("malformed_response")
        if observed_model != PINNED_MODEL or document.get("model") != observed_model:
            _fail("invalid_response")
        status = getattr(response, "status", None)
        if (
            status == "incomplete"
            or document.get("status") == "incomplete"
            or getattr(response, "incomplete_details", None) is not None
            or document.get("incomplete_details") is not None
        ):
            _fail("invalid_response")
        if status != "completed" or document.get("status") != "completed":
            _fail("invalid_response")
        projection = _replay_raw_success(raw_bytes)
        if getattr(response, "output_text", None) != projection["output_text"]:
            _fail("invalid_response")
        usage = validate_usage(document, response)
        if usage != projection["usage"]:
            _fail("invalid_response")
        return TransportResult(
            raw_bytes,
            metadata,
            projection["structured_response"],
            usage,
            projection["actual_cost"],
            projection["conservative_cost_upper_bound"],
        )
    except ExecutionFailure as failure:
        failure.raw_bytes = raw_bytes
        failure.response_metadata = metadata
        if type(document) is dict and response is not None:
            try:
                failure.usage = validate_usage(document, response)
                failure.actual_cost, failure.conservative_cost_upper_bound = (
                    usage_costs(failure.usage)
                )
            except ExecutionFailure:
                pass
        raise failure from None
    except Exception:
        failure = ExecutionFailure("malformed_response")
        failure.raw_bytes = raw_bytes
        failure.response_metadata = metadata
        if type(document) is dict and response is not None:
            try:
                failure.usage = validate_usage(document, response)
                failure.actual_cost, failure.conservative_cost_upper_bound = (
                    usage_costs(failure.usage)
                )
            except ExecutionFailure:
                pass
        raise failure from None


def _accepted_upper(global_root: Path, context: AuthorityContext) -> Decimal:
    total = Decimal(0)
    for cell_id, request_hash in zip(EXECUTION_ORDER, REQUEST_HASHES):
        if (
            classify_state(
                global_root,
                Path("/__unused_local_mirror__"),
                cell_id,
                request_hash,
                context,
            )
            == "terminal"
        ):
            terminal = verify_terminal(global_root, cell_id, request_hash, context)[
                "terminal"
            ]
            value = terminal["conservative_cost_upper_bound"]
            if value is not None:
                total += Decimal(value)
    return total


def _projected_input_bounds(requests: Sequence[Mapping[str, Any]]) -> List[int]:
    return [len(canonical_bytes(body)) + 1024 for body in requests]


def _static_dispatch_upper(projected_input_tokens: int) -> str:
    upper = (
        Decimal(projected_input_tokens) * Decimal("2.50")
        + Decimal(2048) * Decimal("12.00")
    ) / Decimal(1_000_000)
    return format(upper, "f")


def _run_summary(
    global_root: Path,
    local_root: Path,
    context: AuthorityContext,
    dispatches: int,
    *,
    resumed_all_terminal: bool = False,
) -> Dict[str, Any]:
    successes = 0
    failures = 0
    pending = 0
    for cell_id, request_hash in zip(EXECUTION_ORDER, REQUEST_HASHES):
        state = classify_state(global_root, local_root, cell_id, request_hash, context)
        if state == "pending":
            pending += 1
            continue
        if state != "terminal":
            _fail()
        terminal = verify_terminal(global_root, cell_id, request_hash, context)[
            "terminal"
        ]
        if terminal["kind"] == "success":
            successes += 1
        else:
            failures += 1
    if resumed_all_terminal:
        status = "all_terminal"
    elif failures == 0 and pending == 0:
        status = "run_complete"
    else:
        status = "run_incomplete"
    return {
        "status": status,
        "dispatches": dispatches,
        "successes": successes,
        "failures": failures,
        "pending": pending,
    }


def execute_authorized_manifest(
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
    _clock: Optional[Callable[[], str]] = None,
) -> Dict[str, Any]:
    """Execute pending cells under verified authority; tests inject every dependency."""
    requests, visible_evidence = _validate_requests(repo_root)
    revision = repository_preflight(repo_root)
    context = verify_authority(
        candidate_path,
        approval_path,
        mapping_path,
        revision,
        host_fingerprint(),
        account_fingerprint(),
        credential_fingerprint(api_key),
        now,
    )
    bounds = _projected_input_bounds(requests)
    clock = _clock or _utc_now
    states: List[str] = []
    with machine_global_run_lock(global_root):
        for cell_id, request_hash in zip(EXECUTION_ORDER, REQUEST_HASHES):
            index = len(states)
            state = classify_state(
                global_root, local_root, cell_id, request_hash, context
            )
            if state == "terminal":
                verify_terminal(
                    global_root,
                    cell_id,
                    request_hash,
                    context,
                    visible_evidence[index],
                )
            states.append(state)
        if "blocked_orphan_claim" in states:
            _fail("blocked_orphan_claim_no_network_attempt")
        pending = [index for index, state in enumerate(states) if state == "pending"]
        if not pending:
            return _run_summary(
                global_root, local_root, context, 0, resumed_all_terminal=True
            )
        accepted_upper = _accepted_upper(global_root, context)
        unattempted_ceiling = sum(
            Decimal(_static_dispatch_upper(bounds[index])) for index in pending
        )
        if accepted_upper + unattempted_ceiling > Decimal(OWNER_CAP):
            _fail("budget_bound_violation_no_network_attempt")
        client: Any = None
        dispatches = 0
        try:
            client = client_factory(api_key)
            for index in pending:
                cell_id = EXECUTION_ORDER[index]
                request_hash = REQUEST_HASHES[index]
                body = requests[index]
                if _sha256_bytes(canonical_bytes(body)) != request_hash:
                    _fail()
                claim_at = clock()
                _require_live_authority(context, claim_at)
                publish_authority_claim(
                    global_root, local_root, cell_id, request_hash, context, claim_at
                )
                dispatches += 1
                dispatch_completed = False
                try:
                    result = dispatch_once(client, body, request_hash)
                    dispatch_completed = True
                    terminal_at = clock()
                    if api_key.encode("utf-8") in result.raw_bytes:
                        publish_terminal(
                            global_root,
                            local_root,
                            cell_id,
                            request_hash,
                            context,
                            kind="failure",
                            dispatch_invoked=True,
                            server_acceptance="yes",
                            provider_visible_evidence=visible_evidence[index],
                            structured_response=None,
                            raw_bytes=None,
                            recorded_at=terminal_at,
                            failure_category="secret_detected",
                            usage=result.usage,
                            actual_cost=result.actual_cost,
                            conservative_cost_upper_bound=result.conservative_cost_upper_bound,
                            response_metadata=result.response_metadata,
                        )
                        accepted_upper += Decimal(result.conservative_cost_upper_bound)
                        break
                    if result.usage["input_tokens"] > bounds[index]:
                        publish_terminal(
                            global_root,
                            local_root,
                            cell_id,
                            request_hash,
                            context,
                            kind="failure",
                            dispatch_invoked=True,
                            server_acceptance="yes",
                            provider_visible_evidence=visible_evidence[index],
                            structured_response=None,
                            raw_bytes=result.raw_bytes,
                            recorded_at=terminal_at,
                            failure_category="budget_bound_violation",
                            usage=result.usage,
                            actual_cost=result.actual_cost,
                            conservative_cost_upper_bound=result.conservative_cost_upper_bound,
                            response_metadata=result.response_metadata,
                        )
                        break
                    publish_terminal(
                        global_root,
                        local_root,
                        cell_id,
                        request_hash,
                        context,
                        kind="success",
                        dispatch_invoked=True,
                        server_acceptance="yes",
                        provider_visible_evidence=visible_evidence[index],
                        structured_response=result.structured_response,
                        raw_bytes=result.raw_bytes,
                        recorded_at=terminal_at,
                        usage=result.usage,
                        actual_cost=result.actual_cost,
                        conservative_cost_upper_bound=result.conservative_cost_upper_bound,
                        response_metadata=result.response_metadata,
                    )
                    accepted_upper += Decimal(result.conservative_cost_upper_bound)
                    remaining = [item for item in pending if item > index]
                    remaining_ceiling = sum(
                        Decimal(_static_dispatch_upper(bounds[item]))
                        for item in remaining
                    )
                    if accepted_upper + remaining_ceiling > Decimal(OWNER_CAP):
                        break
                except ExecutionFailure as failure:
                    if dispatch_completed:
                        raise
                    terminal_at = clock()
                    raw_bytes = getattr(failure, "raw_bytes", None)
                    metadata = getattr(failure, "response_metadata", None)
                    if raw_bytes is not None and api_key.encode("utf-8") in raw_bytes:
                        raw_bytes = None
                        failure = ExecutionFailure("secret_detected")
                        failure.response_metadata = metadata
                    usage = getattr(failure, "usage", None)
                    actual_cost = getattr(failure, "actual_cost", None)
                    upper = getattr(failure, "conservative_cost_upper_bound", None)
                    if upper is None:
                        upper = _static_dispatch_upper(bounds[index])
                        actual_cost = None
                    publish_terminal(
                        global_root,
                        local_root,
                        cell_id,
                        request_hash,
                        context,
                        kind="failure",
                        dispatch_invoked=True,
                        server_acceptance="yes" if metadata is not None else "unknown",
                        provider_visible_evidence=visible_evidence[index],
                        structured_response=None,
                        raw_bytes=raw_bytes,
                        recorded_at=terminal_at,
                        failure_category=failure.category,
                        usage=usage,
                        actual_cost=actual_cost,
                        conservative_cost_upper_bound=upper,
                        response_metadata=metadata,
                    )
                    accepted_upper += Decimal(upper)
                    remaining = [item for item in pending if item > index]
                    remaining_ceiling = sum(
                        Decimal(_static_dispatch_upper(bounds[item]))
                        for item in remaining
                    )
                    if (
                        usage is not None and usage["input_tokens"] > bounds[index]
                    ) or accepted_upper + remaining_ceiling > Decimal(OWNER_CAP):
                        break
                    continue
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    _fail("client_close_failure")
        return _run_summary(global_root, local_root, context, dispatches)


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    _repo_root: Optional[Path] = None,
    _global_root: Optional[Path] = None,
    _local_root: Optional[Path] = None,
    _api_key: Optional[str] = None,
    _client_factory: Callable[[str], Any] = _create_live_client,
    _now: Optional[str] = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Task 12b offline-first readiness gate"
    )
    commands = parser.add_mutually_exclusive_group()
    commands.add_argument(
        "--execute-authorized-nine-cell-manifest", action="store_true"
    )
    commands.add_argument("--prepare-authorization-candidate", action="store_true")
    commands.add_argument("--export-blind-assessment", action="store_true")
    commands.add_argument("--finalize-authorized-orphan", action="store_true")
    parser.add_argument("--owner-identity")
    parser.add_argument("--expires-at")
    parser.add_argument("--orphan-cell", choices=EXECUTION_ORDER)
    args = parser.parse_args(argv)
    repo_root = (_repo_root or Path.cwd()).resolve()
    local_root = _local_root or _canonical_local_root(repo_root)
    global_root = _global_root or _default_authority_directory()
    try:
        if (args.owner_identity is not None or args.expires_at is not None) and not (
            args.prepare_authorization_candidate
        ):
            _fail()
        if args.orphan_cell is not None and not args.finalize_authorized_orphan:
            _fail()
        if args.finalize_authorized_orphan and args.orphan_cell is None:
            _fail()
        if not any(
            (
                args.execute_authorized_nine_cell_manifest,
                args.prepare_authorization_candidate,
                args.export_blind_assessment,
                args.finalize_authorized_orphan,
            )
        ):
            build_dry_run_summary(repo_root)
            print(
                "task12b status=dry-run network_attempts=0 credential_readiness=not_checked"
            )
            return 0
        clock = (lambda: cast(str, _now)) if _now is not None else _utc_now
        now = clock()
        candidate_path = local_root / "authorization-candidate.json"
        approval_path = local_root / "owner-approval.json"
        mapping_path = local_root / "blind-mapping.json"
        if args.prepare_authorization_candidate:
            if not args.owner_identity or not args.expires_at:
                _fail()
            requests, _ = _validate_requests(repo_root)
            revision = repository_preflight(repo_root)
            api_key = _api_key or _load_credential(repo_root)
            _scan_committed_secret(repo_root, api_key)
            if (
                tuple(_sha256_bytes(canonical_bytes(body)) for body in requests)
                != REQUEST_HASHES
            ):
                _fail()
            issued = _parse_time(now)
            expires = _parse_time(args.expires_at)
            window = int((expires - issued).total_seconds())
            candidate = prepare_non_authorizing_candidate(
                candidate_path,
                mapping_path,
                code_revision=revision,
                owner_identity=args.owner_identity,
                host_fingerprint=host_fingerprint(),
                account_fingerprint=account_fingerprint(),
                credential_fingerprint=credential_fingerprint(api_key),
                issued_at=now,
                expires_at=args.expires_at,
                maximum_execution_window_seconds=window,
            )
            print(
                "task12b status=candidate-prepared network_authority=0 candidate_sha256=%s"
                % candidate["candidate_sha256"]
            )
            return 0
        if args.execute_authorized_nine_cell_manifest:
            api_key = _api_key or _load_credential(repo_root)
            _scan_committed_secret(repo_root, api_key)
            with paid_ambient_guard():
                result = execute_authorized_manifest(
                    repo_root,
                    global_root,
                    local_root,
                    candidate_path,
                    approval_path,
                    mapping_path,
                    api_key=api_key,
                    client_factory=_client_factory,
                    now=now,
                    _clock=clock,
                )
            print(
                "task12b status=%s dispatches=%d successes=%d failures=%d pending=%d"
                % (
                    result["status"],
                    result["dispatches"],
                    result["successes"],
                    result["failures"],
                    result["pending"],
                )
            )
            return 0
        api_key = _api_key or _load_credential(repo_root)
        _scan_committed_secret(repo_root, api_key)
        candidate = read_private_record(candidate_path)["candidate"]
        revision = repository_preflight(
            repo_root, cast(str, candidate.get("execution_code_revision"))
        )
        context = verify_historical_authority(
            candidate_path,
            approval_path,
            mapping_path,
            revision,
            host_fingerprint(),
            account_fingerprint(),
            credential_fingerprint(api_key),
        )
        if args.export_blind_assessment:
            with machine_global_run_lock(global_root):
                packet = build_blind_assessment(
                    repo_root, global_root, local_root, mapping_path, context
                )
                _write_private_bytes_no_overwrite(
                    local_root / "blind-assessment.json",
                    canonical_bytes(packet) + b"\n",
                )
            print(
                "task12b status=blind-assessment-exported assessment_ready=%s network_attempts=0"
                % str(packet["assessment_ready"]).lower()
            )
            return 0
        if args.finalize_authorized_orphan:
            assert args.orphan_cell is not None
            index = EXECUTION_ORDER.index(args.orphan_cell)
            with machine_global_run_lock(global_root):
                finalize_orphan_offline(
                    global_root,
                    local_root,
                    args.orphan_cell,
                    REQUEST_HASHES[index],
                    context,
                    local_root / "owner-orphan-finalization.json",
                    now,
                    _lock_held=True,
                )
            print("task12b status=orphan-finalized network_attempts=0")
            return 0
        _fail("offline_command_requires_verified_private_authority")
    except Exception as exc:
        category = (
            exc.category if type(exc) is ExecutionFailure else "sanitized_failure"
        )
        print("task12b status=blocked category=%s network_details=withheld" % category)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
