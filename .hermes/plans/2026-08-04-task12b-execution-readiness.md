# Task 12b Execution Readiness Implementation Plan

**Plan date:** 2026-08-04

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add a separately gated, resumable, exactly-once nine-cell Task 12b execution and evidence path without authorizing or making any provider request during development, tests, review, or CI.

**Architecture:** Keep the frozen offline calibration contract and renderer unchanged. Add a hash-pinned execution-readiness manifest and a separate runner that reuses Task 12a's proven credential, SDK, TLS, no-retry, and raw-response patterns while strengthening its filesystem and paid-section isolation. An offline preparation step creates a nonce-bearing, expiring authorization candidate plus a private randomized blind-ID mapping; it cannot authorize execution. The owner must later echo the exact candidate SHA-256 out of band, and the operator records that exact approval digest before live mode can proceed. Per-cell machine-global authority claims are atomically written immediately before dispatch and are the authoritative cooperative replay interlock; repository-local copies are audit mirrors. Normal resume never auto-finalizes marker-only state. A separate owner-approved offline recovery action is required after proving no runner remains. Terminal artifacts are atomic no-overwrite cooperative evidence, not adversarially immutable files.

**Tech Stack:** Python 3.9-compatible standard library, existing `openai==2.46.0` and `httpx==0.28.1`, existing Task 12a transport primitives, pytest, Black, Ruff.

**Absolute prohibition:** All implementation and verification use fake injected clients/transports. Do not invoke the live CLI mode, read a real API key, or make any provider/network request.

---

### Task 1: Add the frozen execution-readiness manifest

**Objective:** Bind the unchanged calibration contract to the exact nine request hashes, order, provider settings, current frozen price assumptions, cap, private paths, authority schema, failure policy, and protected paths.

**Files:**
- Create: `experiments/adaptive_selection/controls/task12b_execution_readiness_v1.json`
- Test: `tests/adaptive_selection/test_context_sensitivity_execution.py`

**Steps:**
1. Write failing tests that require exact manifest fields, nine ordered request hashes, zero authority from the committed manifest, and the unchanged contract hash `0bf617…`.
2. Generate the manifest from the existing renderer's deterministic output and the official pricing review. Preserve `$2.00/M` input and `$12.00/M` output as the frozen contract's historical projection, but use the current GPT-5.6 automatic cache-write rate `$2.50/M` as the readiness ceiling: 29,192 conservative input tokens + 18,432 output tokens = `$0.294164`. Also record the even more conservative long-context cache-write/output sensitivity bound (`$5.00/M` and `$18.00/M`) of `$0.477736`; both remain below the immutable `$1.00` owner cap. The readiness code must not continue calling `$0.279568` a maximum.
3. Pin its byte SHA-256 in the runner and test mutation rejection.
4. Keep `network_requests_authorized_by_manifest` at zero; live authority comes only from the separate private authorization record.
5. Run the focused tests and verify the contract bytes and existing 67-test suite remain unchanged.

### Task 2: Implement strict private authority, marker, ledger, and artifact records

**Objective:** Enforce explicit owner approval, exact code/manifests, private permissions, no overwrite, cross-checkout attempt consumption, and resumable no-retry state classification.

**Files:**
- Create: `experiments/adaptive_selection/context_sensitivity_execution.py`
- Test: `tests/adaptive_selection/test_context_sensitivity_execution.py`

**Required records:**
- Private authorization candidate with exact fields: version, unique authorization ID and random nonce, issuance/expiry timestamps and maximum execution window, frozen contract-lineage identity, readiness-manifest hash, exact execution code revision, ordered request hashes, requested model, SDK/HTTP versions, reasoning effort, max output tokens, independently frozen short-context input/cache-read/cache-write/output rates, maximum approved total cost, approved cell count, fixed single-host/account scope, non-secret credential fingerprint, host/account fingerprint, blind-mapping commitment, fixed no-retry policy, and owner identity string. The candidate is canonical `0600` evidence but grants zero authority. Execution additionally requires a separate approval record containing the exact candidate SHA-256 that the owner echoed out of band; code cannot generate or infer that owner echo.
- Private precommitted randomized mapping from canonical cell IDs to unrelated assessment IDs and a blind order independent of execution order. Its canonical hash is bound into the authorization candidate; it is never included in assessor exports and stays private until annotations are locked.
- Per-cell machine-global authority-consumption claims with exact contract-lineage/manifest/code/request/cell/authorization-ID/approval-digest/timestamp fields and canonical hashes. Claims state `authority_consumed=true`; they do not claim dispatch. Repository-local claim copies are non-authoritative audit mirrors. Acquire and hold a nonblocking machine-global account-local run lock for the entire live reconciliation/execution pass; process exit releases the lock.
- One canonical machine-global terminal index plus repository-local terminal copies. Success/failure records distinguish `dispatch_invoked` (`true`, `false`, or `unknown`) from server acceptance (`yes`, `no`, or `unknown`) and bind authorization, request, code, cell, usage, cost, and raw hashes. Raw bytes are stored exactly once.
- Explicit offline recovery/finalization authorization for a global claim without a terminal record. Ordinary resume blocks and writes nothing. After the owner confirms no runner remains and approves the exact orphan claim digest, recovery may publish one global `invalid_ambiguous` terminal record and its local audit copy; it never dispatches.

**State machine:**
- no global claim/no terminal: pending;
- valid global claim plus valid global terminal: terminal and skipped; missing local copies may be recreated only as verified audit mirrors;
- valid global claim but no global terminal: blocked orphan claim; ordinary live/resume exits without dispatch or mutation and requires separately approved offline finalization;
- local claim without the matching global claim, terminal without claim, conflicting records, raw-only state, terminal-only raw reference, or any semantic/hash mismatch: abort preflight;
- all terminal: return a zero-dispatch completed summary.

**TDD cases:** permissions; final and ancestor symlinks; hard links; wrong owner/mode/link count; parent replacement; malformed/tampered/rehashed envelopes; missing or wrong owner-echo digest; expired/window-invalid authorization; wrong nonce/host/account/credential fingerprint; path escape; alternate checkout; atomic no-overwrite; cross-checkout concurrent claim winner under the run lock; orphan claim blocking; separately approved zero-network finalization; completed resume; failure continuation; partial raw/terminal publication; artifact-write failure; wrong revision; wrong request hash/order/model/rates/cap; and no secret in raw bytes, payloads, exceptions, logs, or traces.

### Task 3: Implement the single-attempt OpenAI transport and live gate

**Objective:** Provide the later executable path while proving development and default use remain offline.

**Files:**
- Modify: `experiments/adaptive_selection/context_sensitivity_execution.py`
- Test: `tests/adaptive_selection/test_context_sensitivity_execution.py`

**Steps:**
1. Reuse Task 12a's official-host client and credential patterns, but keep every OpenAI routing/org/project/header variable, `OPENAI_LOG`, proxy variable, `SSL_CERT_FILE`, `SSL_CERT_DIR`, and `SSLKEYLOGFILE` removed—and process-wide logging disabled—from before SDK import/TLS construction until after raw handling, terminal persistence, and client close. Restore only in `finally`. Require exact `openai==2.46.0` and `httpx==0.28.1`, explicit TLS with no key log, `trust_env=False`, redirects disabled, `max_retries=0`, timeout, ignored owner-private `.env`, credential fingerprint, and committed-snapshot secret scan.
2. Render all nine bodies through the unchanged calibration module and verify byte hashes against both committed readiness manifest and private authorization. Live preflight must require branch `main`, a completely clean tracked tree, and `HEAD == origin/main == authorization.code_revision` using already-fetched local refs. It must not fetch, test, build, install, or resolve dependencies inside the paid process; those gates run externally against the exact committed SHA.
3. Expose only one conspicuous live flag: `--execute-authorized-nine-cell-manifest`; default CLI performs prompt-free preflight/dry-run and never imports OpenAI, reads credentials, constructs a client, or dispatches.
4. Obtain a fresh trusted UTC observation before every cell, reject without claim or dispatch after expiry, then atomically publish the global authority-consumption claim and its local audit mirror immediately before `responses.with_raw_response.create`. If global claim publication loses a race or local mirroring fails, make zero provider calls and stop; never roll back the global claim and never describe claim existence alone as a dispatch.
5. Validate request/body identity, raw response size/duplicate-free JSON/status/model/IDs/refusal, exact no-action output-item allowlists, exact SDK/raw output text, raw/SDK usage-detail presence and values (including `usage.input_tokens_details.cached_tokens` and GPT-5.6 `cache_write_tokens` when reported), and Task 12b bounded response schema. Reparse raw bytes during terminal verification, independently replay every success projection and pinned-rate cost, and require exact frozen provider-visible evidence at resume/export. Stop before later cells if usage exceeds a per-cell projection or invalidates cap assumptions; describe `$1.00` as an enforced projection/accepted-usage cap, not a billing guarantee.
6. Continue to later never-attempted cells after a recorded provider failure; stop safely on local evidence-write failure. Never retry, replace, fall back, or alter a request.
7. Tests use injected fake raw responses for state coverage and the real frozen OpenAI SDK with `httpx.MockTransport` for success, 429, 500, redirect, connection failure, and malformed-response cases. Prove at most one HTTP handler invocation, exact official host/path/method/body, no redirect follow, exact runtime versions, stateless cells, no conversation/previous-response state, and no retry/fallback/replacement. Do not run the live flag against a real client.

### Task 4: Add blind export and operational documentation

**Objective:** Produce a condition-blind assessment packet only from verified terminal artifacts and document the exact preflight/recovery/approval procedure.

**Files:**
- Modify: `experiments/adaptive_selection/context_sensitivity_execution.py`
- Create: `docs/research/task12b-execution-readiness.md`
- Modify: `docs/research/task12b-context-sensitivity-calibration.md`
- Test: `tests/adaptive_selection/test_context_sensitivity_execution.py`

**Steps:**
1. Add offline `--export-blind-assessment` that verifies all nine global terminal records and the private blind-mapping commitment. In precommitted blind order, emit only randomized assessment ID, task, provider-visible timestamped evidence, structured response or invalid/failure status, frozen condition-independent rubric criteria, adjudication rules, and critical findings. Exclude canonical cell ID, condition, family/scenario IDs, condition anchors, provenance/source roles, execution order/position/timestamps/latency, request/response IDs, hashes, paths, and filenames. The export must set `assessment_ready=true` only when all nine records are valid successes. Any failed or ambiguous cell remains audit-visible but forces `assessment_ready=false`; it must never be imputed, annotated as a synthetic zero, or passed to the frozen scorer, and no preregistered verdict is derived.
2. Require all nine cells to be terminal and verify every artifact before export.
3. Document separate gates: preparation merge, readiness merge, exact owner authorization, execution, blind assessment, deterministic scoring.
4. Document crash semantics: a claim-only state blocks ordinary resume, is never automatically labeled or retried, and requires separate owner-approved zero-network finalization after proving no runner remains.
5. Document that machine-global claims are cooperative same-account/same-host interlocks, not adversarial proof against deletion, rewriting by the same OS user, or another host. Owner approval is operationally established by echoing the exact nonce-bearing authorization-candidate digest; local unkeyed hashes and modes alone do not prove intent.
6. Document exact private paths and preservation requirements.

### Task 5: Exact-candidate verification and publication

**Objective:** Establish that the PR is offline-ready, scientifically faithful, security-reviewed, and still unauthorized for live calls.

**Files:** all above.

**Steps:**
1. Run focused execution tests, existing 67 Task 12b tests, Task 12a probe tests, full repository suite, Black check, Ruff, compileall, package build, offline dry-run, and prompt/privacy scans.
2. Confirm no output/authority markers were created in canonical real paths and no provider client/network activity occurred.
3. Conduct separate specification/scientific review, then security/code-quality review. Fix all blockers and rerun reviews.
4. Freeze final readiness-manifest hash and exact request hashes in code/docs/tests atomically.
5. Scan the complete branch range for private values and ensure no raw/private evidence is tracked.
6. Commit, push, open one PR, wait for CI, and report exact commit, manifest hash, request hashes, projected cap, and remaining explicit owner-authorization gate.
7. Do not merge and do not execute any live request without a later exact user instruction.
