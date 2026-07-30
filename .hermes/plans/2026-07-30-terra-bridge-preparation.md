# Terra One-Call Development Probe Plan

> **Execution authority:** one benign network-request attempt only. This plan does not authorize calibration calls, DevOps task execution, selector comparison, adaptation, held-out work, or the 99-call bridge trial.

**Goal:** Resolve OpenAI `gpt-5.6-terra` into an auditable Responses API configuration and prove that the existing provider evidence seam can capture one strict structured response without leaking credentials or fabricating unsupported controls.

**Repository:** `/Users/andrewhamilton/Projects/hamiltonhaus/.worktrees/ai-context-manager-terra-bridge`

**Branch/base:** `research/adaptive-context-terra-bridge` from freshly fetched `origin/main` commit `97151920884aeae54846882765a63b4993f0614f`.

**Interpreter:** `/Users/andrewhamilton/Projects/hamiltonhaus/ai-context-manager/.venv/bin/python`

## Scientific and execution boundary

- This is a prospective Task 12 preparation subtask, not corpus authoring or hosted-model efficacy execution.
- The prompt contains no DevOps scenario, selector condition, adaptation feedback, labels, required-context set, answer key, or held-out material.
- The first attempted network request consumes the sole authorization, regardless of timeout, connection ambiguity, HTTP rejection, refusal, incomplete response, parse failure, or success.
- No retry, replacement probe, or amended second attempt is pre-authorized by this plan or contract lineage.
- A future request would require a wholly new, explicitly owner-approved protocol outside this contract.
- A passing probe permits only a non-operative future calibration proposal. It does not authorize a calibration call.

## Known implementation constraints

1. Reuse `ProviderConfiguration`, `ProviderRequest`, `RawTransportResult`, `RecordedCallbackProvider`, and `ProviderExecution`.
2. OpenAI SDK `2.46.0` is installed and exposes `responses.with_raw_response.create`.
3. The Responses object exposes `model` but no distinct revision or system fingerprint. Revision must become nullable and remain `None`/JSON `null`; no placeholder is allowed.
4. Temperature is intentionally omitted. Provider configuration and run-manifest temperature must become nullable with `temperature_supported=False`.
5. Seed remains `None` with `seed_supported=False`; the installed Responses create method has no seed parameter.
6. Exact body evidence means `LegacyAPIResponse.content`: decoded HTTP entity-body bytes exposed by the SDK, not SDK reserialization and not guaranteed TCP/wire bytes.
7. `LegacyAPIResponse.request_id`/`x-request-id` and parsed `Response.id` are distinct and both mandatory.
8. Usage types must be validated against raw JSON before SDK coercion and cross-checked with the parsed response.
9. OpenAI clients default to retries; the live client must use `max_retries=0`, and fake-transport tests must prove one HTTP attempt under retryable failures.
10. JSON is globally ignored; the contract must be force-added and verified as tracked.
11. The credential exists only in the ignored worktree `.env`, mode `0600`; no value may enter code, artifacts, logs, commits, or chat.

## Prospective order

1. Finalize and independently review the protocol, amendment, plan, and literal request manifest.
2. Compute the manifest SHA-256, pin it in the protocol, force-add the JSON, and commit the prospective contract before implementation.
3. Implement the schema migration, OpenAI callback, artifact writer/verifier, and fake-transport tests without a live call.
4. Independently review methods, code, security, and test coverage; remediate every finding.
5. Commit the implementation after the contract commit.
6. Verify protected paths, Git ancestry, hashes, tests, Python 3.9 grammar, build, secret scan, credential presence, projected cost, and empty output directory.
7. Run a dry-run that makes zero network calls.
8. Execute exactly one development request.
9. Atomically preserve and verify either success or failure evidence.
10. Record the factual result, push the branch, open a PR, report CI, and stop. No calibration call occurs under this plan.

---

### Task 1: Freeze the one-call contract

**Files:**

- `.hermes/plans/2026-07-30-terra-bridge-preparation.md`
- `docs/research/terra-bridge-development-protocol.md`
- `docs/research/adaptive-context-protocol-amendments.md`
- `experiments/adaptive_selection/controls/terra_probe_v1.json`

The JSON contains one literal `request_body` equal to the canonical JSON projection passed to `responses.with_raw_response.create`. Omitted controls are absent from that object. Client settings, evidence rules, and preflight rules are separate fields and are never transmitted as API-body fields.

The contract freezes:

- model `gpt-5.6-terra`;
- reasoning effort `medium`;
- strict JSON Schema output;
- service tier `default`;
- `store=false`, `stream=false`, no tools or state;
- 512-token output cap;
- omitted temperature, top-p, and seed;
- SDK `2.46.0`, timeout 30 seconds, `max_retries=0`;
- exact identity, status, refusal, usage, raw-body, request-ID, response-ID, and artifact rules;
- one request attempt across the entire contract lineage;
- fixed failure taxonomy and sanitized failure-artifact schema;
- protected Git path set and cleanliness/ancestry checks;
- prices and a conservative projected-cost guard.

**Cost projection:** canonical request-body UTF-8 byte length plus 1,024 provider-overhead tokens is treated as a conservative input-token projection and rejected above 4,096. At 4,096 input and 512 output tokens, projected cost is `$0.014336`; the `$0.25` threshold is a pre-dispatch projection guard, not a guaranteed billing cap.

**Freeze commands:**

```bash
python3 -m json.tool experiments/adaptive_selection/controls/terra_probe_v1.json >/dev/null
shasum -a 256 experiments/adaptive_selection/controls/terra_probe_v1.json
git diff --check
git add .hermes/plans/2026-07-30-terra-bridge-preparation.md \
  docs/research/terra-bridge-development-protocol.md \
  docs/research/adaptive-context-protocol-amendments.md
git add -f experiments/adaptive_selection/controls/terra_probe_v1.json
git ls-files --error-unmatch experiments/adaptive_selection/controls/terra_probe_v1.json
git commit -m "research: freeze Terra one-call development probe"
```

### Task 2: Migrate the provider evidence schema

**Files:**

- `experiments/adaptive_selection/providers.py`
- `experiments/adaptive_selection/schema.py`
- `tests/adaptive_selection/test_runner.py`
- `tests/adaptive_selection/test_schema.py`
- `tests/adaptive_selection/test_packaging.py`

Changes:

- The three existing fields—`ProviderConfiguration.provider_revision`, `RawTransportResult.observed_provider_revision`, and `ProviderExecution.provider_revision`—become optional. A new optional `provider_revision` field is added explicitly to `RunManifest`.
- Temperature becomes optional and receives an explicit `temperature_supported` boolean in provider configuration and run manifest.
- Probe configuration is `provider_revision=None`, `temperature=None`, `temperature_supported=False`, `seed=None`, `seed_supported=False`.
- Exact serialized shapes, schema version, and provider configuration hash domain/version are bumped prospectively. `build_run_manifest`, request/execution validation, serialization/round-trip tests, and relevant comparison/binding behavior must cover the new run-manifest field.
- Strict provider and model identity matching remains unchanged.
- Compatibility and round-trip tests prove that absence remains JSON `null`, participates in hashes, and cannot be replaced with a fabricated value.

### Task 3: Implement the minimal one-call probe

**Files:**

- `experiments/adaptive_selection/openai_manifest_probe.py`
- `tests/adaptive_selection/test_openai_manifest_probe.py`

Do not add a public CLI entry point or package `__all__` export.

Implementation requirements:

- Reusable callback receives an injected client; reusable library code never loads `.env`.
- Explicit command loads credential presence only and defaults to dry-run.
- Paid mode requires `--execute-development-probe`.
- The OpenAI client uses `max_retries=0` and timeout 30 seconds.
- Pass exactly the manifest `request_body`; absent controls remain absent.
- Use `raw = client.responses.with_raw_response.create(**request_body)`, `raw_bytes = raw.content`, then `response = raw.parse()`.
- Parse `raw_bytes` independently with `json.loads`; require exact nonnegative JSON integers for top-level and available detailed usage; cross-check SDK values.
- Require completed status, no incomplete reason, no refusals, strict schema match, observed model match, nonempty `raw.request_id`, and nonempty `response.id`.
- Preserve raw body bytes unchanged through `RawTransportResult`.
- Convert SDK/HTTP/parser errors into fixed-category sanitized exceptions with `raise ... from None` before they reach `RecordedCallbackProvider`.
- The command catches provider failures without traceback, exception text, headers, request/response body, or client representation.
- Success and failure artifacts use `0700` directories, `0600` files, atomic rename, `fsync`, no overwrite, canonical JSON, and whole-artifact plus embedded hash verification.
- Default stdout contains only a sanitized summary; response text and raw bytes are excluded.

Required fake/offline tests:

1. Literal request-body equality and omitted-control absence.
2. Zero network calls in dry-run.
3. Exactly one HTTP attempt under success, 429, 500, and connection failure.
4. Exact `LegacyAPIResponse.content` byte preservation.
5. Distinct mandatory provider request ID and response ID.
6. Revision and temperature absence remain JSON `null`/unsupported.
7. Raw JSON usage rejects strings, booleans, negatives, missing fields, and SDK coercion mismatches.
8. Refusal, incomplete, failed status, malformed schema, model mismatch, missing IDs, and missing usage fail deterministically.
9. Sanitized failure artifact records attempted status, server-acceptance state, timestamps, latency, category, HTTP status when available, and request ID when available.
10. Traceback formatting and stdout/stderr cannot reveal credential-bearing fake failures.
11. Atomic permissions, no-overwrite behavior, deterministic artifact bytes, and all hash verification.
12. Dirty/protected-path and manifest-hash gates.
13. Projected-cost guard.
14. Python 3.9 grammar and import behavior.

### Task 4: Verify and review before the live attempt

Required checks:

```bash
PY=/Users/andrewhamilton/Projects/hamiltonhaus/ai-context-manager/.venv/bin/python
$PY -m pytest -q \
  tests/adaptive_selection/test_openai_manifest_probe.py \
  tests/adaptive_selection/test_runner.py \
  tests/adaptive_selection/test_schema.py \
  tests/adaptive_selection/test_packaging.py
$PY -m pytest -q
/usr/bin/python3 -m py_compile \
  experiments/adaptive_selection/openai_manifest_probe.py
$PY -m compileall -q experiments/adaptive_selection tests/adaptive_selection
$PY -m black --check experiments/adaptive_selection tests/adaptive_selection
$PY -m isort --check-only experiments/adaptive_selection tests/adaptive_selection
$PY -m build
git diff --check
```

Independent review gates:

- Scientific review: one-attempt authority, narrow claim, prospective order, no calibration authorization, no fabricated controls or revision.
- Code/security review: retries, HTTP attempt count, raw bytes, raw usage types, IDs, failure artifacts, secret/traceback suppression, Git/hash gate, Python 3.9.

Implementation commit must follow the contract commit in Git history.

### Task 5: Execute, record, and stop

Prerequisites:

- contract and implementation commits exist in the correct order;
- protected paths are tracked, clean, and equal to `HEAD`;
- pinned manifest hash matches;
- all tests/build/reviews pass;
- ignored `.env` contains a nonempty key and remains `0600`;
- ignored output directory is empty;
- dry-run succeeds with zero network calls;
- projected cost passes.

The first network attempt consumes authority. Record exactly one of:

- **Pass:** all frozen identity, structured-output, raw-body, ID, usage, latency, cost, secret, and artifact checks pass.
- **Stop/block:** anything else, including transport ambiguity or parameter rejection.

After preserving and verifying evidence, update the amendment record with the factual result, push one branch, open one PR, report CI, and stop.

## Non-goals

- Any context-sensitivity calibration call.
- Any second or replacement Terra request under this contract or lineage.
- DevOps task or corpus execution.
- Selector, adaptation, held-out, or comparative execution.
- The 99-call bridge or 64-case corpus.
- Product integration or selector redesign.
- Premium-model rescue.
- Efficacy, transfer, safety, production, or favorable-economics claims.
- SQLite regeneration claims.
