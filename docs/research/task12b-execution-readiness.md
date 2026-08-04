# Task 12b Execution-Readiness Protocol

## Status

This document describes a **review candidate with zero current execution authority**. The readiness manifest (SHA-256 `09e08e249b914b096271f64fb39a212a70539e231b2ea7198805c4978ecaa4b4`) sets `network_requests_authorized_by_manifest` to `0`. Merging preparation or readiness code does not authorize a provider request. The implemented live path remains blocked until it is merged from an independently reviewed commit and the repository owner approves one exact private authorization candidate by echoing its SHA-256 out of band.

No CI, test, dry-run, candidate-preparation, export, or recovery command may send a provider request.

## Separation of gates

Task 12b has six distinct gates:

1. **Preparation merge:** freezes the scientific contract, nine rendered requests, rubric, scoring rule, and offline renderer.
2. **Readiness merge:** freezes the execution manifest, authority state machine, transport code, evidence validation, and blind export. It still grants zero authority.
3. **Exact owner authorization:** after readiness is merged, an offline preparation step binds one candidate to the exact clean `main` revision, manifest, request hashes, model/settings, current price ceiling, host/account, credential fingerprint, random nonce, expiry, and private blind mapping. The owner must echo that candidate's exact SHA-256 out of band. A generic “go ahead” is insufficient.
4. **Execution:** exactly one eligible attempt per cell in frozen order, with authority consumed before dispatch and no retry, fallback, replacement, model substitution, tool use, or adaptive change.
5. **Blind assessment:** assessors receive precommitted random assessment IDs and order, not canonical cell IDs, conditions, families, provenance, execution position, or provider metadata.
6. **Deterministic scoring:** only nine valid responses may enter the frozen scorer. A failed or ambiguous cell is preserved but cannot be imputed, replaced, or scored into a preregistered verdict.

Each gate is separate. Approval of one does not imply approval of a later gate.

## Frozen identity and budget

The readiness candidate must reconstruct the nine requests through the unchanged renderer and reject any byte drift. It is bound to:

- contract SHA-256 `0bf61722680aca83432f8f82d29b9d309673efbf2e750720682fa2ff4b7b16d1`;
- fixed nine-cell order and request hashes in the readiness manifest;
- `gpt-5.6-terra`;
- OpenAI SDK `2.46.0` and HTTPX `0.28.1`;
- reasoning effort `medium`;
- `2,048` maximum output tokens per cell;
- `store=false`, `stream=false`, `tools=[]`, `max_retries=0`, and a 30-second timeout;
- 29,192 conservative input tokens and 18,432 maximum output tokens across nine cells; and
- an owner cap of USD `1.00`.

Official pricing was reverified on 2026-08-04 at <https://developers.openai.com/api/docs/pricing>. The current conservative short-context ceiling prices all input at the GPT-5.6 automatic cache-write rate of USD `2.50` per million tokens and output at USD `12.00` per million tokens, producing USD `0.294164`. This is the authorization ceiling. The earlier USD `0.279568` ordinary-input calculation is reproducibility context only and is not a current maximum. Actual usage records preserve ordinary input, cached input, cache-write tokens when reported, and output. Missing optional cache-write details must not be assumed to mean zero; conservative cost remains available when exact cost cannot be derived.

The USD `1.00` check is an enforced projection and accepted-usage cap, not a guarantee against provider billing anomalies or unrelated account activity.

## Non-authorizing candidate and owner approval

Offline candidate preparation creates two owner-private records:

- a nonce-bearing, expiring authorization candidate; and
- a randomized mapping from canonical cells to unrelated assessment IDs and a blind order independent of execution order.

The mapping's canonical hash is included in the candidate. Candidate preparation may inspect the exact Git revision and credential fingerprint, but it performs no provider request. The candidate explicitly grants zero network authority.

Execution additionally requires a separate owner-approval record containing the exact candidate SHA-256 that the owner echoed out of band. The code cannot generate, infer, or substitute that echo. The future approval request must show at least:

- exact execution commit;
- contract and readiness-manifest hashes;
- all nine ordered request hashes;
- exact provider/model/settings and dependency versions;
- pricing source/date and USD `0.294164` ceiling;
- USD `1.00` cap;
- private evidence paths;
- fixed order and nine-call scope;
- one-attempt/no-retry semantics; and
- authorization nonce, expiry, and candidate SHA-256.

The approval digest is operational process evidence. Unkeyed hashes and `0600` modes do not provide cryptographic proof against another process running as the same OS user.

## Repository and runtime preflight

Live execution is permitted only from the ordinary `main` checkout when:

- the tracked tree is clean;
- `HEAD` equals the already-fetched `origin/main` and the authorized revision;
- the protected contract, renderer, readiness manifest, runner, tests, and documentation match the authorized commit;
- the nine rendered request hashes and order match both manifests and authorization;
- dependency versions and all provider settings are exact;
- the private authorization, approval, and blind-mapping records are valid, owner-only regular files with mode `0600` and one link;
- the host, account, and credential fingerprints match; and
- the authorization window is current.

The process obtains a fresh UTC observation immediately before every authority claim and rejects the cell without claim or dispatch if that observation is after expiry. Authorization validity is therefore not frozen at process start. A second fresh observation timestamps the terminal after the one allowed attempt.

The paid process does not fetch, build, test, install, or resolve packages. Those gates run beforehand against the exact commit. From before SDK import and TLS/client construction until after raw handling, terminal persistence, and client close, the process removes provider routing, organization, project, custom-header, proxy, certificate, TLS-keylog, and SDK-log ambient variables and disables process logging. The client uses only the official OpenAI HTTPS endpoint, standard TLS verification, `trust_env=false`, redirects disabled, and `max_retries=0`.

## Cooperative authority state

The machine-global state root is account-local under:

`~/.local/state/ai-context-manager/task12b-authority/<contract-lineage-sha256>/`

The repository-local audit root is ignored:

`.local/adaptive-selection-task12b/`

A nonblocking machine-global lock is held across reconciliation and execution. Machine-global claims and terminal records are authoritative. Repository-local copies are non-authoritative audit mirrors.

Immediately before each dispatch, the runner publishes the machine-global authority-consumption claim and then its repository-local mirror. A claim states that authority is consumed; it does **not** claim that dispatch occurred. Terminal evidence separately records:

- `authority_consumed`;
- whether dispatch was invoked (`true`, `false`, or `unknown`);
- whether server acceptance was `yes`, `no`, or `unknown`;
- a fixed sanitized failure category when applicable;
- bounded raw-response hash and bytes when an HTTP response exists;
- response/request/model/status metadata;
- usage and exact or conservative cost; and
- exact contract, request, authorization, code, host/account, and credential identity.

Private records use owner-only directories and files, no symlink traversal, one-link checks, descriptor-relative access, fsync, and atomic no-overwrite publication. They are cooperative evidence, not adversarially immutable against the same OS user.

## Crash and resume semantics

State is fail-closed:

- no global claim and no terminal: pending;
- valid global claim plus valid global terminal: terminal and skipped;
- global claim without terminal: blocked orphan claim;
- local-only claim, terminal without claim, raw-only evidence, missing index, conflicting mirrors, or any semantic/hash mismatch: abort.

Ordinary resume never turns a claim-only state into a synthetic terminal, deletes it, or retries it. An orphan requires a separate, owner-approved, zero-network finalization bound to the exact orphan-claim digest after confirming no runner remains. Finalization records `invalid_ambiguous`, `dispatch_invoked=unknown`, and `server_acceptance=unknown`; it never dispatches.

A provider refusal, malformed response, HTTP error, connection ambiguity, invalid schema, or bounded-evidence failure is final for that cell. The runner may continue only to cells that have never consumed authority and only while the original order, authorization window, and cost invariants remain valid. A local evidence-publication failure stops the run.

## Raw evidence and blindness

Raw responses and terminal metadata remain private and Git-ignored with `0700` directories and `0600` files. They must never be printed, committed, or placed in CI artifacts. Fixed public status messages exclude credentials, prompts, provider text, exception text, and private identifiers.

For every recorded success, verification reparses the immutable raw bytes with duplicate-key rejection at both the provider envelope and nested structured-output layers. It replays completion/refusal state, pinned model, response ID, HTTP `200` acceptance metadata, the exact no-action output-item allowlists, sole output text, bounded response schema, usage, and pricing. Reasoning summary entries are recursively restricted to exact `summary_text` objects; unrequested annotations, log probabilities, action fields, tool fields, duplicate keys, and unknown nested fields are rejected. Those independently derived values must equal the terminal projections. Raw and SDK usage-detail presence and values must agree; SDK-only nonzero cache reads or an SDK cache-write value absent from raw evidence are rejected. Resume and blind export also reconstruct the provider-visible task and evidence from the frozen contract and require exact equality with the terminal record. Rehashing altered raw bytes, contradictory HTTP metadata, or altered provider-visible evidence therefore does not make them acceptable.

Blind export first verifies all authoritative claims, terminals, raw hashes, and the precommitted private mapping. It emits, in precommitted blind order:

- randomized assessment ID;
- task;
- provider-visible timestamped evidence;
- structured response or invalid/failure status;
- condition-independent criteria;
- adjudication rules; and
- critical findings.

It adds no canonical cell ID, condition, family/scenario identifiers, condition anchors, provenance/source-role metadata, execution order/position/timestamps/latency, provider request/response IDs, evidence paths, or controller filenames. The substantive task and timestamped incident evidence are preserved exactly as sent to the provider; those frozen strings can themselves mention source revisions or filenames needed to assess the incident and are not controller metadata. The private mapping is retained until annotations are locked.

`assessment_ready` is true only for nine verified valid successes. Any failed or ambiguous cell forces `assessment_ready=false`; no score or preregistered verdict may be derived.

## Operational commands

All commands run from the ordinary repository checkout. The module is offline by default:

```bash
python -m experiments.adaptive_selection.context_sensitivity_execution
```

After readiness is reviewed and merged, candidate preparation is a separate offline action. It reads the ignored credential only to bind its fingerprint, writes the private candidate and blind mapping without overwrite, prints the safe candidate SHA-256, and grants zero network authority:

```bash
python -m experiments.adaptive_selection.context_sensitivity_execution \
  --prepare-authorization-candidate \
  --owner-identity '<exact owner identity>' \
  --expires-at '<UTC RFC3339 expiry>'
```

The CLI deliberately provides no owner-approval writer. Only after a separately created approval record contains the exact candidate SHA-256 echoed out of band may the conspicuous live command become eligible:

```bash
python -m experiments.adaptive_selection.context_sensitivity_execution \
  --execute-authorized-nine-cell-manifest
```

Blind export and orphan finalization are zero-network commands. Export may occur after the execution window expires because it verifies historical authority and immutable terminal evidence. Orphan finalization additionally requires a separately created owner record bound to the exact orphan-claim digest and confirmation that no process remains:

```bash
python -m experiments.adaptive_selection.context_sensitivity_execution \
  --export-blind-assessment

python -m experiments.adaptive_selection.context_sensitivity_execution \
  --finalize-authorized-orphan --orphan-cell '<exact canonical cell>'
```

Candidate, approval, mapping, terminal, raw, export, and orphan-finalization records remain private and ignored. None belongs in Git or CI artifacts.

## Review and publication gates

Before the readiness PR can be called ready:

1. focused execution tests, the frozen 67-test calibration suite, Task 12a probe tests, and the full repository suite pass;
2. Python 3.9 syntax, Black, lint, compile, and package build checks pass;
3. offline dry-run reconstructs the exact contract and request hashes without creating private state;
4. fake-client and real frozen-SDK `httpx.MockTransport` tests prove one HTTP handler invocation for success, refusal, incomplete response, redirect, 429, 500, malformed response, and connection failure;
5. privacy scans show no credential, private evidence, or raw artifact in the complete branch range;
6. independent scientific/specification and security/code-quality reviewers pass the exact candidate commit;
7. CI passes; and
8. the owner separately merges the readiness PR.

Even after merge, live calls remain `0/9` and unauthorized until the exact candidate digest is explicitly approved.
