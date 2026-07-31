# Terra One-Call Development Probe Protocol

- **Status:** Prospective freeze effective only at the contract commit
- **Protocol version:** `terra-bridge-development-v1`
- **Probe contract:** `terra-config-probe-v1`
- **Date drafted:** 2026-07-30
- **Parent protocol:** [Adaptive Context Pilot Experiment Charter 1.1-draft](adaptive-context-charter.md)
- **Execution boundary:** one benign configuration request attempt; no calibration, confirmatory, adaptation, held-out, or comparative execution
- **Manifest SHA-256:** `bd9d481ad9287da993b09d6412f2df7de9131665958d31597db6eb810a6abedc`

## Purpose and claim boundary

This protocol asks only whether one request to the selected OpenAI `gpt-5.6-terra` endpoint can be represented honestly by the existing provider evidence boundary using the prospectively frozen configuration. Passing does not establish hosted determinism, context sensitivity, adaptive efficacy, transfer, safety, production value, or favorable economics.

This is a Task 12 preparation subtask, not corpus authoring or hosted-model efficacy execution. The parent charter remains draft and continues to prohibit confirmatory execution. The proposed context-sensitivity calibration and 99-call bridge each require a later, separate, explicitly owner-approved execution contract.

## Prospective evidence declaration

At freeze:

- no Terra request has been made for this project;
- no hosted-model adaptation, held-out, calibration, or comparative output exists;
- no bridge corpus has been authored or sealed;
- the probe prompt contains no DevOps task, selector condition, feedback, label, required-context set, answer key, or held-out material;
- the OpenAI credential exists only in the clean worktree's ignored `.env`, mode `0600`, and is excluded from source, artifacts, logs, tests, commits, and reports.

## Sole network-request authority

The first attempted network request consumes the sole authorization regardless of success, timeout, connection ambiguity, HTTP rejection, refusal, incomplete response, parsing failure, missing evidence, or whether a response is received. OpenAI SDK retries are disabled with `max_retries=0`.

There is no automatic retry, parameter fallback, replacement probe, or pre-authorized amended second request anywhere in this contract lineage. Failure produces `stop/block`. Any future request requires a wholly new protocol and explicit owner approval outside this contract.

A pass permits only drafting a non-operative future calibration proposal. It does not authorize a calibration call.

## Frozen request projection

`experiments/adaptive_selection/controls/terra_probe_v1.json` contains one literal `request_body`. The implementation passes exactly that mapping to `responses.with_raw_response.create`. Equality means equality of canonical JSON projections; this protocol does not claim to freeze internal SDK-emitted HTTP JSON bytes.

Only these API-body fields are present:

- model `gpt-5.6-terra`;
- the exact benign input and instructions;
- maximum output tokens `512`;
- fixed non-sensitive metadata;
- parallel tool calls `false`;
- reasoning effort `medium`;
- service tier `default`;
- storage `false`;
- streaming `false`;
- strict JSON Schema under `text.format`;
- empty tools;
- truncation `disabled`.

Temperature, top-p, seed, conversation, previous response, and cache fields are absent. The installed Responses method has no seed parameter. Generic SDK signatures do not establish model-specific temperature or top-p support.

Client settings are separate from the API body: OpenAI Python SDK `2.46.0`, timeout 30 seconds, and `max_retries=0`.

## Identity and evidence-schema migration

The adapter records `response.model` as the observed model identifier. The installed Responses schema exposes no distinct revision or system fingerprint. Revision therefore remains `None` and serializes as JSON `null`. No alias, placeholder, SDK version, date, or response ID may be substituted as a revision or snapshot.

Before the live attempt:

- the existing provider-configuration, raw-transport, and execution revision fields must become nullable, and a new nullable `provider_revision` field must be added explicitly to `RunManifest`;
- temperature must become nullable with an explicit `temperature_supported` boolean;
- the probe uses `provider_revision=None`, `temperature=None`, `temperature_supported=false`, `seed=None`, and `seed_supported=false`;
- serialized shapes, schema version, and provider-configuration hash domain must be versioned; `build_run_manifest`, request/execution validation, serialization/round-trip tests, and relevant comparison/binding behavior must cover the new run-manifest field;
- strict requested/observed provider and model matching remains mandatory.

A model mismatch fails the probe.

## Raw body, IDs, status, and token accounting

The implementation must use:

```python
raw = client.responses.with_raw_response.create(**request_body)
raw_bytes = raw.content
response = raw.parse()
```

`raw.content` is defined here as the exact decoded HTTP entity-body bytes exposed by `LegacyAPIResponse`, without SDK reserialization. It is not guaranteed to be TCP/wire bytes because the HTTP client may decode content encoding.

Both identifiers are mandatory and distinct:

- provider request ID: `raw.request_id` / `x-request-id`;
- response ID: `response.id`.

The response must be completed, have no incomplete reason or refusal, and conform to the frozen strict schema.

The implementation parses `raw.content` independently with `json.loads` and requires exact nonnegative JSON integers for input, output, and total tokens and for cached/reasoning details when present. It cross-checks those values against the SDK projection before passing top-level counts into `RawTransportResult`. No local estimate, zero substitution, or SDK-coerced string/boolean is accepted.

## Failure evidence and secret handling

A sanitized failure artifact is mandatory whenever technically possible before nonzero exit. It contains contract/configuration/code hashes, request-attempted status, server acceptance (`no`, `yes`, or `unknown`), timestamps, latency, a frozen categorical failure reason, HTTP status when available, and provider request ID when available.

It must not contain API keys, authorization or complete headers, environment dumps, client representations, exception messages, tracebacks, request bodies, or response bodies. Transport, SDK, HTTP, and parser exceptions are converted to fixed-category internal exceptions with `raise ... from None`; the command never prints exception chains or tracebacks.

Success and failure artifacts use `0700` directories, `0600` files, atomic writes with `fsync`, no overwrite, canonical JSON, and whole-artifact plus embedded hash verification. Default output excludes response text and raw bytes.

## Preflight and budget

The manifest freezes exact protected paths. Before execution, every protected path must be tracked in `HEAD`, equal to the index and `HEAD`, and free of staged, unstaged, and untracked changes. The manifest hash must equal the value pinned above; the contract commit must precede the implementation commit; `HEAD` and implementation-file hashes are recorded. Ignored `.env` and ignored local artifacts are outside the clean-tree gate.

The pre-dispatch projected input is the canonical request-body UTF-8 byte length plus 1,024 provider-overhead tokens and is rejected above 4,096. At 4,096 projected input tokens and the 512-token output cap, the frozen Terra rates produce a worst-case projection of `$0.014336`, below the `$0.25` guard. This is a conservative projected-cost guard, not a guaranteed actual billing cap.

## Retention limitation

The request sets `store=false`. This is not a claim of zero provider retention. OpenAI's data-controls documentation, checked 2026-07-30, states that API data is not used for training unless the customer opts in, while default abuse-monitoring logs may retain customer content for up to 30 days subject to stated exceptions; approved retention controls can differ. Source: <https://developers.openai.com/api/docs/guides/your-data>.

The prompt is deliberately benign and contains no private environment fact, credential, endpoint, customer data, production identifier, or held-out material.

## Decision

The probe passes only if identity, nullable revision, literal request projection, completed strict structured output, refusal/incomplete checks, exact raw-body bytes, both IDs, raw-validated usage, timestamps, latency, cost, protected-path integrity, artifact hashes, and secret scanning all pass.

Possible outcomes:

- **Pass:** record the result and permit drafting—but not executing—a separate calibration proposal.
- **Stop/block:** anything else. The sole request authority is consumed if network dispatch was attempted.

## Explicitly prohibited

- any second or replacement Terra request under this contract or lineage;
- any context-sensitivity calibration call;
- DevOps task or corpus execution;
- static/adaptive, adaptation, held-out, or comparative execution;
- the proposed 99-call bridge or 64-case corpus;
- selector redesign, premium-model rescue, or product integration;
- efficacy, transfer, safety, production, or favorable-economics claims;
- SQLite regeneration claims.

## Sources captured prospectively

- Terra model catalog: <https://developers.openai.com/api/docs/models/gpt-5.6-terra>
- OpenAI pricing: <https://developers.openai.com/api/docs/pricing>
- Structured outputs: <https://developers.openai.com/api/docs/guides/structured-outputs>
- Reasoning models: <https://developers.openai.com/api/docs/guides/reasoning>
- Data controls: <https://developers.openai.com/api/docs/guides/your-data>

Provider documentation may change. Git records the sources and contract used; the sole live result records observed behavior on its execution date.
