# Task 12b Context-Sensitivity Calibration Review Candidate

## Status and scope

Task 12b is an **offline review candidate with no execution authority**. Its canonical contract is [`task12b_context_calibration_v1.json`](../../experiments/adaptive_selection/controls/task12b_context_calibration_v1.json). The validator and renderer can inspect that contract, construct canonical request bodies in memory, and emit hashes and conservative budget projections. They cannot contact OpenAI or any other provider, and the CLI has no live mode.

A later, separately approved calibration is only a descriptive investment screen: it could show whether each total frozen current-evidence packet changes one Terra response enough to improve actionable incident-resolution utility in three real-incident replays. The estimand includes packet evidence quantity and detail; it does not isolate recency or decisive content from packet size. Because there is one unseeded response per cell, a null or adverse result cannot distinguish generation variance from context insensitivity and cannot statistically falsify context sensitivity. The gate governs whether these selector-style packets justify investing in the larger bridge; it does not establish adaptive efficacy, transfer, population performance, production safety, or economic value. It is not part of the parent protocol's confirmatory experiment.

## Provenance and sanitization boundary

All scenario evidence is derived from primary artifacts rather than polished wiki summaries, assistant conclusions, former consulting work, or day-job material. The contract records source kind, source reference, capture time, and sanitization for every scenario, and binds each sanitized evidence excerpt to an evidence-level source locator:

- owner-authorized home-host diagnostic sessions and an earlier incident snapshot;
- tracked source, public Git commit metadata, and a clean GitHub Actions log for the provider-packaging incident; and
- local Git refs, object data, reflogs, and preserved hashes for the repository-divergence incident.

Provider-visible text excludes provenance records, evidence IDs, source-role labels, condition labels, rubrics, decision rules, and execution order. It retains each observation's sanitized capture timestamp so current and historical snapshots remain distinguishable without exposing the internal source-role label. Private hostname, address, integration, MAC, household, credential, and path identifiers were removed or replaced. The only retained synthetic network identifiers are the documentation address `192.0.2.31` and `edge-controller-a`. Validation rejects the contract if any listed private identifier pattern appears in rendered text, case-insensitively.

The renderer also rejects every scenario's predeclared conclusion phrases. This prevents the prompt from embedding the adjudicator's answer rather than the underlying observations. Request bodies contain only the fixed instructions, task, numbered timestamped evidence content, frozen model controls, and strict response schema. Cell IDs remain client-side and are absent from provider requests.

## Exact nine-cell design

There are exactly three scenario families, each with exactly one cell under each condition:

| Family | Correct current context | Withheld context | Stale context |
| --- | --- | --- | --- |
| Personal host/network operations | `cell-k4m2` | `cell-p7q9` | `cell-v3x8` |
| Software dependency/evidence boundary | `cell-b8n5` | `cell-r2d6` | `cell-y9h4` |
| Git repository-state diagnosis | `cell-c6w1` | `cell-m8t3` | `cell-z5f7` |

The fixed execution order is:

1. `cell-k4m2`
2. `cell-y9h4`
3. `cell-m8t3`
4. `cell-v3x8`
5. `cell-b8n5`
6. `cell-z5f7`
7. `cell-p7q9`
8. `cell-r2d6`
9. `cell-c6w1`

Cell IDs are opaque. Correct-context cells require decisive evidence. Withheld and stale cells prohibit decisive evidence, and stale cells require historical evidence. Evidence IDs are globally unique but never provider-visible.

## Blind rubric and scoring plan

If a later execution is authorized, response order is frozen before execution and condition labels remain hidden from assessors. Human annotators see the task, timestamped evidence, response, condition-independent rubric criteria, adjudication rules, and critical findings, but not condition, condition-named anchors, source-role labels, source provenance, or execution position. They assign the predeclared criterion statuses (`met`, `partially_met`, `not_met`, `contradicted`, or `unresolved`). Deterministic scoring maps those statuses to `1.0`, `0.5`, `0.0`, `0.0`, and `0.0`, respectively, then applies each family's criterion weights, which must sum exactly to `1.00`. Condition-named anchors remain frozen review-only audit guidance after blind statuses are submitted; they cannot alter those statuses.

The primary outcome is condition-independent actionable incident-resolution utility against the frozen assessor-side target, not whether a response is merely appropriate for the evidence it received. Calibrated uncertainty in withheld or stale cells may satisfy uncertainty and safety criteria, but it cannot satisfy diagnosis, classification, preservation, alignment, or remediation criteria that remain unresolved. Conversely, generic caution in a correct-context cell does not earn resolution credit when the supplied decisive evidence supports a narrower diagnosis or action. Condition-specific anchors freeze these distinctions before outcome access.

A critical finding caps the normalized score at `0.20`. The material-advantage threshold is a normalized score delta of `0.20`. Rubric text and critical findings remain assessor-side and are never included in provider requests. The offline `score_annotations()` path validates one opaque annotation record per cell, applies the frozen status mapping and criterion weights, enforces the critical cap, calculates both family deltas, and derives `continue`, `narrow`, or `stop_or_redesign_once` without post-hoc arithmetic. Ambiguous or incomplete scoring may not be repaired with post-hoc interpretation.

## Decision gate

The prospective decision rule is intentionally narrow:

- **Continue:** correct context exceeds both withheld and stale context by at least `0.20` in at least two families, with no critical finding in any correct-context response.
- **Narrow:** exactly one family meets that rule; any later bridge is limited to that family.
- **Stop or redesign once:** no family passes, only cosmetic or confidence differences appear, stale context causes no measurable error, any correct-context response has a critical finding, or scoring needs post-hoc interpretation.
- **Invalid execution:** preserve a failed or ambiguous cell without retry or replacement under the same contract.

No result may be generalized beyond the three incident replays.

## Review, owner, and execution gates

The contract authorizes **zero network requests**. Any future execution requires all of the following before a provider call:

1. exact contract freeze;
2. independent scientific and security review;
3. explicit owner approval;
4. re-verification of pricing; and
5. a wholly separate execution mechanism and authority record.

The future policy is one attempt per cell with no SDK retry, fallback, replacement, or premium-model rescue. The offline module intentionally imports neither `openai` nor `httpx`, contains no network or execution function, and offers no live CLI option.

## Offline renderer and budget guard

For each cell, the renderer deep-copies the frozen request template and inserts the fixed instructions plus `Task:` and numbered timestamped evidence content. Canonical JSON uses sorted keys, compact separators, UTF-8, and non-ASCII preservation. The response schema cites supporting evidence by number, limits free text to printable ASCII, and caps the maximum schema-valid free text below the 2,048-token output allowance.

The conservative projection treats every canonical request UTF-8 byte as one input token, adds 1,024 input tokens of overhead per cell, and reserves 2,048 output tokens per cell. It applies the provisional frozen input/output rates with decimal arithmetic and rejects totals above USD `1.00`. The default CLI emits only:

- the raw contract SHA-256;
- per-cell request SHA-256, UTF-8 byte count, projected input/output tokens, and projected maximum cost; and
- aggregate projected maximum cost and cap.

It emits no prompt, instructions, task, evidence, provenance, or rubric content.

## Source snapshots

The review candidate binds its claims to snapshots named in the contract rather than to mutable external systems:

- home-host sessions captured on 2026-07-25 and 2026-08-03;
- provider-probe repository history (`e520080`, `4a11315`, and `d92f15b`) and GitHub Actions run `30594892534`; and
- the local repository snapshot at `origin/main=2dd1482`, including the preserved local commit, reflog observations, and byte-verified plan copy.

These references support review and audit only. The offline validator reads no source snapshot and performs no GitHub, provider, HTTP, or other network lookup.
