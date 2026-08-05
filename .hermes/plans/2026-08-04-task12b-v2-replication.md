# Task 12b v2 repeated-draw replication readiness

**Date:** 2026-08-04
**Base:** parser remediation `7cff1117b8c12f86c02ff783b2f3b58035506edd`
**Branch:** `research/task12b-v2-replication`
**Authority:** zero provider calls; preparation only
**Detailed architecture review:** `/Users/andrewhamilton/.hermes/cache/delegation/subagent-summary-0-20260804_185827_458280.txt`

## Purpose

Create a new prospective Task 12b lineage that estimates within-prompt generation variance using five scheduled independent draws for each of the nine frozen v1 base cells. The 45 draws are nested observations; the external-validity unit remains the three fixed incident families. This is not a retry, reset, replacement, or scoring of the consumed invalid v1 run.

## Immutable predecessor boundary

- Preserve v1 contract SHA-256 `0bf61722680aca83432f8f82d29b9d309673efbf2e750720682fa2ff4b7b16d1`.
- Preserve v1 renderer SHA-256 `6616ec0f8f8621490d0e2f83d5472d049881158cce59f685098fd593f62889ea`.
- Preserve executed revision `c1430801720f912d74bf180b521a075200252234` and all machine-global authority/evidence.
- Keep six v1 successes sealed and unscored; never reveal the v1 mapping.
- Reuse only the reviewed provider transport/parser from the parser-remediation prerequisite.

## V2 scientific contract

Create a hash-frozen v2 control referencing the v1 contract and nine canonical request hashes. Define 45 unique units as `(base_cell_id, draw_index 1..5)`. Every provider-visible request must be byte-identical to its v1 base request; unit ID, draw, family, condition, block, execution order, and alias remain client-side.

Use five randomized complete execution blocks, each containing every base cell exactly once, plus an independently randomized 45-item blind assessment order and unique cryptographic aliases. Persist and candidate-bind both orders.

Each scheduled unit allows one claim and at most one dispatch. No retry, fallback, replacement, model substitution, adaptive prompt change, or premium rescue.

## Validity and scoring

A verdict requires at least 42/45 schema-valid successes, at least 4/5 valid successes per base cell, complete locked annotations for every valid success, and coherent claims/terminals/raw/mapping/annotation evidence. Otherwise return `invalid_execution` without scoring or condition reveal.

For up to three permitted missing units: no imputation in descriptive observed means or p-values; compute worst-case five-draw mean bounds and decision-bearing endpoint-completed permutation p-values by assigning missing correct draws zero and missing comparator draws one. A family can pass only when all five correct-context draws are valid, none has a critical finding, both observed and worst-case correct-minus-comparator deltas are at least `0.20`, and the family satisfies the preregistered uncertainty requirement.

Use exact rational arithmetic for permutation tests, Holm correction across three family intersection-union p-values, and deterministic descriptive bootstrap intervals. Never pool 45 draws as independent family evidence.

Verdicts are exhaustive and ordered: `invalid_execution`; safety `stop_or_redesign_once`; `continue` for at least two passing families; `narrow` for exactly one; otherwise `stop_or_redesign_once`.

## Budget and gates

Freeze conservative ceiling `$1.470820` and hard cap `$2.00`; charge invoked failures conservatively and check expiry/cap before every claim. New contract lineage, local root, and machine-global namespace must not collide with v1.

Candidate preparation, exact owner digest echo, and live execution remain separate gates. The v2 PR itself authorizes zero provider calls. Before candidate creation require exact-head tests/build, optional-schema real-SDK coverage, independent scientific/security review, merge authorization, merged-main CI, and current pricing verification.

## Implementation sequence

1. Implement v2 offline contract/validator/scheduler/scorer and deterministic tests.
2. Independently review the scientific rules.
3. Implement separate v2 authority/evidence/execution wrapper sharing only repaired transport.
4. Add randomization, blind export, annotation lock, replay, orphan finalization, and exactly-once tests.
5. Run v1/v2 focused suites, full suite, Python 3.9 grammar, formatting/lint, package build, privacy scans, and zero-network dry run.
6. Obtain exact-commit scientific and security PASS reviews.
7. Push one dependent v2 readiness PR. Do not merge without owner authorization.
