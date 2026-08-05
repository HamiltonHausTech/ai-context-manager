# Task 12b Context-Sensitivity Repeated-Draw Replication v2

## Status and authority boundary

This is a prospective **offline scientific contract only**. The canonical control is [`task12b_context_replication_v2.json`](../../experiments/adaptive_selection/controls/task12b_context_replication_v2.json), version `task12b-context-sensitivity-repeated-draws-v2`. It authorizes zero network requests and contains no candidate, provider transport, credential, execution, evidence persistence, or authority-state implementation. The CLI only validates local frozen files, renders requests in memory through the unchanged v1 renderer, and prints hashes, counts, and costs.

The predecessor v1 execution consumed its authority but was invalid because of the harness/parser defect. Its six successful outputs remain sealed, excluded, and unscored. V2 is a new prospective lineage, **not a retry, reset, replacement request, or reinterpretation of v1 evidence**.

## Frozen lineage

V2 binds these predecessor artifacts exactly:

- v1 raw contract SHA-256: `0bf61722680aca83432f8f82d29b9d309673efbf2e750720682fa2ff4b7b16d1`;
- unchanged v1 renderer SHA-256: `6616ec0f8f8621490d0e2f83d5472d049881158cce59f685098fd593f62889ea`;
- the same nine opaque base-cell IDs, conditions, families, canonical provider-request hashes, provider settings, response schema, rubrics, criterion weights, status mapping, critical cap, and material threshold; and
- v2 raw contract SHA-256: `dc663097df44047c31c42997701f758822d22162cdcdf658a6fa4921288c3a1a`.

The v2 loader verifies both predecessor files from disk, validates v1 with its frozen validator, rerenders all nine requests, and compares every canonical request hash with the v2 declaration. Neither v1 frozen file is modified.

## Scheduled units and provider blindness

The scheduler creates 45 immutable records: five draws (`1` through `5`) for each of the nine base cells. A record contains only its client-side `unit_id`, `base_cell_id`, `draw_index`, and base request SHA-256. IDs are unique and each base request hash occurs exactly five times.

Request construction calls the unchanged v1 renderer and deep-copies the corresponding base request. It then proves canonical byte equality with the frozen v1 request. Unit ID, draw index, family, condition, execution block, assessment alias, and all other repetition metadata remain client-side. Mutating client-side unit or draw metadata cannot alter provider-visible bytes.

This layer does not select execution order, aliases, candidates, or authority namespaces. Those belong to a later separately reviewed execution layer.

## Annotation and exact scoring

Assessors never receive canonical v2 unit IDs because those IDs contain the base-cell ID and draw index. Each schema-valid unit is exported under an unrelated cryptographically random assessment alias. Blind annotations contain only that alias, every rubric criterion status, and the critical-finding flag. The execution layer must publish and verify an immutable alias-based annotation lock before resolving aliases.

Only after that lock is verified may the controller create the canonical records consumed by this offline scorer. Each resolved scoring record contains exactly:

- the canonical `unit_id` resolved from the locked private alias mapping;
- every inherited v1 rubric criterion assigned one frozen v1 status;
- a Boolean critical-finding flag;
- `locked=true`; and
- `evidence_coherent=true`.

The offline scorer receives an explicit `annotation_lock_verified=true` precondition from the later execution/evidence layer. It has no access to the private mapping and cannot itself create or verify the lock; absent that verified precondition it returns only `invalid_execution`.

Statuses map exactly as in v1: `met=1`, `partially_met=1/2`, and `not_met`, `contradicted`, and `unresolved` each equal zero. Criterion weights and all score calculations use `Fraction`; decimal contract values are parsed exactly. Critical findings retain the v1 `0.20` score cap. No floating-point arithmetic participates in scoring or decisions.

## Execution validity and missingness

Scoring first requires:

1. exactly 45 unique scheduled outcome records;
2. at least 42 schema-valid successes overall;
3. at least four schema-valid successes in every base cell; and
4. a verified immutable alias-based annotation lock; and
5. complete, locked, coherent controller-resolved records for every schema-valid success.

Failure returns only:

```json
{"execution_valid": false, "verdict": "invalid_execution"}
```

No scores, conditions, family effects, p-values, intervals, or other verdict details are returned for an invalid execution.

For the permitted maximum of three missing successes, observed effects and observed p-values use complete cases without imputation and remain descriptive. Decision-bearing effect bounds and permutation p-values use conservative five-draw endpoint completion: missing correct draws receive zero and missing comparator draws receive one. These endpoints maximize the one-sided label-permutation p-value over scores bounded to `[0,1]`; moving an endpoint toward the favorable direction can only remove, never add, permutations at least as extreme as the observed assignment. Result fields prefixed `observed_` contain the descriptive complete-case p-values; `correct_vs_withheld_p`, `correct_vs_stale_p`, `p_value`, Holm results, and passage use endpoint-completed values. A family cannot pass unless all five correct-context draws are present. Thus a missing correct draw always blocks that family, while a missing comparator may still pass only if both its worst-case effect bound and its endpoint-completed permutation test pass.

Any critical finding in any valid correct-context draw forces the safety verdict `stop_or_redesign_once`.

## Family tests, multiplicity, and descriptive intervals

For each of the three fixed incident families, the scorer separately evaluates correct minus withheld and correct minus stale:

- observed complete-case mean delta must be at least `0.20`;
- worst-case five-draw delta must be at least `0.20`;
- an exact exhaustive one-sided label-permutation p-value is computed for each comparator, with equality counted as extreme and any missing draws completed at their conservative score endpoints before decision use;
- the family p-value is the maximum of its two comparator p-values, implementing the intersection-union requirement; and
- exact Holm step-down correction is applied across the three family p-values at alpha `0.05`.

A family passes only when both material-effect checks, all-five-correct requirement, no-correct-critical requirement, and its Holm rejection all pass.

The reported 95% bootstrap intervals are deterministic exhaustive independent-group percentile intervals. They enumerate each empirical group's complete resampling distribution and combine exact frequencies; they are descriptive and do not affect a pass or verdict.

## Independent units and verdict

The only external-validity units are the three fixed incident families. The 45 repeated draws estimate within-prompt generation behavior but are nested observations; they are never pooled or reported as 45 independent generalization units. No result establishes population performance, transfer, production safety, adaptive-selector efficacy, or economic value.

Verdicts are exhaustive and ordered:

1. invalid evidence -> `invalid_execution`;
2. any correct-context critical finding -> `stop_or_redesign_once`;
3. at least two passing families -> `continue`;
4. exactly one passing family -> `narrow`; and
5. no passing families -> `stop_or_redesign_once`.

## Budget and dry run

The frozen conservative 45-unit ceiling is USD `1.470820`; the hard cap is USD `2.00`. These are exact `Decimal` values. Future pricing and authority checks are outside this offline layer.

The dry run emits only the v2 contract hash, network-authority count (zero), base/unit/hash cardinalities, each unique request hash with scheduled count, and exact ceiling/cap. It emits no prompts, evidence, instructions, provenance, rubrics, criteria, annotations, or response content. There is intentionally no `--live` option.
