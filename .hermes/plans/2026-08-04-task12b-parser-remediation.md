# Task 12b parser remediation and prospective v2 boundary

**Date:** 2026-08-04
**Base revision:** `c1430801720f912d74bf180b521a075200252234`
**Branch:** `research/task12b-v2-readiness`
**Status:** implementation in progress; zero execution authority

## Preserved invalid run

Candidate `sha256:72cf036cc1952a8e8abaf745083cad9bece36c49ba936205d6506078093b37d4` was exactly authorized and consumed all nine v1 authorities. The provider returned nine HTTP 200 responses. Six responses validated; three terminal records were classified `malformed_response` because their completed reasoning items carried the pinned SDK's provider-valid optional `content` field as `[]`. A zero-dispatch resume verified all nine terminals and the blind export remained `assessment_ready=false`.

The six valid responses remain sealed and unscored. The mapping remains unrevealed. No v1 cell may be retried, replaced, regenerated, or rescored. The exact executed evidence stays bound to revision `c143080`.

## Parser-remediation PR

1. Add failing unit and real-SDK `httpx.MockTransport` regressions reproducing a completed reasoning item with `content: []`.
2. Permit reasoning `content` only when exactly an empty list.
3. Reject absent-policy violations including nonempty lists, null, mappings, strings, booleans, and unknown reasoning fields.
4. Preserve duplicate-key rejection, recursive summary validation, message/output allowlists, raw/SDK agreement, no-retry semantics, and the frozen v1 calibration artifacts.
5. Verify focused/full tests, formatting, build, frozen hashes, clean diff, and exact-commit independent scientific and security review.
6. Publish one focused PR. It grants zero hosted-call authority and does not reset v1 machine-global state.

## Prospective v2 direction

A later separate PR will define a new experiment lineage with five scheduled independent draws for each of the nine frozen cells (45 total dispatch units). Scheduled draws are observations, not retries. The provider-visible request for a base cell remains frozen; private unit identity includes base cell plus draw index.

The v2 contract must prospectively freeze:

- a separate machine-global namespace and authorization identity;
- exact 45-unit order/randomization and private blind aliases;
- no retry, fallback, replacement, or adaptive prompt changes per scheduled draw;
- invalid/missing-draw handling and minimum validity thresholds;
- within-cell variation summaries without treating repeated draws as additional incident families;
- family-level aggregation, uncertainty reporting, and exhaustive continue/narrow/stop/invalid rules;
- conservative execution ceiling `$1.470820` and hard cap `$2.00`;
- candidate preparation, exact owner digest echo, and live execution as separate gates;
- comprehensive pinned-SDK optional output-item coverage before authorization.

The old 99-call bridge remains unauthorized. V2 readiness will authorize zero calls until independently reviewed, merged, verified on merged main, privately prepared, and exactly approved by the owner.
