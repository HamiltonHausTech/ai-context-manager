# ADR 0003: Strong Static Selection Is the Primary Baseline

- **Status:** Accepted; protocol-specific lock details pending before freeze
- **Date:** 2026-07-28
- **Updated:** 2026-07-28 for protocol 1.1-draft

## Context

An adaptive selector can appear successful if it is compared only with naive top-K retrieval or an intentionally weak static policy. Full eligible context is also an unsuitable primary fairness baseline: it is useful for measuring token cost and attention dilution, but it may use a different amount of context and does not represent the strongest credible non-learning alternative.

The [experiment charter](../research/adaptive-context-charter.md) defines the sequential quality-non-inferiority and token-efficiency claim. The prospective correction from protocol 1.0 is recorded in the [amendment record](../research/adaptive-context-protocol-amendments.md), and implementation sequencing is in the [plan](../../.hermes/plans/2026-07-28_170432-adaptive-context-proof-plan.md).

## Decision

Use **exactly one locked strong static rules-and-metadata policy** as the primary baseline for the adaptive-context pilot. Before held-out authoring/sealing or outcome access:

1. define the eligible static-policy candidate set;
2. define the development/adaptation-only selection criterion;
3. define a deterministic tie-breaker (for example, prefer the simpler policy, then the lower-context policy, then a stable configuration ID, in that declared order); and
4. select and seal exactly one primary static policy using development/adaptation information only.

The precise tie-breaker must be chosen and recorded in the frozen protocol; it may not be improvised after seeing held-out results. Other static candidates, similarity/top-K, and the existing explainable selector remain secondary/exploratory comparisons. Full eligible context remains a cost-and-attention reference, not a primary fairness baseline.

The primary static and adaptive policies receive:

- the same eligible candidate pool and representations;
- the same context-token budget, prompt, model/provider, generation configuration, and tools;
- the same metadata and development information, except that adaptive mode receives only the protocol-permitted prior adaptation feedback; and
- comparable engineering effort, hyperparameter-search budget, reviewer access, and tuning opportunity before lock.

The adaptive candidate may differ only through transparent utility learned from permitted prior adaptation feedback. Baseline selection, tuning records, candidate results on development/adaptation material, and the applied tie-breaker must be retained in the audit trail.

## Consequences

### Positive

- The sole confirmatory contrast is unambiguous and cannot switch to whichever baseline is convenient after outcome access.
- Any adaptive advantage must exceed a prospectively selected practical non-learning alternative.
- Equal inputs, tuning opportunity, and budgets reduce confounding.
- Full context and alternative baselines remain useful diagnostics without inflating confirmatory multiplicity.

### Negative and risks

- Building and documenting a strong static candidate set requires additional design and review effort.
- The locked static policy may match or outperform adaptation, producing a valid negative result.
- Development selection can still overfit; provenance-group splits and frozen selection rules mitigate but do not eliminate that risk.
- “Comparable tuning effort” requires an auditable resource definition before work begins.

### Follow-up constraints

- Freeze the candidate set, selection criterion, exact tie-breaker, tuning budget, and selected primary policy before confirmatory execution.
- Report all executed baselines and negative results, but label every non-primary contrast secondary/exploratory.
- Do not give adaptive selection richer metadata, a larger budget, or a different generation configuration.
- Static policy or selection-rule changes after held-out outcome access require a new protocol and newly sealed evaluation.
