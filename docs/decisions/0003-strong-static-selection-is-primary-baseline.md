# ADR 0003: Strong Static Selection Is the Primary Baseline

- **Status:** Accepted
- **Date:** 2026-07-28

## Context

An adaptive selector can appear successful if it is compared only with naive top-K retrieval or an intentionally weak static policy. Full eligible context is also an unsuitable primary fairness baseline: it is useful for measuring token cost and attention dilution, but it may use a different amount of context and does not represent the strongest credible non-learning alternative.

The [experiment charter](../research/adaptive-context-charter.md) requires comparable selector modes and makes downstream task quality—not retrieval quality—the primary outcome. The rationale and staged implementation are recorded in the [approved plan](../../.hermes/plans/2026-07-28_170432-adaptive-context-proof-plan.md).

## Decision

Use the strongest credible static rules-and-metadata selector as the primary baseline for the adaptive-context pilot. Develop and select that baseline using adaptation/validation information only, then freeze it before confirmatory held-out outcomes are viewed.

The primary baseline receives the same eligible candidate pool, context-token budget, prompt, model, generation configuration, and metadata available to the adaptive candidate. The adaptive candidate may differ only through utility learned from permitted prior adaptation feedback.

Retain similarity/top-K and the existing static explainable selector as informative secondary baselines. Retain full eligible context as a cost-and-attention reference, not as a sandbagged primary baseline or a budget-matched claim when its budget differs.

## Consequences

### Positive

- Any adaptive advantage must exceed a practical non-learning alternative.
- Equal inputs and budgets reduce confounding.
- Full context remains useful for diagnosing token cost and attention dilution without distorting the primary claim.

### Negative and risks

- Building a strong static baseline requires additional design and review effort.
- The best static policy may match or outperform adaptation, producing a negative result.
- Selecting among static policies can itself overfit unless selection is confined to adaptation/validation data and frozen before held-out evaluation.

### Follow-up constraints

- Report all baselines and negative results, not only comparisons favorable to adaptation.
- Do not give the adaptive selector richer metadata, a larger budget, or a different generation configuration.
- Static feature and policy changes after held-out outcome access require a new protocol and newly sealed evaluation.
