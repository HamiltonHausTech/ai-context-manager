# ADR 0002: Feedback Is Event Data, Not Truth

- **Status:** Accepted
- **Date:** 2026-07-28

## Context

Feedback can be sparse, ambiguous, contradictory, delayed, or attributable to several selected context items. Treating an acceptance, correction, rubric result, or judge signal as immediate organizational truth would erase provenance and conflate an observed event with a reviewed reusable conclusion. It would also make experiments difficult to reproduce because later mutations could rewrite the evidence from which utility estimates were derived.

The [experiment charter](../research/adaptive-context-charter.md) therefore requires leakage controls, frozen manifests, and auditable feedback handling.

## Decision

Store raw feedback as append-only event data. Every feedback event must be tied to the relevant run, task case and family, selected context where applicable, observed outcome, signal type and value, source, timestamp, and provenance. Corrections create new events; they do not overwrite prior events.

Treat utility estimates, summaries, and candidate experiences as derived artifacts. Promotion from raw feedback into reusable experience is a separate, reviewed lifecycle with explicit evidence, provenance, confidence, policy version, and—where required—human approval. Promotion must not mutate or delete the source events.

## Consequences

### Positive

- Experimental evidence remains reproducible and auditable.
- Contradictory or corrected feedback can coexist without rewriting history.
- Utility estimates can be rebuilt under a new learning policy.
- Governance can distinguish observed behavior from reviewed reusable knowledge.

### Negative and risks

- Append-only storage and provenance increase schema and operational complexity.
- Event interpretation and credit assignment remain necessary; the log alone does not establish causality.
- Promotion review introduces delay and may limit adaptation speed.

### Follow-up constraints

- Derived utility must record its source event IDs and learning-policy version.
- No successful run assigns equal causal credit to every selected context item by default.
- Raw feedback must not be presented as validated organizational policy, fact, or preference.
