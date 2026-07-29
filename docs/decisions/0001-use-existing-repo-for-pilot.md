# ADR 0001: Use the Existing Repository for the Pilot

- **Status:** Accepted
- **Date:** 2026-07-28

## Context

The adaptive-context hypothesis requires controlled comparisons, event provenance, selection traces, token budgeting, storage, and evaluation. The existing `ai-context-manager` repository already contains relevant retrieval, feedback, memory-lifecycle, provenance, hybrid-scoring, storage, and evaluation capabilities. Creating a new repository before the hypothesis has repeatable support would duplicate infrastructure and prematurely establish a product boundary.

The pilot also needs isolation so experimental assumptions do not become claims about the core package. The scientific protocol is defined in the [adaptive context charter](../research/adaptive-context-charter.md) and the broader work is sequenced in the [approved implementation plan](../../.hermes/plans/2026-07-28_170432-adaptive-context-proof-plan.md).

## Decision

Use `ai-context-manager` as the pilot substrate. Keep the research harness isolated under `experiments/adaptive_selection/`, reuse existing package seams where they are adequate, and avoid changing core retrieval behavior merely to accommodate an unproven experiment.

Do not extract a new repository or promote the harness into the core package until repeatable evidence supports doing so. After such evidence, explicitly decide among promoting reusable modules, extracting a dedicated runtime repository, or retaining a research-only harness.

## Consequences

### Positive

- The pilot can reuse existing infrastructure and focus effort on experimental validity.
- Isolation makes experimental code and claims distinguishable from supported package behavior.
- Delaying extraction preserves architectural options until requirements are evidence-based.

### Negative and risks

- The harness inherits constraints and technical debt from the existing repository.
- Care is required to prevent experimental dependencies and assumptions from leaking into the core package.
- A later extraction may require migration work if the evidence supports a separate runtime.

### Follow-up constraints

- No pilot UI, MCP gateway, enterprise tenancy, or autonomous production executor is implied by this decision.
- Changes to core package seams require separate justification and tests.
- Repository extraction requires a new ADR supported by repeatable experimental evidence.
