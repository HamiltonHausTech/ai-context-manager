# Adaptive Context Pilot Experiment Charter

- **Status:** Preregistered protocol; final acceptance of the provisional threshold is an open decision
- **Protocol version:** 1.0
- **Intended dataset version:** `devops_v1`
- **Date:** 2026-07-28

This document preregisters the controlled pilot described in the [approved implementation plan](../../.hermes/plans/2026-07-28_170432-adaptive-context-proof-plan.md). It defines the confirmatory claim, comparison conditions, measurements, analysis boundary, and conditions under which the claim would be weakened or rejected. It does not report evidence that the adaptive approach works.

Related architecture decisions:

- [ADR 0001: Use the existing repository for the pilot](../decisions/0001-use-existing-repo-for-pilot.md)
- [ADR 0002: Feedback is event data, not truth](../decisions/0002-feedback-is-event-data-not-truth.md)
- [ADR 0003: Strong static selection is the primary baseline](../decisions/0003-strong-static-selection-is-primary-baseline.md)

## Research question

For recurring, controlled DevOps troubleshooting tasks, does a feedback-informed context selector improve downstream outcomes on held-out cases from known task families relative to the strongest credible static rules-and-metadata selector, when both operate on the same eligible candidates and under the same context-token budget?

## Hypotheses

### Confirmatory primary hypothesis

On held-out cases from known task families, the adaptive selector will produce higher downstream task quality than the strongest static rules-and-metadata baseline under the same candidate pool and token budget.

Downstream task quality is the primary outcome. Retrieval measures explain possible mechanisms and failures but cannot establish the primary hypothesis by themselves.

### Secondary hypotheses

These hypotheses are labeled secondary and are not substitutes for the primary outcome:

1. Adaptive selection reduces the rate of corrections required on held-out cases.
2. Adaptive selection reduces irrelevant or misleading context selected.
3. Adaptive selection reduces input-context tokens without a material loss of downstream task quality.
4. Adaptive utility transfers to new context-item IDs within a known task family through reusable features rather than ID memorization.

### Exploratory hypotheses

1. Feature-level utility estimates outperform component-ID-local estimates.
2. Effects differ predictably by context role, memory kind, source scope, confidence, recency, or static relevance bucket.
3. Some task families benefit from static metadata alone while others benefit from adaptation.
4. Natural behavioral feedback is sufficiently dense and unambiguous to support stable estimates after the controlled stages.

Exploratory findings may motivate later protocols but will not be represented as confirmation of the primary hypothesis.

## Experimental unit and domain

The experimental unit is a **task-case execution under one selector mode and one frozen run configuration**. Paired comparisons use executions of the same task case across modes; repeated model runs are nested within task case and mode rather than treated as independent cases.

The controlled domain is synthetic-but-realistic DevOps troubleshooting. The intended corpus contains eight task families:

1. Hybrid-network return routing.
2. Terraform drift and state problems.
3. Kubernetes scheduling failures.
4. IAM authorization failures.
5. DNS resolution failures.
6. Load-balancer health failures.
7. Security-group versus NACL diagnosis.
8. Certificate or TLS-chain failures.

The intended corpus has **64 cases: 8 families × 8 cases**. Within each family, five ordered cases are adaptation cases and three are sealed held-out cases, for 40 adaptation cases and 24 held-out cases overall. Held-out cases will use context-item IDs not seen during adaptation while preserving reusable feature structure.

Each case will define its prompt, environment facts, eligible context candidates, required and useful context, misleading and irrelevant context, expected diagnostic steps, prohibited assumptions or unsafe actions, a scoring rubric, and the feedback event revealed after an adaptation run.

## Comparison conditions

The experiment will compare:

1. **Full-context reference:** all eligible context that fits the reference condition. This measures token cost and possible attention dilution; it is not the primary baseline.
2. **Similarity/top-K baseline:** similarity-ranked selection packed under the common token budget.
3. **Static explainable selector:** query relevance, lifecycle filtering, metadata, recency, and fixed importance weights, with an inclusion/exclusion trace.
4. **Strongest static rules + metadata baseline:** the strongest credible non-learning policy developed using adaptation/validation information only. This is the primary baseline.
5. **Adaptive candidate:** the static feature set plus transparent utility estimates learned only from previously revealed adaptation feedback.

For comparable budgeted modes, the task, the same eligible candidate pool, candidate representations, context-token budget, prompt/template, model and provider, generation configuration, tool availability, and scoring protocol must be identical. The adaptive mode receives neither richer metadata nor a larger budget. The full-context condition is a reference and any unavoidable budget difference must be explicit in the manifest and excluded from claims of budget-matched superiority.

Static-policy selection and baseline choice must be frozen before confirmatory held-out outcomes are viewed. See [ADR 0003](../decisions/0003-strong-static-selection-is-primary-baseline.md).

## Outcomes and required measurements

### Primary outcome

The primary outcome is rubric-scored **downstream task quality** on sealed held-out cases. The rubric will reward required diagnostic reasoning and correct conclusions and penalize critical omissions, false claims, prohibited assumptions, and unsafe actions. The final deterministic/human scoring blend remains an open decision and must be fixed before confirmatory execution.

### Required secondary and diagnostic measurements

Every comparable mode will report, overall and by task family:

- downstream task-quality score and pass/success rate;
- critical or safety-related failures;
- corrections required and correction rate;
- input-context tokens;
- context precision and recall;
- irrelevant context selected;
- misleading context selected;
- latency and estimated cost;
- selection and outcome stability across repeated runs.

Retrieval metrics, including context precision, recall, ranking measures, and token efficiency, are diagnostic. Good retrieval scores cannot override an incorrect or unsafe downstream answer.

## Corpus sealing and leakage controls

1. Adaptation cases run in declared order; feedback is revealed only after the corresponding adaptation execution.
2. Held-out prompts, labels, rubrics, feedback, and outcomes are sealed from selection-policy development and learning.
3. The adaptive selector may learn only from prior adaptation feedback. It may not consume held-out feedback until all held-out selection and generation are complete.
4. The corpus, code, policy configuration, prompt, model configuration, and manifests are versioned and hashed before confirmatory evaluation.
5. No selector, threshold, prompt, rubric, baseline, or feature may be tuned after viewing confirmatory held-out outcomes. Any later change belongs to a new protocol and a new held-out evaluation.

## Scoring and blinding

Deterministic rubric checks will be used wherever possible, especially for required steps, explicit false claims, prohibited actions, and unsafe recommendations. Where human judgment is required, outputs will be presented without selector-mode labels and scored against the frozen rubric. Rater identity, adjudication, missing ratings, and disagreements will be retained in the audit record. Model-based judging, if used, will be declared in the final scoring blend and treated as a frozen measurement component rather than ground truth.

## Repeated runs and frozen manifests

Each model condition will be run repeatedly; the exact number of repetitions is an open decision that must be resolved before confirmatory execution. Each run will have a frozen manifest recording at least corpus version/hash, code revision, selector/policy version, provider and exact model identifier, prompt/template hash, temperature and seed where supported, tool availability, token budget, timestamps, token accounting, and raw-response hash. Runs with incompatible manifests will not be pooled in the confirmatory comparison.

## Confirmatory analysis

The confirmatory analysis population is all valid, preregistered held-out task cases run under compatible frozen manifests. Exclusions may be made only for predeclared integrity failures, such as a provider failure that produced no answer or verified held-out leakage; every exclusion and its reason must be reported.

Analysis will include:

- paired adaptive-minus-primary-baseline effect sizes on task quality;
- uncertainty estimates and confidence intervals for paired effects;
- paired comparisons for correction rate and input tokens;
- critical-failure counts, without allowing gains elsewhere to offset a safety regression;
- per-family effects and the number of families showing the declared direction;
- stability across repeated runs;
- feature-only, ID-local/component-ID, and no-learning/static ablations;
- complete reporting of null, adverse, ambiguous, and negative results.

The confidence-interval method and final scoring aggregation will be frozen before confirmatory execution. Analyses not listed as confirmatory here—including post-hoc subgroups, alternate scoring blends, alternate thresholds, and feature mining—are exploratory and must be labeled as such.

## Provisional continuation gate

The following gate is preregistered as **provisional** and must receive final threshold acceptance before held-out results are viewed:

- Mean adaptive quality is no worse than 0.03 below the best static baseline.
- Adaptive selection improves either correction rate or input-token use by at least 20% relative to the best static baseline.
- There is no increase in critical or safety-related failures.
- Improvement is visible in at least five of eight task families.
- A component-ID ablation indicates that the result is not primarily memorization.

Passing this gate supports only a decision to **continue investigating**. It is not proof of broad organizational learning, production value, or superiority in other domains.

## Falsification and stopping conditions

The primary thesis is weakened or rejected for this pilot if, after documented correction of implementation or measurement defects, any of the following persists:

- adaptive held-out quality degrades by more than the predeclared tolerance;
- apparent gains disappear when held-out component IDs are replaced or ID-local utility is disabled;
- strong static rules and metadata account for essentially all observed improvement;
- feedback improves adaptation cases but not held-out cases;
- improvement is concentrated in too few task families to satisfy the gate;
- feedback is too sparse or ambiguous to produce stable utility estimates;
- effect direction is unstable across repeated runs under the frozen configuration;
- critical or safety-related failures increase.

The confirmatory run must stop and be declared invalid—not silently repaired—if held-out data leak into adaptation, manifests are incompatible across compared conditions, scorer blinding is broken in a way likely to bias judgments, or corpus/rubric corruption prevents valid scoring. An implementation defect discovered before held-out outcomes are inspected may be corrected with an audit record and a new frozen manifest. A defect discovered after inspection requires a new protocol version and newly sealed confirmatory evaluation. Resource or provider failures may stop execution without a scientific conclusion; partial results will be retained and labeled incomplete.

## Confirmatory versus exploratory boundary

Only the primary hypothesis, declared secondary outcomes, primary-baseline comparison, population, measurements, paired analyses, ablations, per-family analysis, and gate specified above are confirmatory. Alternative models, prompts, selectors, feature sets, task-family regroupings, exclusion rules, thresholds, scoring blends, or analyses conceived after viewing held-out outcomes are exploratory. Exploratory work must be reported separately and cannot retroactively change whether this protocol passed.

## Feedback evidence boundary

Raw feedback is evidence about an event, not reusable truth. It is append-only and tied to the run, task, selected context, outcome, signal source, and provenance. Promotion into reusable experience is a separate reviewed lifecycle, as specified by [ADR 0002](../decisions/0002-feedback-is-event-data-not-truth.md).

## Amendment and audit policy

Corrections or clarifications create a new protocol version and a dated audit record describing the reason, exact change, author, timing relative to corpus sealing and outcome access, and expected analytical impact. After preregistration, there will be **no silent edits**. Amendments made after confirmatory outcomes are viewed cannot redefine the original confirmatory analysis or gate; they define exploratory work or a new experiment with newly sealed data.

## Explicit limitations

- The corpus is synthetic and may not reproduce the distribution, incentives, ambiguity, or operational constraints of real incidents.
- Rubrics are authored and may encode author assumptions or favor particular diagnostic styles.
- Model stochasticity can remain despite frozen settings, and providers may not guarantee deterministic execution.
- Sixty-four cases provide a pilot-scale estimate and limited per-family precision.
- Known-family held-out cases test transfer within represented families, not transfer to unseen domains.
- This pilot cannot prove broad organizational learning, durable production benefit, causal credit assignment from natural feedback, or safety in autonomous operations.

## Open decisions to resolve before confirmatory execution

The following are deliberately unresolved and are not to be inferred from later defaults:

1. **Exact model/provider and generation configuration.**
2. **Number of repeated model runs per case and mode.**
3. **Final scoring blend** among deterministic checks, blinded human review, and any frozen model judge.
4. **Final threshold acceptance** for the provisional continuation gate, including whether the 20% relative improvement and 0.03 quality tolerance are accepted unchanged.

These decisions must be resolved, versioned, and audited before held-out outcomes are viewed. If their resolution changes the confirmatory design, protocol version 1.1 (or later) will supersede this version prospectively; version 1.0 remains in history.
