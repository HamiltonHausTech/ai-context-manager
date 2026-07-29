# Adaptive Context Pilot Experiment Charter

- **Status:** Draft preregistration; confirmatory execution prohibited until open decisions and power/sensitivity simulation are resolved
- **Protocol version:** 1.1-draft
- **Intended dataset version:** `devops_v1`
- **Date:** 2026-07-28

This document prospectively specifies the controlled pilot described in the [implementation plan](../../.hermes/plans/2026-07-28_170432-adaptive-context-proof-plan.md). Protocol 1.1-draft supersedes 1.0 prospectively, before corpus authoring or sealing, model execution, or outcome access; the complete rationale and change trail are in the [protocol amendment record](adaptive-context-protocol-amendments.md). This is not yet a frozen preregistration and does not report evidence that the adaptive approach works.

Related records:

- [Protocol amendment record](adaptive-context-protocol-amendments.md)
- [ADR 0001: Use the existing repository for the pilot](../decisions/0001-use-existing-repo-for-pilot.md)
- [ADR 0002: Feedback is event data, not truth](../decisions/0002-feedback-is-event-data-not-truth.md)
- [ADR 0003: Strong static selection is the primary baseline](../decisions/0003-strong-static-selection-is-primary-baseline.md)
- [Adaptive context implementation plan](../../.hermes/plans/2026-07-28_170432-adaptive-context-proof-plan.md)

## Research question and sequential confirmatory claim

For recurring, controlled DevOps troubleshooting task families, after a fixed five-case adaptation curriculum, can a feedback-informed selector first preserve downstream quality relative to one locked strong static policy and, only if that quality guard passes, reduce input-context tokens?

The confirmatory decision is sequential:

1. Establish downstream task-quality non-inferiority of adaptive selection versus the **one locked primary static baseline**.
2. Only if the quality guard passes, test superiority on **one primary efficiency endpoint: input-context tokens**.

Correction rate is secondary/descriptive and cannot substitute for token efficiency. Quality superiority, if observed, is also secondary; it is not required to pass the confirmatory sequence.

## Hypotheses

### Confirmatory hypotheses

- **Quality guard (first):** the equal-family-weighted adaptive-minus-locked-static task-quality effect after the fixed five-case adaptation history is above the accepted non-inferiority boundary. The current margin is provisionally `-0.03` on the eventual normalized quality scale.
- **Efficiency test (second, conditional):** if and only if the quality guard passes, adaptive selection uses fewer input-context tokens than the locked static policy under the same maximum context-token budget.

The `0.03` quality margin is not accepted yet. The rubric scale and weights must be frozen, and the preregistered sensitivity/power simulation must show what effects this eight-family design can distinguish. Before protocol freeze, the team must also choose whether passing non-inferiority requires a conservative uncertainty bound above `-0.03` or uses an explicitly labeled heuristic continuation rule. No confirmatory execution may begin while that choice remains open.

### Secondary hypotheses and outcomes

These are not substitutes for either step of the confirmatory sequence:

1. Adaptive selection improves downstream quality.
2. Adaptive selection reduces correction count or correction rate.
3. Adaptive selection reduces irrelevant or misleading selected context.
4. Adaptive utility transfers to new context-item IDs within a known family through reusable features rather than ID memorization.

### Exploratory hypotheses

1. Feature-level utility estimates outperform component-ID-local estimates.
2. Effects vary by context role, memory kind, source scope, confidence, recency, or static relevance bucket.
3. Some task families benefit from static metadata alone while others benefit from adaptation.
4. Natural behavioral feedback is sufficiently dense and unambiguous to support stable estimates after controlled stages.

Exploratory findings may motivate later protocols but cannot be represented as confirmation of this protocol.

## Experimental unit, nesting, and estimand

The independent unit for primary inference is the **task-family learning trajectory**, yielding eight top-level units. The 24 held-out cases are clustered within their eight families; repeated generations are nested measurements within case and mode. Neither held-out cases nor generations are independent experimental units, and repetitions improve measurement precision rather than independent sample size.

The exact primary quality estimand is the **equal-family-weighted mean paired adaptive-minus-locked-static difference after the fixed five-case adaptation history, aggregating generations within each held-out case, then cases within each family, then giving each of the eight family effects equal weight**. The token estimand follows the same nesting and equal-family weighting; token superiority is expressed in the pre-freeze analysis specification as adaptive-minus-static tokens (with lower values favorable), with any relative reduction reported in addition to the absolute effect.

The analysis must not bootstrap 24 cases independently. It will use a predeclared family-clustered or hierarchical resampling/modeling method that respects the nesting. Because there are only eight top-level units, all intervals must be labeled **coarse**. Reports must show the raw eight family effects, their median and range, the equal-family mean, and the number in the favorable direction. Per-family effects are mandatory descriptive reporting, not eight separate confirmatory tests.

## Domain, corpus, and learning trajectories

The controlled domain is synthetic-but-realistic DevOps troubleshooting. The intended corpus contains eight task families:

1. Hybrid-network return routing.
2. Terraform drift and state problems.
3. Kubernetes scheduling failures.
4. IAM authorization failures.
5. DNS resolution failures.
6. Load-balancer health failures.
7. Security-group versus NACL diagnosis.
8. Certificate or TLS-chain failures.

The intended corpus has 64 cases: eight families by eight cases. Within each family, five cases form a fixed ordered adaptation curriculum and three are sealed held-out cases. The exact adaptation order must be declared before execution. Primary claims are restricted to learning under that curriculum; claims about order robustness require counterbalanced orders in a later protocol.

For the primary pilot, learned utility state is reset before each family so the eight family trajectories remain independent. The family execution order is recorded. Held-out context-item IDs differ from adaptation IDs while preserving predeclared reusable feature structure.

Each case defines its prompt, environment facts, eligible candidates, required/useful/misleading/irrelevant context, expected diagnostic steps, prohibited assumptions or unsafe actions, scoring rubric, provenance/template group, and the locked feedback event available after an adaptation run.

## Comparison conditions and baseline fairness

The experiment may execute these conditions:

1. **Full-context reference:** all eligible context that fits; a cost/attention reference, not a primary baseline.
2. **Similarity/top-K baseline:** a secondary comparison.
3. **Static explainable selector variants:** development candidates only.
4. **Locked primary static policy:** exactly one strongest credible non-learning rules-and-metadata policy selected using development/adaptation information only.
5. **Adaptive candidate:** the same static features plus transparent utility estimates learned only from permitted prior adaptation feedback.

Before freeze, the team must predefine the static-policy candidate set, development criterion, and deterministic tie-breaker. Exactly one primary static policy is then locked without held-out outcomes. Static and adaptive modes receive comparable engineering/tuning effort and the same access to development information and metadata, except that adaptive mode alone receives the protocol-defined prior adaptation feedback. They otherwise share the eligible candidate pool and representations, context-token budget, prompt/template, model/provider, generation configuration, tools, and scoring protocol. All contrasts other than adaptive versus this one locked static policy are secondary or exploratory. See [ADR 0003](../decisions/0003-strong-static-selection-is-primary-baseline.md).

## Feedback comparability

In oracle and simulated stages, the adaptation signal is a locked feedback record generated independently of which selector ran; selectors cannot change the label or amount of feedback they receive. This isolates selection-policy learning from policy-induced feedback.

Natural model/human feedback is different: it depends on the selected context and produced answer and therefore creates a policy-dependent online trajectory. Natural-feedback results must be labeled as such, analyzed as trajectories rather than interchangeable supervised labels, and cannot confirm the oracle/simulated claim without a later protocol designed for that estimand.

## Corpus sealing and leakage controls

1. Split corpus material at the **scenario-template/provenance-group** level, not merely by case or context-item ID.
2. Detect exact duplicates and near-duplicate text/content across adaptation, development, and held-out splits; retain normalized content hashes and the near-duplicate review record.
3. Freeze the feature ontology and adaptive-policy specification before held-out authoring, **or** use separate held-out authors who do not see selector-development results. Record which protection is used.
4. Require an independent domain reviewer, without access to selector results, to review required evidence, misleading/irrelevant distractors, expected conclusions, unsafe actions, and rubrics.
5. Reveal adaptation feedback only after its case executes, in the fixed within-family order. The adaptive selector may consume only prior permitted feedback.
6. Keep held-out prompts, labels, rubrics, feedback, and outcomes sealed from selector/policy development and learning until every held-out selection and generation is complete.
7. Before confirmatory execution, hash and seal prompts, candidate pools, labels, rubrics, scoring rules, corpus splits/provenance groups, policy configuration, code, model/prompt configuration, and manifests.
8. No selector, threshold, prompt, rubric, baseline, feature, exclusion rule, or aggregation may be tuned after held-out outcome access. A later change requires a new protocol and newly sealed evaluation.

## Outcomes and measurements

### Confirmatory endpoints

- One quality non-inferiority endpoint: frozen rubric-scored downstream task quality.
- One conditional efficiency-superiority endpoint: input-context tokens.
- The adaptive-versus-one-locked-static contrast.
- The ID-renaming/component-ID ablation, **only if retained in the final continuation gate before freeze**.

### Secondary and descriptive measurements

Every comparable mode reports overall and by family: quality and pass rate, correction count/rate, input tokens, critical/severe events, context precision/recall, irrelevant and misleading context, selection stability, and repeated-generation stability. Critical safety events are governed separately below.

### Exploratory measurements and analyses

Other selector/reference contrasts, per-family heterogeneity tests, retrieval/ranking metrics, latency, estimated cost, feature analyses, alternate quality weights, alternate thresholds, other ablations, post-hoc subgroups, and natural-feedback analyses are exploratory. Retrieval metrics can explain mechanisms but cannot override an incorrect or unsafe downstream answer.

## Scoring, blinding, and measurement validation

Scorers must be blinded to selector mode, selection traces, filenames, run order, and condition labels. Output presentation is randomized and uses opaque identifiers. Rubric calibration and scorer training use development cases only, never held-out outputs.

Two independent human scorers evaluate a substantial, predeclared stratified subset and every critical or safety judgment. The protocol freeze records the subset fraction/strata, agreement statistic, disagreement handling, and adjudication procedure; reports include agreement and adjudication counts. Any scorer identity, missing rating, and disagreement remains in the audit record.

LLM judging is exploratory unless it is prospectively validated against blinded human judgments to a predeclared acceptance criterion. Deterministic scorers are tested on meaning-preserving paraphrases and adversarial phrasing, not just keyword fixtures. The final rubric scale, weights, aggregation, and deterministic/human blend must be fixed before the sensitivity simulation and confirmatory execution.

## Repeated generations, order, and manifests

The recommended primary design uses **five matched repetitions** for the adaptive and locked-static modes; the minimum is three if cost-constrained. The final repetition count is an explicit pre-freeze decision. Repetitions use predeclared matching and aggregation and do not increase the independent sample size beyond eight trajectories.

Within execution constraints, primary-mode generations are randomized/interleaved across mode and opaque run order rather than running one mode in a block. Each generation is stateless except for the explicit protocol-controlled adaptive utility state. Disable uncontrolled provider/session memory, response caches, and hidden carry-over where technically possible; document any unavoidable cache. Record provider revision/model identifier and execution time. A frozen manifest also records corpus/code/policy hashes, prompt/template hash, temperature and seed where supported, tools, budget, token accounting, and raw-response hash. Incompatible manifests are not pooled.

## Sensitivity and power simulation before freeze

Before protocol freeze, preregister and run a simulation over plausible family-level effect distributions, within-family case variation, repeated-generation variation, quality-score bounds, and candidate family-clustered/hierarchical analyses. Report operating characteristics for the proposed non-inferiority rule and conditional token test, including detectable family-level effects, false-continuation behavior, sensitivity to outlier families, and consequences of three versus five repetitions.

With only eight top-level units, the simulation is for design calibration and limitations, not conventional p-value theater. The final protocol must state what family-level effects the design can and cannot reliably detect, justify or revise the `0.03` margin and token threshold, and label uncertainty as coarse. If plausible effects are not detectable, narrow the claim, add independent families, or treat the pilot as descriptive.

## Sequential continuation decision

The current continuation gate is a draft pending the simulation and open decisions:

1. Apply the predeclared quality non-inferiority rule versus the one locked static policy. Stop the confirmatory sequence if it fails.
2. Only after quality non-inferiority passes, apply the predeclared superiority rule to input-context tokens. Correction rate cannot satisfy this step.
3. Apply the zero-tolerance safety guardrail independently of statistical results.
4. If retained before freeze, require the ID-renaming/component-ID ablation to rule out an advantage primarily caused by ID memorization.

The exact token-superiority threshold and whether the quality gate uses a conservative uncertainty bound or a heuristic continuation rule remain open. Family-direction counts are descriptive and are not an additional multiplicity-generating gate unless prospectively justified before freeze. Passing supports only continued investigation, not broad organizational learning, production value, or cross-domain superiority.

## Safety engineering guardrail

Safety is not an equivalence or statistical “no increase” claim. Define severe/critical failure categories and stop triggers before freeze. Manually review every severe or critical event under blinded conditions, with two independent scorers and adjudication. Stop execution on any preregistered severe-failure trigger, preserve the partial audit trail, and investigate without trading the event against quality or token gains. Report event counts and the paired cases by mode; do not claim safety equivalence from sparse events.

## Negative controls and falsification

The staged harness must include no-effect controls where feedback cannot change utility, misleading shared-feature controls that should expose harmful transfer, feature perturbation tests, template/provenance-group ablations, and the predeclared ID-renaming ablation if retained. A valid harness must detect deliberately introduced leakage and false adaptive advantages.

The thesis is weakened or rejected for this pilot if adaptive quality breaches the accepted non-inferiority rule; token superiority fails after the guard passes; gains disappear under retained anti-memorization controls; static rules explain the result; improvement stays in adaptation cases; feedback is too sparse/ambiguous; effect direction is unstable; or a safety stop is triggered. Null, adverse, ambiguous, and negative results are reported completely.

The run is invalid—not silently repaired—if held-out data leak into adaptation, incompatible manifests undermine pairing, scorer blinding is materially broken, or corpus/rubric corruption prevents valid scoring. A defect found before outcome access may be corrected with an audit record and new frozen manifest. A defect found after outcome access requires a new protocol and newly sealed evaluation.

## Stage-specific claims

- **Tiny deterministic fixtures** can establish schema, ordering, leakage checks, state reset, scorer mechanics, and report reconstruction; they cannot establish empirical benefit.
- **Oracle feedback** can establish that learning logic uses known informative labels and transfers under constructed assumptions; it cannot establish learnability from behavioral feedback.
- **Simulated feedback** can establish behavior under a declared noise model; it cannot establish that the model resembles human/production feedback.
- **Frozen-model synthetic trials** can estimate model-mediated effects on the sealed synthetic domain under one provider revision and curriculum; they cannot establish external validity, natural-feedback causality, production safety, or unseen-domain transfer.
- **Naturalistic dogfood** can expose workflow, ambiguity, and operational failure modes; absent a separately preregistered controlled design, it cannot confirm the synthetic pilot or support causal product claims.

## Confirmatory versus exploratory boundary

Confirmatory scope is deliberately narrow: one adaptive-versus-locked-static contrast, one quality non-inferiority endpoint, one conditional input-token endpoint, and the ID-renaming ablation only if retained in the final gate. Raw per-family effects are mandatory descriptive reporting, not eight confirmatory tests. Everything else—including other selectors, heterogeneity tests, retrieval metrics, latency/cost, feature analyses, alternate weights, and other ablations—is secondary/descriptive or exploratory as labeled above and cannot retroactively change the decision.

## Feedback evidence boundary

Raw feedback is evidence about an event, not reusable truth. It is append-only and tied to the run, task, selected context, outcome, signal source, and provenance. Promotion into reusable experience is a separate reviewed lifecycle under [ADR 0002](../decisions/0002-feedback-is-event-data-not-truth.md).

## Amendment and audit policy

Corrections or clarifications create a new protocol version and a dated entry in the [amendment record](adaptive-context-protocol-amendments.md), including reason, exact change, author, timing relative to authoring/sealing/execution/outcome access, and analytical impact. There are no silent edits. Post-outcome amendments cannot redefine the original analysis or gate; they define exploratory work or a new experiment with newly sealed data.

## Explicit limitations

- Eight independent family trajectories provide coarse precision and weak support for distributional assumptions; 24 held-out cases and repeated generations do not change that top-level sample size.
- The synthetic corpus may not reproduce real incident distributions, incentives, ambiguity, or operational constraints.
- Authored rubrics may encode author assumptions despite independent review and blinding.
- Provider behavior can drift despite frozen settings and recorded revisions.
- Fixed within-family order identifies effects for this curriculum, not order-robust learning.
- Known-family held-out cases test transfer within represented families, not unseen domains.
- This pilot cannot prove broad organizational learning, durable production benefit, causal credit assignment from natural feedback, safety equivalence, or safety in autonomous operations.

## Open decisions required before protocol freeze

1. Exact model/provider, provider-revision policy, and generation configuration.
2. Three or five matched repetitions; exact interleaving, matching, and generation aggregation.
3. Frozen rubric scale/weights; human subset fraction/strata; agreement/adjudication rules; deterministic/human scoring blend; any human-validation criterion for an LLM judge.
4. The simulation inputs, family-clustered/hierarchical uncertainty method, and acceptance criterion for detectable effects.
5. Whether quality non-inferiority uses a conservative uncertainty bound or an explicitly heuristic continuation rule; acceptance or revision of the provisional `0.03` margin.
6. Exact input-token superiority threshold and scale (absolute plus relative reporting).
7. Static-policy candidate set, development criterion, and deterministic tie-breaker; evidence of comparable tuning opportunity.
8. Final severe/critical taxonomy and preregistered stop triggers.
9. Whether the ID-renaming ablation remains part of the confirmatory gate and its pass criterion.
10. Exact fixed adaptation order per family, family execution-order procedure, and state-reset verification.
11. Corpus leakage protection choice: ontology/policy frozen before held-out authoring or separate held-out authors; provenance grouping, duplicate/near-duplicate method, and independent reviewer identity/process.
12. Which artifacts may be committed versus retained securely.

All items must be resolved, simulated where applicable, versioned, reviewed, and recorded in a later frozen protocol before corpus sealing/model execution. Confirmatory execution under 1.1-draft is prohibited.
