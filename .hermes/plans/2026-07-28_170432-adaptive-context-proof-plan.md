# Adaptive Context Proof Experiment Implementation Plan

> **For Hermes:** Use the `subagent-driven-development` skill to implement this plan task-by-task. Do not begin implementation until the experiment charter and continuation gates are approved.

> **Scientific interpretation:** The [protocol 1.1-draft charter](../../docs/research/adaptive-context-charter.md) is normative for hypotheses, units, estimands, confirmatory boundaries, gates, safety, leakage controls, and stage-specific claims. The [amendment record](../../docs/research/adaptive-context-protocol-amendments.md) explains prospective changes from 1.0, and [ADR 0003](../../docs/decisions/0003-strong-static-selection-is-primary-baseline.md) governs the one-baseline lock. If this implementation plan conflicts with the charter, follow the charter and amend the plan; do not infer a scientific decision from an implementation default.

**Goal:** Determine whether feedback-informed context selection preserves downstream quality versus one locked strong static policy and, only after that guard passes, reduces input-context tokens on held-out cases after a fixed adaptation curriculum.

**Architecture:** Keep `ai-context-manager` as the experimental substrate rather than creating a new repository now. Reuse its explainable retrieval, token budgeting, memory lifecycle, provenance, hybrid scoring, storage, and evaluation code; add an isolated research harness under `experiments/adaptive_selection/`. Promote code into the core package—or extract a new runtime—only after the experiment produces repeatable evidence.

**Tech Stack:** Python 3.11, existing `ai_context_manager` package, dataclasses, SQLite, JSON fixtures, pytest, existing model/provider dependencies, deterministic fake providers for unit tests, and a frozen hosted or local model for controlled trials.

---

## 1. Decision and current baseline

### Recommendation

Do **not** start a clean repository yet. The existing repository already contains most of the non-novel infrastructure needed for the test:

- `ai_context_manager/retrieval.py`: candidate filtering, query-aware ranking, deduplication, token-budget packing, compression, and inclusion/exclusion traces.
- `ai_context_manager/feedback.py`: component-level time-decayed scores.
- `ai_context_manager/hybrid.py`: semantic/importance/recency/feedback weighting.
- `ai_context_manager/memory.py` and `consolidation.py`: memory kinds, confidence, provenance, contradiction, expiry, and supersession.
- `ai_context_manager/evaluation.py`: precision, recall, reciprocal rank, NDCG, token efficiency, exclusion accuracy, stability, and a downstream-utility hook.
- `evaluations/agent_memory_retrieval.json`: an initial but very small retrieval fixture.
- `bourbon_research/`: a useful naturalistic dogfood application with noisy-source and context-selection problems.

The repository is clean on `main`; its configured test suite passes, with the live PostgreSQL integration test skipped unless explicitly enabled.

### What the repository does **not** yet prove

The current test `test_feedback_improves_measured_retrieval_quality` manually assigns a higher feedback score to the relevant record and then verifies that feedback-weighted ranking moves it upward. That proves score plumbing, not learning from task outcomes.

Missing experimental capabilities:

1. Versioned task profiles and task families.
2. Rich feedback events tied to a run, selected context, outcome, correction type, and provenance.
3. Comparable execution modes using the same candidate pool and token budget.
4. A learned utility estimate that generalizes beyond a component ID.
5. A held-out corpus with required, useful, misleading, and irrelevant context.
6. Frozen generation configuration and reproducible run manifests.
7. Blinded or deterministic outcome scoring.
8. Confidence intervals, per-family analysis, and explicit continuation/stopping gates.

### Scope boundary

For the pilot, do not build a SaaS UI, MCP gateway, agent orchestrator, autonomous production executor, vector platform, enterprise tenancy, or automatic promotion of feedback into organizational truth.

### Decisions confirmed with Andrew (2026-07-28)

- Use the existing `ai-context-manager` repository as the pilot substrate.
- Use controlled DevOps troubleshooting as the primary experimental corpus.
- Keep `bourbon_research` in its current role: Andrew's personal project and a naturalistic dogfood application for exercising the context manager. It is not the primary controlled benchmark.
- Treat the project as a scientific investigation: preregister confirmatory hypotheses and gates, distinguish exploratory from confirmatory analysis, prefer effect sizes and uncertainty over anecdotes, preserve negative results, and do not alter held-out criteria after seeing outcomes.

---

## 2. Experiment charter

### Sequential confirmatory hypotheses

After a fixed five-case adaptation history in each family, first establish adaptive task-quality non-inferiority versus exactly one prospectively locked strong static policy. Only if that guard passes, test superiority on one primary efficiency endpoint: input-context tokens. The independent units are the eight task-family learning trajectories; cases are clustered within family and generations are nested measurements.

The exact estimand is the equal-family-weighted mean paired adaptive-minus-locked-static difference, aggregating generations within held-out case and cases within family. The provisional quality margin is `-0.03`, but it is not accepted until the rubric scale/weights and pre-freeze sensitivity/power simulation justify it. The frozen protocol must specify whether the guard uses a conservative uncertainty bound or an explicitly heuristic continuation rule.

### Secondary hypotheses

1. Adaptive selection improves quality, reduces corrections, or reduces irrelevant/misleading context.
2. Learned utility transfers to unseen cases and unseen context-item IDs within a known task family.
3. Feature-level learning performs better than memorizing component IDs.

Correction rate is secondary/descriptive and cannot replace the input-token endpoint. Quality superiority is also secondary.

### Falsification conditions

The thesis is weakened or rejected for this design if any of the following persist after correcting implementation defects:

- Adaptive selection degrades held-out quality by more than the predeclared tolerance.
- Gains disappear when component IDs are replaced on held-out cases.
- Static metadata/rules account for essentially all improvement.
- Feedback helps adaptation cases but not held-out cases.
- Effects are too unstable across family trajectories to support the scoped claim.
- Natural feedback is too sparse or ambiguous to produce stable utility estimates.
- Results change direction across repeated runs with the frozen model configuration.

### Provisional continuation gate

Lock the final gate before corpus sealing or model execution. The draft sequence is:

1. Pass the predeclared quality non-inferiority rule against the one locked static policy; otherwise stop the confirmatory sequence.
2. Only then pass the predeclared superiority rule for input-context tokens. Corrections cannot substitute.
3. Satisfy the separately preregistered zero-tolerance severe/critical safety guardrail; manually review every such event and stop on declared triggers. Do not claim statistical safety equivalence.
4. If retained before freeze, pass the component-ID/ID-renaming ablation.

Raw effects for all eight families, their median/range, and family-direction counts are mandatory descriptive reporting, not eight confirmatory tests or an unqualified five-of-eight gate. The exact token threshold and quality decision rule remain open pending simulation.

Treat this as a **continue-investigating** threshold, not proof of a company or broad organizational learning.

---

## 3. Experimental design

### Corpus

Create eight recurring task families with eight cases each: 64 total cases, split within each family into five ordered adaptation cases and three held-out cases (40 adaptation, 24 held out).

Predeclare the exact five-case adaptation order in every family and reset learned utility state between families. Primary claims apply only to that curriculum; record family execution order. Split at scenario-template/provenance-group level rather than ID alone, detect exact and near duplicates across splits using normalized content hashes, and preserve the review log. Freeze feature ontology and adaptive policy before held-out authoring or use separate held-out authors, recording the chosen protection.

Suggested families:

1. Hybrid-network return routing.
2. Terraform drift and state problems.
3. Kubernetes scheduling failures.
4. IAM authorization failures.
5. DNS resolution failures.
6. Load-balancer health failures.
7. Security-group versus NACL diagnosis.
8. Certificate or TLS-chain failures.

Each case must include:

- task prompt and task-family ID;
- environment facts;
- candidate context items;
- required context IDs;
- useful-but-optional context IDs;
- misleading context IDs;
- irrelevant context IDs;
- expected diagnostic steps;
- prohibited assumptions or unsafe actions;
- scoring rubric;
- feedback event to reveal after the adaptation run;
- stable IDs for the case, but new context-item IDs in held-out cases.

Use synthetic-but-realistic infrastructure cases first so ground truth is controlled. Use `bourbon_research` later as naturalistic dogfood, not as the primary proof corpus: changing web sources and subjective narrative quality make it a poor first controlled benchmark.

An independent domain reviewer who cannot see selector results must review required evidence, distractors, expected conclusions, unsafe actions, and rubrics. Hash and seal prompts, candidate pools, labels, rubrics, scoring rules, provenance-group splits, and manifests.

### Execution modes

All comparable selectors receive the same task, candidates, token budget, model, system prompt, and generation settings.

1. **Reference: full eligible context** — establishes the high-token reference and reveals attention dilution; not the primary fairness baseline.
2. **Baseline A: static explainable selector** — existing query relevance, lifecycle filters, metadata, recency, and fixed importance weights.
3. **Baseline B: similarity/top-K** — semantic or deterministic fixture similarity under the same token budget.
4. **Baseline C: locked strong static rules + metadata** — exactly one non-learning selector chosen on development/adaptation information using a predefined criterion and deterministic tie-breaker; this is the sole primary baseline.
5. **Candidate D: adaptive selector** — the same static features plus utility learned only from prior adaptation feedback.

Do not give the adaptive selector richer metadata or a larger budget than the baselines. Give static and adaptive development comparable metadata access, engineering effort, reviewer access, and tuning/search opportunity. Alternative baselines and static candidates are secondary/exploratory.

### Staged learning

- **Stage 0: tiny deterministic fixtures** — verify schema, ordering, state reset, controls, leakage detection, scorer mechanics, and report reconstruction; this cannot establish empirical benefit.
- **Stage 1: oracle feedback** — use selector-independent locked corpus labels to verify learning under constructed assumptions; this cannot establish learnability from behavioral feedback.
- **Stage 2: simulated behavioral feedback** — use selector-independent locked signals from a declared noise model; this cannot validate that noise model against human feedback.
- **Stage 3: frozen-model synthetic execution** — estimate effects for one sealed synthetic domain, provider revision, and curriculum with blinded scoring; this cannot establish production value, natural-feedback causality, unseen-domain transfer, or safety.
- **Stage 4: naturalistic dogfood** — use policy-dependent model/human feedback in a bounded workflow and label it as an online trajectory; absent a separate preregistration, this can expose workflow failures but cannot confirm the synthetic result.

A stage must pass its integrity checks before moving to the next.

Include no-effect controls, misleading shared-feature controls, feature perturbations, template/provenance ablations, and the ID-renaming ablation if retained. Oracle/simulated feedback must be identical regardless of selector behavior; natural feedback is policy-dependent and estimates a different trajectory.

### Initial adaptive policy

Keep v0 transparent. Do not begin with reinforcement learning or a complex ML model.

Represent each context item with reusable features such as:

- task family;
- memory kind;
- context role/type;
- source scope;
- tags/capabilities;
- static relevance bucket;
- confidence bucket;
- recency bucket.

Maintain a smoothed utility estimate for features and feature combinations. Score an item as:

```text
adaptive_score = static_score + learning_weight * estimated_feature_utility
```

Record every contributing feature and learned value in the decision trace. Do not use component ID as the primary learned feature; keep an ID-local signal only as an explicit ablation.

Do not assign equal credit to every selected item after a successful run and call that causal learning. In the controlled corpus, use item/rubric labels for Stage 1. In later stages, record outcome feedback separately and test leave-one-item-out or paired ablation for suspected high-impact items.

---

## 4. Proposed files

### Research documentation

- Create: `docs/research/adaptive-context-charter.md`
- Create: `docs/research/adaptive-context-evaluation.md`
- Create: `docs/decisions/0001-use-existing-repo-for-pilot.md`
- Create: `docs/decisions/0002-feedback-is-event-data-not-truth.md`
- Create: `docs/decisions/0003-primary-baseline-is-strong-static-selection.md`

### Experimental harness

- Create: `experiments/adaptive_selection/__init__.py`
- Create: `experiments/adaptive_selection/schema.py`
- Create: `experiments/adaptive_selection/repository.py`
- Create: `experiments/adaptive_selection/selectors.py`
- Create: `experiments/adaptive_selection/learning.py`
- Create: `experiments/adaptive_selection/providers.py`
- Create: `experiments/adaptive_selection/scoring.py`
- Create: `experiments/adaptive_selection/runner.py`
- Create: `experiments/adaptive_selection/report.py`
- Create: `experiments/adaptive_selection/cli.py`
- Create: `experiments/adaptive_selection/config/pilot.toml`
- Create: `experiments/adaptive_selection/datasets/devops_v1.json`
- Create: `experiments/adaptive_selection/README.md`

### Tests

- Create: `tests/adaptive_selection/test_schema.py`
- Create: `tests/adaptive_selection/test_repository.py`
- Create: `tests/adaptive_selection/test_selectors.py`
- Create: `tests/adaptive_selection/test_learning.py`
- Create: `tests/adaptive_selection/test_scoring.py`
- Create: `tests/adaptive_selection/test_runner.py`
- Create: `tests/adaptive_selection/test_report.py`
- Create: `tests/adaptive_selection/fixtures/tiny_experiment.json`

### Existing files likely to change later

- Modify: `pyproject.toml` — add an experiment CLI entry point and only the minimal development dependencies actually required.
- Modify: `.gitignore` — ignore generated experiment databases, raw model outputs, and result bundles while retaining small checked-in golden reports.
- Modify: `README.md` — distinguish proven package behavior from the new research hypothesis.
- Modify: `ai_context_manager/evaluation.py` only if generic paired-comparison metrics belong in the core package; otherwise keep them isolated in the harness.
- Avoid modifying `ai_context_manager/retrieval.py` until the harness demonstrates a missing reusable seam.

---

## 5. Implementation tasks

### Task 1: Resolve and freeze the draft experiment charter

**Objective:** Resolve protocol 1.1-draft's open decisions and simulation before freezing hypotheses, endpoints, one primary baseline, corpus split, sequential gate, and falsification criteria. Confirmatory execution is prohibited while the charter remains draft.

**Files:**
- Create: `docs/research/adaptive-context-charter.md`
- Create: `docs/research/adaptive-context-protocol-amendments.md`
- Create: `docs/decisions/0001-use-existing-repo-for-pilot.md`

**Steps:**
1. Treat the linked charter as normative and preserve amendments prospectively.
2. Freeze the rubric scale/weights and candidate family-clustered or hierarchical analysis.
3. Preregister and run a sensitivity/power simulation over plausible family, case, and generation variation; report detectable family-level effects and three-versus-five repetition behavior without conventional p-value theater.
4. Resolve every open question in the charter, including the non-inferiority rule, token threshold, repetitions, scoring, safety stops, static tie-breaker, order, and leakage controls.
5. Review the resulting frozen version before any corpus sealing or model execution.

**Verification:** A reviewer can identify the eight independent units, exact estimands, sole primary contrast, sequential endpoints, simulation justification, held-out set, safety stops, and exploratory boundary without reading code.

### Task 2: Define versioned experiment records

**Objective:** Make every task, context item, run, selection, outcome, and feedback event reproducible.

**Files:**
- Create: `experiments/adaptive_selection/schema.py`
- Test: `tests/adaptive_selection/test_schema.py`

**Required types:**

```python
TaskCase
TaskProfile
ContextItem
RunManifest
SelectionDecision
TaskOutcome
FeedbackEvent
UtilityEstimate
ExperimentResult
```

`FeedbackEvent` must include at least: event ID, run ID, task-case ID, task-family ID, signal type, value, optional context-item IDs, correction text/category, source (`oracle`, `simulated`, `human`, `judge`), timestamp, and provenance.

**TDD cycle:** Write JSON round-trip, validation, and schema-version rejection tests; verify failure; implement the minimum dataclasses/serialization; verify pass; commit.

### Task 3: Add an append-only SQLite experiment repository

**Objective:** Persist raw experimental evidence without conflating it with approved memory.

**Files:**
- Create: `experiments/adaptive_selection/repository.py`
- Test: `tests/adaptive_selection/test_repository.py`

**Tables:**

```text
experiment_runs
selection_decisions
model_outputs
outcomes
feedback_events
utility_estimates
```

Keep feedback events append-only. Corrections create new events; they do not rewrite history. Utility estimates are derived artifacts and must record the source event IDs and learning-policy version.

**Verification:** A temporary SQLite database can round-trip a complete run, preserve ordering and provenance, and rebuild utility estimates from raw events.

### Task 4: Create the tiny deterministic corpus

**Objective:** Prove the harness before investing in the full 64-case corpus.

**Files:**
- Create: `tests/adaptive_selection/fixtures/tiny_experiment.json`
- Test: `tests/adaptive_selection/test_schema.py`

Include two task families, two adaptation cases per family, and one held-out case per family. Held-out context IDs must differ from adaptation IDs while preserving reusable features.

**Verification:** Schema validation passes and fails clearly for missing rubrics, overlapping required/excluded IDs, duplicate IDs, or leaked feedback.

### Task 5: Implement comparable selectors

**Objective:** Make the baselines and candidate differ only in selection policy.

**Files:**
- Create: `experiments/adaptive_selection/selectors.py`
- Test: `tests/adaptive_selection/test_selectors.py`

Implement:

```python
FullContextSelector
SimilarityTopKSelector
StaticPolicySelector
AdaptivePolicySelector
```

Reuse `RetrievalPipeline.select_candidates`, ranking signals, and budget-packing behavior rather than duplicating token logic. Every selector returns selected items plus inclusion/exclusion reasons and score factors.

**Verification:** Given the same candidates and budget, every mode obeys the exact budget, sees the same eligible pool, and emits a complete trace.

### Task 6: Implement transparent feature-level learning

**Objective:** Convert prior feedback events into inspectable utility estimates without component-ID memorization.

**Files:**
- Create: `experiments/adaptive_selection/learning.py`
- Test: `tests/adaptive_selection/test_learning.py`

Start with smoothed mean rewards using configurable priors and minimum evidence counts. Implement separate feature-level and ID-local estimators so the ID-local contribution can be disabled.

**Required tests:**

- no feedback equals static ranking;
- relevant feedback raises reusable feature utility;
- correction feedback lowers the appropriate feature utility;
- held-out IDs benefit only from shared features;
- one extreme event cannot dominate because of smoothing;
- later correction events do not erase earlier provenance;
- changing learning-policy version forces estimate recomputation.

### Task 7: Add deterministic task scoring

**Objective:** Measure task outcome independently of retrieval metrics.

**Files:**
- Create: `experiments/adaptive_selection/scoring.py`
- Test: `tests/adaptive_selection/test_scoring.py`

Score required diagnostic steps, critical omissions, false claims, prohibited actions, correction count, and rubric total. Keep retrieval precision/NDCG as diagnostic metrics, not the primary task-quality result.

Blind scorers to mode, traces, filenames, run order, and condition labels; randomize presentation. Calibrate rubrics on development cases only. Use two independent human scorers for a predeclared substantial stratified subset and every critical judgment, and report agreement/adjudication. Keep LLM judging exploratory unless prospectively validated against blinded humans. Test deterministic scoring against paraphrases and adversarial wording, not only keywords.

**Verification:** The scorer penalizes an unsafe or technically wrong answer even if it retrieved nominally relevant context.

### Task 8: Add frozen model adapters and run manifests

**Objective:** Prevent model drift and configuration differences from masquerading as learning.

**Files:**
- Create: `experiments/adaptive_selection/providers.py`
- Test: `tests/adaptive_selection/test_runner.py`

Implement a provider protocol and a deterministic fake provider first. A real provider adapter must record provider, exact model identifier, prompt/template hash, temperature, seed where supported, tool availability, timestamps, token accounting, and raw response hash.

For the primary adaptive/static modes, support five matched repetitions (minimum three if cost-constrained; final count resolved before freeze), randomized/interleaved execution, stateless generations, disabled uncontrolled memory/caches, provider revision/time capture, and predeclared aggregation. Repetitions are nested measurements, not new independent units.

**Verification:** Rerunning the fake provider reproduces byte-identical outputs and manifests. The harness refuses to compare runs with incompatible manifests unless explicitly overridden and labeled invalid for primary comparison.

### Task 9: Build the ordered experiment runner

**Objective:** Execute adaptation cases sequentially while preventing held-out leakage.

**Files:**
- Create: `experiments/adaptive_selection/runner.py`
- Create: `experiments/adaptive_selection/cli.py`
- Test: `tests/adaptive_selection/test_runner.py`

Required commands:

```bash
python -m experiments.adaptive_selection.cli validate --dataset ...
python -m experiments.adaptive_selection.cli run --mode static --dataset ... --output ...
python -m experiments.adaptive_selection.cli run --mode adaptive --dataset ... --output ...
python -m experiments.adaptive_selection.cli compare --left ... --right ...
```

The adaptive run may learn only from selector-independent locked feedback revealed after each adaptation case in the fixed predeclared within-family order. Reset utility state between families and record family order. Held-out feedback remains sealed until selection and generation are complete. Natural/model-human feedback must use a separately labeled policy-dependent online trajectory.

**Verification:** A leakage test deliberately places the answer in held-out feedback and confirms it cannot influence selection.

### Task 10: Produce paired reports and uncertainty estimates

**Objective:** Report the sequential adaptive-versus-one-locked-static decision and why, without pseudoreplication.

**Files:**
- Create: `experiments/adaptive_selection/report.py`
- Test: `tests/adaptive_selection/test_report.py`

Report overall and per-family:

- task-quality score;
- success/pass rate;
- critical failures;
- corrections required;
- selected input tokens;
- context precision/recall;
- irrelevant and misleading context selected;
- latency and estimated cost;
- equal-family-weighted paired differences after the fixed five-case history;
- family-clustered/hierarchical coarse uncertainty intervals, never an independent 24-case bootstrap;
- all eight raw family effects plus median and range;
- feature-level versus ID-local ablation;
- failures and negative results.

Do not collapse everything into a single vanity score.

The confirmatory report contains one quality non-inferiority endpoint and, conditional on its passage, one input-token superiority endpoint. Correction rate is secondary/descriptive. Other selectors, heterogeneity tests, retrieval metrics, latency/cost, feature analyses, alternate weights, and non-gate ablations are exploratory. Per-family effects are descriptive, not eight confirmatory tests.

### Task 11: Run an end-to-end deterministic pilot

**Objective:** Verify experimental integrity before calling a real model.

**Steps:**
1. Validate the tiny dataset.
2. Run all selector modes with the deterministic provider.
3. Confirm no held-out leakage.
4. Confirm reports can be regenerated entirely from the SQLite event log.
5. Run the ID-renaming ablation.
6. Run the full repository tests.

**Commands:**

```bash
.venv/bin/python -m pytest tests/adaptive_selection -v
.venv/bin/python -m pytest -q
```

**Exit criterion:** The harness detects a deliberately introduced adaptive advantage and a deliberately introduced false advantage/leakage defect.

### Task 12: Author and review the full DevOps corpus

**Objective:** Build 64 realistic cases without using held-out results to tune the selector.

**Files:**
- Create: `experiments/adaptive_selection/datasets/devops_v1.json`
- Create: `docs/research/adaptive-context-evaluation.md`

Use provenance-group splitting, duplicate/near-duplicate detection, and independent review:

1. Authoring protection: freeze the feature ontology and adaptive policy before held-out authoring, or assign separate held-out authors; record which approach was used.
2. Independent domain review, without selector results: correctness, realistic failure mode, required evidence, distractors, rubrics, and unsafe recommendations.
3. Experimental review: no provenance/template leakage, no duplicate or near-duplicate content, balanced distractors, distinct held-out IDs, and comparable difficulty across families.

Hash and seal prompts, candidate pools, labels, rubrics, scoring rules, provenance-group splits, and the finalized corpus. Changes after freeze create `devops_v2.json`; do not silently edit v1.

### Task 13: Run the frozen-model pilot

**Objective:** Compare all modes with a real model under identical conditions.

**Steps:**
1. Pin the provider/model and record the complete manifest.
2. Select exactly one primary static policy on development/adaptation data using the predefined criterion and tie-breaker; retain proof of comparable tuning opportunity.
3. Seal code/config, policy, corpus components, splits, rubrics, and scoring rules before model execution.
4. Run matched adaptive and locked-static repetitions in randomized/interleaved order, stateless except for explicit adaptive state, resetting that state between families.
5. Run references and other baselines only as secondary/exploratory conditions.
6. Score randomized opaque outputs with scorers blinded to mode, traces, filenames, run order, and labels; use two humans as preregistered.
7. Run the family-clustered/hierarchical sequential analysis and only predeclared gate ablations.
8. Record negative and ambiguous findings without rewriting the gate.

### Task 14: Make the continuation decision

**Objective:** Decide what, if anything, should be built next.

Possible outcomes:

- **Continue:** repeat the experiment with another domain and a larger corpus.
- **Narrow:** adaptive selection works only for structured recurring tasks; focus there.
- **Simplify:** static metadata/rules perform as well; productize those rather than claiming learning.
- **Redirect:** provenance/governance or evaluation is more valuable than adaptive selection.
- **Stop:** no reliable held-out benefit.

Only after a repeatable positive result should the team decide whether to:

1. promote the experimental modules into `ai_context_manager`;
2. extract a clean `adaptive-context-runtime` repository; or
3. keep the work as a research harness.

---

## 6. Validation matrix

| Layer | Proof required |
|---|---|
| Schema | Versioned records round-trip and reject invalid/leaky fixtures |
| Persistence | Raw events are append-only and can rebuild derived utility |
| Selection | Same eligible candidates and budget across modes |
| Learning | Held-out IDs benefit from shared features, not hidden ID leakage |
| Scoring | Technical failures override retrieval success |
| Reproducibility | Provider, prompt, corpus, policy, and code versions are recorded |
| Comparison | Eight family-trajectory effects, equal-family estimands, coarse clustered/hierarchical uncertainty, and mandatory descriptive family summaries are reported |
| Scientific integrity | Draft decisions and sensitivity/power simulation are resolved before freeze; gates, corpus, and scoring are sealed before model execution |
| Operational integrity | Existing repository tests continue to pass |

---

## 7. Risks and mitigations

### Feedback ambiguity

Mitigation: use typed signals (`factual_correction`, `preference`, `omission`, `irrelevant_context`, `unsafe_action`, `accepted`, `task_success`) instead of a single score. Keep raw events separate from promoted experience.

### Credit assignment

Mitigation: begin with oracle labels; later use paired/leave-one-item-out tests for high-impact items. Never infer that all selected context caused a successful result.

### Baseline sandbagging

Mitigation: lock exactly one strong static rules + metadata policy using development/adaptation data, a predefined criterion, and deterministic tie-breaker. Keep budgets, candidate pools, metadata access, and tuning opportunity comparable.

### Overfitting and memorization

Mitigation: held-out cases use new component IDs; include feature-only and ID-local ablations; report unseen-family performance separately in later experiments.

### Model nondeterminism and drift

Mitigation: freeze configuration, store provider revision/time and manifests/hashes, use five matched repetitions (minimum three), randomize/interleave primary modes statelessly with uncontrolled memory/caches disabled, predeclare aggregation, and do not combine incompatible model versions. Repetitions do not increase the eight-unit sample size.

### Corpus author bias

Mitigation: split by scenario-template/provenance group, detect duplicates/near duplicates, freeze ontology/policy before held-out authoring or use separate authors, require independent domain review without selector results, seal every corpus/scoring component, and include misleading as well as irrelevant context.

### Premature product architecture

Mitigation: keep the harness isolated under `experiments/`; no UI, MCP, enterprise, or production-execution work during the pilot.

### Existing repository claims

Mitigation: revise “enterprise-grade,” “proving feedback,” and unsupported performance language before treating the repository as a public research artifact.

---

## 8. Open questions to settle before protocol freeze

1. Which model/provider should be frozen for the first real trial?
2. Will primary modes use three or five repetitions, and what exact matching/interleaving/aggregation applies?
3. What frozen rubric scale/weights, human-scoring subset, agreement/adjudication rule, and deterministic/human blend apply? What criterion, if any, validates an LLM judge?
4. What family-level variance/effect scenarios and clustered/hierarchical method will the preregistered sensitivity/power simulation use, and what can eight units detect?
5. Does quality non-inferiority require a conservative uncertainty bound or use a heuristic continuation rule, and is `0.03` justified after simulation?
6. What is the one input-token superiority threshold? Correction rate cannot substitute.
7. What static candidate set, development criterion, deterministic tie-breaker, and comparable tuning budget lock the one primary baseline?
8. What severe/critical taxonomy and zero-tolerance stop triggers apply, and is ID-renaming retained in the confirmatory gate?
9. What fixed within-family curricula, family-order procedure, and state-reset checks apply?
10. Will ontology/policy freeze precede held-out authoring or will separate authors be used, and how are provenance grouping, near-duplicate detection, and independent review implemented?
11. Which proposed task families should be replaced before authoring, and which artifacts may be committed versus retained securely?
12. After the controlled pilot, which bounded dogfood workflow and separate natural-feedback protocol should be considered?

---

## 9. Immediate next step

Resolve protocol 1.1-draft and run its preregistered sensitivity/power simulation—not the implementation architecture. Only after the charter's open decisions, one-baseline lock, endpoints, corpus controls, scoring, safety triggers, and sequential gate are frozen may the corpus be sealed or model calls begin.
