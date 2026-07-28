# Adaptive Context Proof Experiment Implementation Plan

> **For Hermes:** Use the `subagent-driven-development` skill to implement this plan task-by-task. Do not begin implementation until the experiment charter and continuation gates are approved.

**Goal:** Determine whether feedback-informed context selection improves recurring-task outcomes on held-out cases compared with strong static selection, without meaningful quality loss.

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

### Primary hypothesis

For recurring infrastructure-analysis tasks, feedback-informed context ranking will outperform the strongest static selector on held-out cases from known task families under the same token budget.

### Secondary hypotheses

1. Adaptive selection reduces irrelevant context and repeated corrections.
2. Adaptive selection can reduce input tokens without materially reducing task quality.
3. Learned utility transfers to unseen cases and unseen context-item IDs within a known task family.
4. Feature-level learning performs better than memorizing component IDs.

### Falsification conditions

The thesis is weakened or rejected for this design if any of the following persist after correcting implementation defects:

- Adaptive selection degrades held-out quality by more than the predeclared tolerance.
- Gains disappear when component IDs are replaced on held-out cases.
- Static metadata/rules account for essentially all improvement.
- Feedback helps adaptation cases but not held-out cases.
- Improvement is concentrated in one task family rather than recurring across families.
- Natural feedback is too sparse or ambiguous to produce stable utility estimates.
- Results change direction across repeated runs with the frozen model configuration.

### Provisional continuation gate

Lock the final gate before viewing held-out results. A reasonable pilot gate is:

- Mean adaptive quality is no worse than 0.03 below the best static baseline.
- Adaptive selection improves either correction rate or input-token use by at least 20% relative to the best static baseline.
- There is no increase in critical/safety-related failures.
- Improvement is visible in at least five of eight task families.
- A component-ID ablation shows that the result is not primarily memorization.

Treat this as a **continue-investigating** threshold, not proof of a company or broad organizational learning.

---

## 3. Experimental design

### Corpus

Create eight recurring task families with eight cases each: 64 total cases, split within each family into five ordered adaptation cases and three held-out cases (40 adaptation, 24 held out).

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

### Execution modes

All comparable selectors receive the same task, candidates, token budget, model, system prompt, and generation settings.

1. **Reference: full eligible context** — establishes the high-token reference and reveals attention dilution; not the primary fairness baseline.
2. **Baseline A: static explainable selector** — existing query relevance, lifecycle filters, metadata, recency, and fixed importance weights.
3. **Baseline B: similarity/top-K** — semantic or deterministic fixture similarity under the same token budget.
4. **Baseline C: strong static rules + metadata** — the strongest non-learning selector; this is the primary baseline to beat.
5. **Candidate D: adaptive selector** — the same static features plus utility learned only from prior adaptation feedback.

Do not give the adaptive selector richer metadata or a larger budget than the baselines.

### Staged learning

- **Stage 1: oracle feedback** — use corpus relevance/outcome labels to verify the learning and evaluation plumbing.
- **Stage 2: simulated behavioral feedback** — emit correction/acceptance signals from deterministic rubric outcomes.
- **Stage 3: model execution** — run a frozen model and score outputs blind to selector mode.
- **Stage 4: naturalistic dogfood** — use real Andrew feedback in a bounded `bourbon_research` or infrastructure workflow.

A stage must pass its integrity checks before moving to the next.

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

### Task 1: Write and approve the experiment charter

**Objective:** Freeze the hypothesis, primary metric, baselines, corpus split, continuation gate, and falsification criteria before implementation can bias the design.

**Files:**
- Create: `docs/research/adaptive-context-charter.md`
- Create: `docs/decisions/0001-use-existing-repo-for-pilot.md`

**Steps:**
1. Copy the charter in Sections 2–3 into the project document.
2. Resolve the open questions in Section 8 below.
3. Mark the charter `status: preregistered` with a dataset version.
4. Commit only the charter and ADR.

**Verification:** A reviewer can identify the primary hypothesis, strongest baseline, held-out set, continuation gate, and stopping conditions without reading code.

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

**Verification:** The scorer penalizes an unsafe or technically wrong answer even if it retrieved nominally relevant context.

### Task 8: Add frozen model adapters and run manifests

**Objective:** Prevent model drift and configuration differences from masquerading as learning.

**Files:**
- Create: `experiments/adaptive_selection/providers.py`
- Test: `tests/adaptive_selection/test_runner.py`

Implement a provider protocol and a deterministic fake provider first. A real provider adapter must record provider, exact model identifier, prompt/template hash, temperature, seed where supported, tool availability, timestamps, token accounting, and raw response hash.

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

The adaptive run may learn only from feedback revealed after each adaptation case. Held-out feedback remains sealed until selection and generation are complete.

**Verification:** A leakage test deliberately places the answer in held-out feedback and confirms it cannot influence selection.

### Task 10: Produce paired reports and uncertainty estimates

**Objective:** Report whether the adaptive mode beat the strongest baseline and why.

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
- paired differences;
- bootstrap confidence intervals;
- feature-level versus ID-local ablation;
- failures and negative results.

Do not collapse everything into a single vanity score.

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

Use a two-pass review:

1. Domain review: correctness, realistic failure mode, required evidence, and unsafe recommendations.
2. Experimental review: no leakage, balanced distractors, distinct held-out IDs, and comparable difficulty across families.

Hash and version the finalized corpus. Changes after preregistration create `devops_v2.json`; do not silently edit v1.

### Task 13: Run the frozen-model pilot

**Objective:** Compare all modes with a real model under identical conditions.

**Steps:**
1. Pin the provider/model and record the complete manifest.
2. Run the full-context reference and all three baselines.
3. Identify the strongest static baseline on the adaptation/validation data only.
4. Run the adaptive mode in ordered adaptation.
5. Seal code/config before evaluating held-out outputs.
6. Score outputs blinded to selector mode where human judgment is required.
7. Run paired analysis and all ablations.
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
| Comparison | Paired results, per-family effects, uncertainty, and ablations are reported |
| Scientific integrity | Gates are fixed before held-out evaluation |
| Operational integrity | Existing repository tests continue to pass |

---

## 7. Risks and mitigations

### Feedback ambiguity

Mitigation: use typed signals (`factual_correction`, `preference`, `omission`, `irrelevant_context`, `unsafe_action`, `accepted`, `task_success`) instead of a single score. Keep raw events separate from promoted experience.

### Credit assignment

Mitigation: begin with oracle labels; later use paired/leave-one-item-out tests for high-impact items. Never infer that all selected context caused a successful result.

### Baseline sandbagging

Mitigation: designate strong static rules + metadata as the primary baseline. Keep budgets and candidate pools identical.

### Overfitting and memorization

Mitigation: held-out cases use new component IDs; include feature-only and ID-local ablations; report unseen-family performance separately in later experiments.

### Model nondeterminism and drift

Mitigation: freeze configuration, store manifests and hashes, use repeated runs, and do not combine incompatible model versions.

### Corpus author bias

Mitigation: separate authoring and review, preregister the dataset, and include misleading as well as irrelevant context.

### Premature product architecture

Mitigation: keep the harness isolated under `experiments/`; no UI, MCP, enterprise, or production-execution work during the pilot.

### Existing repository claims

Mitigation: revise “enterprise-grade,” “proving feedback,” and unsupported performance language before treating the repository as a public research artifact.

---

## 8. Open questions to settle before implementation

1. Which model/provider should be frozen for the first real trial?
2. Should quality scoring be fully deterministic, human-reviewed, model-judged, or a predeclared combination?
3. Is a 20% relative efficiency/correction improvement the right continuation threshold?
4. Which of the proposed eight DevOps task families, if any, should be replaced before corpus authoring?
5. After the controlled pilot, which bounded `bourbon_research` workflow should be used for the first naturalistic dogfood comparison?
6. What feedback categories should require explicit human confirmation before becoming reusable experience?
7. Which experiment artifacts may be committed, and which may contain sensitive prompts or provider outputs and must remain local?

---

## 9. Immediate next step

Review and approve the experiment charter—not the implementation architecture. Once the hypothesis, corpus, primary metric, strongest baseline, continuation gate, and stopping conditions are fixed, execute Tasks 1–4 before adding any model calls.
