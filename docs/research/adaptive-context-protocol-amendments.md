# Adaptive Context Protocol Amendment Record

This append-only record accompanies the [adaptive context charter](adaptive-context-charter.md), [ADR 0003](../decisions/0003-strong-static-selection-is-primary-baseline.md), and the [implementation plan](../../.hermes/plans/2026-07-28_170432-adaptive-context-proof-plan.md). Git history preserves every superseded text; an amendment never silently rewrites the scientific record.

## 2026-07-28 — Protocol 1.0 to 1.1-draft

- **Accountable author:** Andrew (Drew) Hamilton, project owner
- **Drafting and implementation support:** Hermes Agent
- **Review:** independent delegated scientific-methods and documentation-quality reviewers
- **Status:** prospective draft amendment
- **Timing:** made before corpus authoring or sealing, model execution, or access to any adaptation or held-out outcomes
- **Effect:** protocol 1.1-draft prospectively supersedes 1.0 for future work; protocol 1.0 remains audit-visible in commit history. Confirmatory execution is prohibited until all open decisions and the sensitivity/power simulation are resolved and a later version is frozen.

### Methodological corrections

1. **Top-level unit and estimand.** Changed primary inference from task-case executions to eight task-family learning trajectories. Held-out cases are clustered within family and generations are nested measurements. Defined the exact equal-family-weighted paired adaptive-minus-locked-static quality estimand after five adaptation cases. Replaced independent 24-case bootstrapping with family-clustered/hierarchical analysis and mandatory raw family effects, median/range, and coarse intervals. **Why:** treating nested cases or generations as independent would create pseudoreplication and overstate precision.
2. **Sequential confirmatory decision.** Replaced an omnibus quality-superiority hypothesis and an “either corrections or tokens” gate with quality non-inferiority first, followed only on passage by superiority for one efficiency endpoint, input-context tokens. Correction rate and quality superiority are secondary. **Why:** the prior alternative efficiency outcomes allowed outcome shopping and did not match the practical claim of preserving quality while reducing context.
3. **Provisional quality margin and decision rule.** Kept `0.03` only as a provisional margin pending frozen rubric scaling/weights and sensitivity simulation. Made the choice between a conservative uncertainty-bound gate and a heuristic continuation rule explicit and unresolved. **Why:** a numeric margin has no defensible interpretation before the measurement scale and achievable precision are established.
4. **One fair primary baseline.** Required exactly one static policy, selected only from development/adaptation information using a predefined candidate set, criterion, and deterministic tie-breaker. Required comparable metadata access and tuning effort for static and adaptive modes; all other baselines are secondary/exploratory. **Why:** selecting the “best” baseline after outcomes or underinvesting in static tuning would bias the comparison.
5. **Repetitions and execution order.** Recommended five matched repetitions for the two primary modes, with three as a cost-constrained minimum and the final count open before freeze. Required randomized/interleaved, stateless execution, disabled uncontrolled memory/caches, provider revision/time capture, and predeclared aggregation. **Why:** repetitions reduce measurement noise but do not create additional independent families; blocked execution can confound condition with provider drift or carry-over.
6. **Learning order and state.** Required a fixed predeclared within-family adaptation order, claims restricted to that curriculum, learned-state reset between families, and recorded family execution order. **Why:** order and cross-family carry-over otherwise alter the treatment and undermine trajectory independence.
7. **Leakage and corpus controls.** Required splitting by scenario-template/provenance group, duplicate and near-duplicate detection with content hashes, ontology/policy freeze before held-out authoring or separate authors, independent domain review without selector results, and hashes/seals for prompts, pools, labels, rubrics, scoring rules, and splits. **Why:** new IDs alone do not prevent semantic/template leakage or authoring-to-policy leakage.
8. **Safety guardrail.** Replaced statistical “no increase” language with a zero-tolerance engineering guardrail, manual review of every severe/critical event, preregistered stop triggers, and reporting of counts and paired cases without safety-equivalence claims. **Why:** eight trajectories cannot support a credible statistical equivalence claim for rare severe events.
9. **Scoring bias and validity.** Expanded blinding to mode, traces, filenames, run order, and labels; randomized presentation; limited rubric calibration to development cases; required two independent human scorers for a substantial stratified subset and all critical judgments; required agreement/adjudication reporting. LLM judges remain exploratory unless validated against blinded humans, and deterministic scorers must pass paraphrase tests. **Why:** condition cues, unvalidated judges, and keyword-only checks can manufacture apparent effects.
10. **Confirmatory boundary.** Narrowed confirmation to one adaptive-versus-locked-static contrast, one quality endpoint, one conditional token endpoint, and ID-renaming only if retained in the gate. Classified other selectors, heterogeneity tests, retrieval metrics, latency/cost, feature analyses, alternative weights, and other ablations as exploratory. Per-family effects remain mandatory descriptive reports, not eight tests. **Why:** explicit multiplicity and scope control prevents post-hoc promotion of favorable results.
11. **Feedback comparability.** Required selector-independent locked feedback in oracle/simulated stages and labeled natural/model-human feedback as a policy-dependent online trajectory. **Why:** behavior-dependent feedback changes the data-generating process and does not estimate the same effect as fixed labels.
12. **Negative controls.** Added no-effect controls, misleading shared-feature controls, feature perturbation, template/provenance ablations, leakage probes, and conditional ID-renaming. **Why:** a positive harness must also detect no-effect, harmful-transfer, memorization, and leakage mechanisms.
13. **Stage-specific claims.** Defined what tiny deterministic, oracle, simulated, frozen-model synthetic, and dogfood stages can and cannot establish. **Why:** plumbing tests and constructed signals cannot support empirical, causal, production, or safety claims.
14. **Sensitivity/power simulation.** Added a preregistered pre-freeze simulation of family-level effects and nested noise, comparing three/five repetitions and candidate clustered/hierarchical rules. Conventional p-value framing is explicitly discouraged. **Why:** with only eight top-level units, the design must expose detectable effect sizes and limitations before selecting thresholds.
15. **Status and audit semantics.** Changed status from preregistered to draft preregistration and version to 1.1-draft; linked the charter, amendment, ADR, and plan. **Why:** unresolved design choices and simulation prevent a truthful claim that the protocol is fully frozen.

### Expected analytical impact

The amendment reduces nominal precision, narrows the confirmatory claim, and may make continuation harder. It also makes the resulting claim identifiable: an equal-family average under a fixed five-case curriculum, compared with one prospectively locked static policy. Repeated generations support more precise family measurements but not a larger independent sample. The simulation may require a larger number of families, a narrower descriptive claim, revised thresholds, or stopping before model execution.

### Outcome access declaration

At the time of this amendment, no corpus had been authored or sealed, no model condition had run, and no adaptation or held-out outcome existed or had been accessed. The corrections therefore do not respond to observed effects and cannot favor a known result.

## 2026-07-30 — Separate Terra development-probe protocol

- **Accountable author:** Andrew (Drew) Hamilton, project owner
- **Drafting and implementation support:** Hermes Agent
- **Status:** prospective development-only protocol
- **Timing:** committed before any Terra request, bridge-corpus authoring or sealing, model condition, adaptation output, held-out output, or comparative outcome existed
- **Related protocol:** [Terra Bridge Development Protocol](terra-bridge-development-protocol.md)
- **Effect on protocol 1.1-draft:** none. The parent charter remains draft and confirmatory execution remains prohibited.

### Scope and rationale

Andrew prospectively selected OpenAI `gpt-5.6-terra` as the intended bridge model based on the planning assumption that its published price and model-tier positioning offer an acceptable screening tradeoff. This is an owner decision, not an experimental finding about comparative capability, null interpretability, or economics. Before corpus work, the project requires one benign API/configuration probe to determine whether the selected endpoint can be represented honestly by the existing provider-evidence boundary.

The separate development protocol authorizes exactly one attempted non-domain Responses API network request under a prospectively committed JSON contract. The first attempted request consumes the sole authority regardless of outcome; SDK retries, fallback, replacement requests, and pre-authorized amended attempts are prohibited. The contract freezes one literal API request body, client retry/timeout settings, the requested model alias, medium reasoning, omitted temperature/top-p/seed, strict structured output, disabled tools and state, `store=false`, output cap, standard service tier, pricing, projected-cost guard, decoded HTTP-body capture, raw usage validation, both request and response IDs, sanitized failure evidence, and JSON `null` for the absence of a separately reported provider revision. Nullable revision and temperature schema migrations are required before the live request.

The probe cannot establish model quality, context sensitivity, adaptive efficacy, transfer, safety, production value, or favorable economics. Passing permits only drafting—not executing—a separate development-only context-sensitivity calibration proposal. Any calibration call, second Terra probe, or proposed 99-call bridge requires a wholly new, explicitly owner-approved contract; all remain prohibited here.

### Outcome access declaration

At this amendment's freeze, no Terra API request had been made for this project, no hosted-model result existed, and no bridge corpus had been authored or sealed. The probe prompt contains no DevOps task, selector condition, feedback, labels, required-context set, answer key, or held-out material.
