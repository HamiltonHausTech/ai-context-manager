# Task 12b v2 offline execution readiness

**Status:** preparation and verification only; **0 provider calls authorized**.

## Boundary

This layer prepares the prospective 45-unit repeated-draw replication described in `task12b-context-sensitivity-replication-v2.md`. It does not retry, reset, score, reveal, or reinterpret the consumed invalid v1 run. The six v1 successes and the v1 blind mapping remain sealed and unscored.

The historical v1 evidence remains bound to executed revision `c1430801720f912d74bf180b521a075200252234`. Parser remediation is later (`e98c503`), and the v2 scientific contract is based at `edf09bc04a79e00934395d7fe21fd6b17612bc10`. A future readiness revision and any separately authorized execution revision are distinct identities; current-head code must never be substituted when verifying historical records.

The v2 orchestration module imports only the repaired v1 `dispatch_once` transport/parser and its result/failure types. Candidate, approval, namespace, claims, terminals, reconciliation, randomization, blind export, annotation locking, and orphan handling are v2-specific.

## Zero-authority readiness manifest

`experiments/adaptive_selection/controls/task12b_execution_readiness_v2.json` freezes:

Raw SHA-256: `d0b86651b0d33282222738821a816dc01b218f5b3e7e87f6d9a44a8b836e7afc`.

- `network_requests_authorized_by_manifest = 0`;
- 45 scheduled units and five balanced nine-unit blocks;
- 45 cryptographically unrelated assessment aliases and an independent assessment order;
- candidate non-authority and separate exact-digest owner approval;
- model/SDK settings, conservative ceiling `USD 1.470820`, and hard cap `USD 2.00`;
- the immutable v1 contract and renderer hashes.

Candidate preparation writes only a private randomization mapping and a non-authorizing candidate. It never writes owner approval. Candidate creation may receive an already-derived credential fingerprint, but does not import the SDK, read a credential, or access a provider.

## Randomization and authority

Candidate-time randomization creates five randomized complete blocks. Each block contains exactly one draw from every base cell. Block order and within-block order are randomized. Assessment aliases are unique `assessment-<32 lowercase hex>` values. The 45-item assessment order is shuffled independently and must differ from execution order. The candidate hash-binds the full private mapping plus separate execution-order and assessment-mapping commitments.

The machine-global namespace digest is derived solely from the v2 contract lineage. Its parent home is selected from the stable OS account identity obtained with `pwd.getpwuid(os.getuid())`, so a forged `HOME` has no effect. V1 and v2 roots are disjoint. Production has no root override. Test-only functions accept injected roots.

Every authority record is keyed by canonical `unit_id`, not request hash. Five units intentionally share each request hash. Claims and terminals bind unit, base cell, draw, request hash, lineage, candidate, approval, revision, host, account, credential, and mapping commitment. Publication is owner-private and no-overwrite. Machine-global records are authoritative; local records are exact mirrors. Missing, partial, conflicting, terminal-only, raw-only, or orphan states fail closed. A nonblocking machine-global run lock serializes reconciliation and execution across checkouts.

## Explicit execution path

The default CLI is an offline dry run. The CLI exposes explicit candidate preparation, historical authority verification, blind export, annotation lock, scoring, orphan finalization, and `--execute-authorized-45-unit-manifest` routes. It never writes owner approval. Candidate and execution routes require a clean `main` exactly at the already-fetched `origin/main`; all candidate identities bind stable host/account/credential fingerprints. Execution reuses the reviewed v1 paid ambient guard, official client construction, and repaired transport/parser. Offline post-execution commands accept the already-derived credential fingerprint and do not read a credential or contact a provider.

For each pending unit it:

1. checks live expiry and remaining cap before claim;
2. publishes one unit-keyed claim;
3. invokes the shared repaired v1 transport at most once;
4. charges validated conservative usage or the static scheduled upper bound for every success, failure, or ambiguity;
5. publishes one terminal and continues to later scheduled units when safe.

Terminal units are never dispatched again. An orphan claim blocks ordinary resume. Offline orphan finalization requires a separately authored owner record that echoes the exact orphan claim digest and confirms no process remains; it records an `invalid_ambiguous` terminal and grants no replacement draw.

No provider-visible request contains unit, draw, condition, family, alias, block, or order metadata. Requests are rendered by the frozen v1 renderer and remain byte-identical to the nine v1 request bodies.

## Blind assessment and scoring

Blind export follows the independently randomized assessment order and includes exactly the alias, task, provider-visible timestamped evidence, success/failure status, structured response, inherited condition-independent rubric criteria, adjudication rules, and critical-finding definitions. It verifies historical owner approval plus exact global/local reconciliation and hardened terminal semantics first. It omits canonical IDs, family/condition, draw/block/order, provider IDs, hashes, controller paths, execution timestamps/latency, and authority metadata.

Annotations use aliases. The controller validates their exact shape, publishes an immutable no-overwrite lock, and binds its canonical SHA-256 before resolving aliases. Mapping substitution after lock fails because both lock and candidate bind the private mapping commitment. Resolved records are passed to the existing v2 scientific scorer only after lock verification. Missing/failed units receive no fabricated annotations; incomplete annotation for any schema-valid success produces no scientific verdict.

## Offline usage

```bash
python -m experiments.adaptive_selection.context_sensitivity_execution_v2
```

Expected safe summary:

```text
task12b-v2 status=dry-run network_attempts=0 credential_readiness=not_checked scheduled_units=45
```

The dry-run summary contains only lineage hashes, counts, and budget values—never task, evidence, rubric, response, mapping, or sealed-output content.

## Authorization statement

Merging this readiness layer would leave execution at **0/45 authorized calls**. Pricing verification, exact-head review/CI, candidate preparation permission, an expiring candidate, and a separately authored exact candidate-digest owner echo are all future gates. Generic approval, merge permission, or candidate preparation is not execution authority.
