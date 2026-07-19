# AI Context Manager Modernization Plan

## Product direction

Reframe the project from context-window compression to **agent memory curation**:
durable structured memory, explainable retrieval, and deliberate assembly of the
smallest useful context for a task.

Larger model context windows reduce the urgency of truncation, but they do not
solve relevance, persistence, attention dilution, latency, cost, or memory
quality. Those are the project's durable problems.

## Baseline (2026-07-19)

- Environment: Python 3.11.15 in `.venv`
- Package installed editable with base dependencies
- Configured suite: 11 passing tests
- Additional top-level pytest module: 1 passing test
- Measured coverage: 33% across `ai_context_manager`
- Standalone `test_runner.py`: fails on a fresh clone because root `config.toml`
  is absent and intentionally ignored
- Clean offline import: fails if tiktoken's `cl100k_base` data has not already
  been downloaded; tokenizer initialization currently happens at import time
- Optional ChromaDB, PostgreSQL, Ollama, and live OpenAI integrations were not
  exercised in this baseline

## Principles

1. Structured records are the source of truth; rendered prompt text is a view.
2. Retrieval behavior must be explicit, deterministic, and inspectable.
3. Persistence must round-trip every supported component without loss.
4. Optional integrations must not break the core package or offline operation.
5. Backends implement shared behavioral contracts.
6. Summarization consolidates knowledge; it is not only an overflow fallback.

## Phase 1: Establish trustworthy contracts

Target: a small, dependable core with tests that describe intended behavior.

- Define a versioned serialized component schema.
- Add `to_dict`/`from_dict` behavior for every component.
- Preserve component-specific fields, scores, timestamps, and identifiers.
- Specify tag matching as an explicit mode (`all` or `any`) rather than an
  implicit behavior.
- Define duplicate registration behavior: reject, replace, or upsert.
- Define retrieval ordering and tie-breaking.
- Define exact token-budget guarantees, including a zero remaining budget.
- Decide whether `get_task_context(task_id)` filters by component ID, a task
  relationship field, or both; then implement and test it.
- Replace print-based dry runs with returned trace data or a supplied output
  callback.
- Make tokenizer loading lazy and preserve an offline fallback for download or
  initialization failures.

Exit criteria:

- JSON persistence round-trips all component types without semantic data loss.
- Core retrieval tests cover filtering, ranking, budgeting, summarization, empty
  inputs, duplicates, and task isolation.
- Importing the base package performs no network access.
- Core tests run from a fresh clone without a manually created config file.

## Phase 2: Separate policy from storage

Target: context assembly behavior that can evolve independently of persistence.

- Introduce a retrieval request containing query, filters, limit, budget, and
  policy options.
- Introduce a retrieval result containing selected records, scores, exclusions,
  token counts, and reasons.
- Split candidate retrieval, ranking, budget packing, rendering, and
  summarization into independently testable stages.
- Replace broad exception swallowing with typed failures and explicit degraded
  modes.
- Add backend contract tests and run the same suite against in-memory, JSON, and
  SQLite stores.
- Add atomic JSON writes and document concurrency expectations.

Exit criteria:

- A caller can explain why each memory was included or excluded.
- JSON and SQLite pass the same storage contract suite.
- Storage failures cannot silently masquerade as an empty memory set.

## Phase 2 completion (2026-07-19)

- Added typed `RetrievalRequest`, `RetrievalResult`, item, and decision models.
- Split retrieval into independently callable candidate selection, ranking, and
  budget-packing stages.
- Added inclusion and exclusion reasons for filters, budget decisions,
  summarization, and processing failures.
- Kept `ContextManager.get_context()` as a compatibility wrapper and added
  `ContextManager.retrieve()` for explainable retrieval.
- Added typed storage read/write errors and removed empty-result fallbacks from
  memory backend operations.
- Made JSON writes atomic with flush, fsync, and same-directory replacement.
- Added transactional `SQLiteMemoryStore` support and configuration loading.
- Added shared create/read/update/delete and manager round-trip contract tests
  for JSON and SQLite.
- Documented concurrency expectations for local and external backends.
- Expanded the verified suite to 45 passing tests and overall coverage to 43%;
  the retrieval pipeline itself has 91% coverage.

## Phase 3: Real hybrid retrieval

Target: useful retrieval based on meaning and structured state.

- Define an embedding provider interface with model name, dimension, and version
  recorded alongside embeddings.
- Replace PostgreSQL random vectors with a real provider.
- Use stable content hashing; do not use Python's process-randomized `hash()` for
  persisted behavior.
- Combine semantic similarity with metadata filters, recency, importance,
  component feedback, and task relationships.
- Normalize score direction and scale across ChromaDB and pgvector.
- Define re-embedding and migration behavior when models change.
- Make degraded retrieval visible when embeddings are unavailable.

Exit criteria:

- A relevance fixture produces comparable ordering across supported vector
  backends.
- PostgreSQL semantic retrieval uses real embeddings and passes integration
  tests.
- Embedding model changes are detectable and recoverable.

## Phase 3 completion (2026-07-19)

- Added a provider-independent embedding interface with strict dimension
  validation and lazy sentence-transformer loading.
- Removed PostgreSQL's randomized placeholder vectors; both PostgreSQL and
  ChromaDB now use injected real embedding providers.
- Standardized searchable-text construction across vector backends.
- Added stable SHA-256 content hashes plus provider, model, version, and
  dimension metadata to every embedded record.
- Added stale detection and selective `reembed_all()` migration support.
- Added backend-independent hybrid ranking across semantic similarity,
  importance, recency, and feedback, normalized to `[0, 1]`.
- Added explicit semantic availability/degradation reporting and removed silent
  zero-vector/empty-result fallbacks.
- Fixed PostgreSQL connection setup to consistently use resolved environment or
  argument values.
- Added offline relevance fixtures and live integration tests for ChromaDB and
  PostgreSQL/pgvector using the same deterministic embedding provider.
- Verified real pgvector schema creation, persistence, cosine search, provenance,
  hybrid ranking, and full provider-version re-embedding.
- Expanded the complete verified suite to 58 passing tests and overall coverage
  to 53% when both live vector integrations are enabled.

## Phase 4: Memory consolidation and evaluation

Target: improve memory quality over time rather than accumulating raw history.

- Distinguish episodic events, durable facts, user preferences, goals, and
  derived summaries.
- Add provenance and links from derived memories to source records.
- Add consolidation policies for merging, superseding, and expiring memories.
- Preserve contradictory information instead of silently overwriting it.
- Build an evaluation dataset containing retrieval queries, relevant memories,
  distractors, and expected exclusions.
- Track retrieval precision/recall, token efficiency, context stability, and
  downstream task quality.

Exit criteria:

- Consolidated memories retain traceable provenance.
- Retrieval changes can be compared against a repeatable evaluation set.
- Feedback has a measured benefit rather than only an intuitive scoring role.

## Phase 4 completion (2026-07-19)

- Added explicit episode, durable-fact, preference, goal, derived-summary, and
  generic memory categories.
- Added persistent lifecycle metadata for provenance, supersession,
  contradictions, expiry, and confidence across JSON, ChromaDB, and PostgreSQL.
- Added deterministic derive, merge, expire, contradiction-recording, and
  contradiction-resolution policies.
- Preserved contradictory claims until an explicit resolution supersedes one.
- Made traditional and semantic retrieval exclude inactive memories by default,
  with explainable lifecycle decisions and an audit override.
- Added a versioned retrieval evaluation dataset containing relevant memories,
  distractors, and graded relevance.
- Added precision, recall, reciprocal-rank, NDCG, token-efficiency,
  expected-exclusion, context-stability, and downstream-utility metrics.
- Added an application-supplied downstream utility scorer with NDCG as the
  default proxy.
- Added a controlled comparison proving that feedback improves reciprocal rank
  and measured utility on the evaluation fixture.
- Added an offline consolidation example suitable for later integration into
  the web demo.
- Expanded the complete live-backend suite to 68 passing tests and overall
  coverage to 57%; consolidation and evaluation coverage are 94% and 99%.

## Phase 5: Package and operational hardening

Target: align public claims with supported behavior.

- Narrow and bound dependency versions according to compatibility policy.
- Move provider-specific dependencies behind optional extras.
- Add a development extra containing pytest, coverage, formatting, linting, and
  typing tools.
- Add the declared `py.typed` marker or remove the package-data declaration.
- Test the actual supported Python version matrix.
- Separate unit, contract, and integration test markers in CI.
- Make security jobs fail on actionable findings instead of unconditionally
  converting failures into successful runs.
- Verify package URLs and remove unsupported “enterprise-grade” language until
  the corresponding reliability criteria are met.
- Provide a checked-in non-secret example config and a fresh-clone quick-start
  test.

## Query-aware retrieval refinement (2026-07-19)

The first real application exercise showed that static confidence ranking can
faithfully fill a token budget while still selecting irrelevant context. The
core retrieval path was therefore extended without embedding application-domain
rules into the library.

- Added an optional task query to `RetrievalRequest`; callers without a query
  retain the previous static-score ordering.
- Added explainable relevance, importance, and recency score factors.
- Added configurable minimum relevance, maximum component count, redundancy
  suppression, and optional required-concept constraints.
- Added query-aware compression instructions so summaries retain information
  useful to the current task rather than merely shortening text.
- Added an application-supplied semantic relevance hook. Applications may use
  embeddings, a reranker, domain relationships, or another scorer without
  replacing candidate selection or budget packing.
- Preserved hard lifecycle, project, and tag boundaries independently of soft
  relevance policy.
- Extended semantic-manager fallback paths so the query reaches core ranking
  even when a vector backend is unavailable.
- Added regression coverage for confidence-versus-relevance conflicts,
  required-concept misses, conceptual matches, redundancy, task-aware
  compression, and legacy compatibility.
- Validated the behavior against deliberately noisy research memory: a
  1,600-token confidence-ranked trace containing beer, sake, absinthe, and
  tequila was reduced to a 234-token task-focused trace with explicit exclusion
  reasons.

The intended product boundary is now explicit: the library supplies stable
retrieval mechanics, explainable signals, presets/hooks in future refinements,
and safe defaults; the consuming application remains the final authority on
domain policy and the acceptable precision/recall balance.

## Recommended first milestone

Deliver Phase 1 as a compatibility-minded `0.3.0` foundation. Avoid adding new
stores or integrations during that milestone. The highest-value implementation
order is:

1. Characterization tests for current behavior.
2. Versioned component serialization and lossless JSON round trips.
3. Explicit filtering, ranking, and token-budget contracts.
4. Lazy offline-safe tokenizer initialization.
5. Self-contained tests and fresh-clone documentation.

This milestone creates the stable seam needed for later retrieval and
consolidation work without committing the project to its current accidental
semantics.

## Phase 1 completion (2026-07-19)

Completed foundation work:

- Added a versioned component schema and lossless round trips for all built-in
  component types.
- Preserved compatibility with legacy rendered-content records.
- Passed structured serialization data through JSON, Chroma, and PostgreSQL
  store representations.
- Made tokenization lazy and offline-safe.
- Added guaranteed token-budget truncation after summarization.
- Preserved `any` tag matching as the default and added explicit `all` matching.
- Made task-scoped retrieval exclude summaries belonging to other tasks.
- Added a checked-in example configuration and made the standalone test runner
  work without a root `config.toml`.
- Added the declared `py.typed` package marker.
- Defined and tested `skip`, `replace`, and `error` duplicate-registration
  policies while retaining `skip` as the compatible default.
- Added serializer/deserializer registration for application-defined component
  types.
- Made dry runs return a structured selection trace while retaining their
  console preview for compatibility.
- Added corrupt-record validation and fallback coverage.
- Expanded the verified suite from 12 to 31 passing tests and coverage from 33%
  to 39%.
