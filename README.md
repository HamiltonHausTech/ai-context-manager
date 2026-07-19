# AI Context Manager

A modular context management system for AI-powered applications with intelligent summarization and feedback learning.

## Features

- **Modular Architecture**: Pluggable components, stores, and summarizers
- **Token-Aware Budgeting**: Intelligent context management with automatic summarization
- **Feedback Learning**: Time-weighted scoring system for component prioritization
- **Multiple Storage Backends**: JSON and SQLite support
- **Privacy-Focused**: Local LLM support via Ollama
- **Flexible Summarization**: OpenAI, Ollama, and naive summarizers

## Quick Start

### 1. Install Dependencies

**Basic Installation:**
```bash
pip install -e .
```

**With ChromaDB Support (Development):**
```bash
pip install -e .[vector]
```

**With PostgreSQL Support (Production):**
```bash
pip install -e .[production]
```

**Full Installation (All Features):**
```bash
pip install -e .[all]
```

### 2. Set Up Environment Variables

Copy `env.example` to `.env` and configure:

```bash
cp env.example .env
```

Edit `.env` with your settings:

```env
# Required for OpenAI summarizer
OPENAI_API_KEY=your_openai_api_key_here

# Optional Ollama configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b-instruct
```

The `bourbon-research` CLI automatically loads the nearest `.env` file. Values
already present in the process environment take precedence, and `.env` is ignored
by Git.

### 3. Configure the System

Create a local configuration from the checked-in, offline-safe example:

```bash
cp config.example.toml config.toml
```

Then edit `config.toml` based on your setup:

**For production agents (PostgreSQL + pgvector):**
```toml
[summarizer]
type = "auto_fallback"  # Tries Ollama, falls back to naive
model = "mistral"

[feedback_store]
type = "sqlite"
db_path = "feedback.db"

[memory_store]
type = "postgres_vector"  # Enterprise-grade vector database
host = "localhost"
port = 5432
database = "ai_context"
user = "postgres"
password = "your_password"
table_name = "agent_memory"
embedding_dimension = 384
index_type = "hnsw"
```

**For development agents (ChromaDB):**
```toml
[summarizer]
type = "auto_fallback"  # Tries Ollama, falls back to naive
model = "mistral"

[feedback_store]
type = "json"
filepath = "feedback.json"

[memory_store]
type = "vector"  # ChromaDB semantic similarity search
collection_name = "agent_memory"
persist_directory = "./chroma_db"
embedding_model = "all-MiniLM-L6-v2"
```

**For automatic fallback (simpler setup):**
```toml
[summarizer]
type = "auto_fallback"  # Tries Ollama, falls back to naive
model = "mistral"

[feedback_store]
type = "json"
filepath = "feedback.json"

[memory_store]
type = "json"
filepath = "memory.json"
```

**For local development (no external dependencies):**
```toml
[summarizer]
type = "naive"  # Simple truncation, works anywhere

[feedback_store]
type = "json"
filepath = "feedback.json"

[memory_store]
type = "json"
filepath = "memory.json"
```

**When on your local network with Ollama:**
```toml
[summarizer]
type = "ollama"
model = "mistral"
# host will be read from OLLAMA_HOST environment variable

[feedback_store]
type = "json"
filepath = "feedback.json"

[memory_store]
type = "json"
filepath = "memory.json"
```

**For OpenAI integration:**
```toml
[summarizer]
type = "openai"
model = "gpt-3.5-turbo"
# api_key will be read from OPENAI_API_KEY environment variable

[feedback_store]
type = "json"
filepath = "feedback.json"

[memory_store]
type = "json"
filepath = "memory.json"
```

### 4. Basic Usage

```python
from ai_context_manager import ContextManager, TaskSummaryComponent
from ai_context_manager.config import Config
from ai_context_manager.utils import load_stores_from_config, load_summarizer

# Load configuration
config = Config("config.toml")
feedback_store, memory_store = load_stores_from_config(config.data)

# Initialize context manager
ctx = ContextManager(
    feedback_store=feedback_store,
    memory_store=memory_store,
    summarizer=load_summarizer(config)
)

# Add a component
task = TaskSummaryComponent(
    id="task-001",
    task_name="Example Task",
    summary="This is an example task summary",
    tags=["example", "demo"]
)

ctx.register_component(task)

# Get context
context = ctx.get_context(
    include_tags=["example"],
    token_budget=500,
    summarize_if_needed=True
)

print(context)
```

## Configuration

### Summarizers

- **auto_fallback**: **Recommended** - Tries Ollama first, falls back to naive (best of both worlds)
- **naive**: Simple truncation (no external dependencies) - **works anywhere**
- **ollama**: Local LLM via Ollama API (requires local Ollama instance)
- **openai**: OpenAI GPT models (requires API key and internet)

### Storage

- **JSON**: File-based storage for simple deployments
- **SQLite**: Transactional local storage for feedback and component memory
- **Vector**: ChromaDB-based semantic similarity search (development)
- **PostgreSQL + pgvector**: Enterprise-grade vector database (production)

### Explainable Retrieval

`get_context()` remains the convenient string/metadata API. Applications that
need selection diagnostics can use the typed retrieval API:

```python
from ai_context_manager import RetrievalRequest

result = ctx.retrieve(RetrievalRequest(
    query="Complete the current deployment task safely",
    required_terms=["deployment"],
    include_tags=["task", "memory"],
    token_budget=500,
    summarize_if_needed=True,
    min_relevance=0.10,
    deduplicate=True,
))

print(result.context)
for decision in result.decisions:
    print(decision.component_id, decision.included, decision.reason)
```

Retrieval runs as separate candidate selection, query-aware ranking, redundancy
control, and budget-packing stages. Query-aware decisions report relevance,
importance, and recency factors. `required_terms` is an optional precision guard;
omit it when conceptual matches should be recovered by an injected semantic
`relevance_scorer`. Calls without `query` preserve legacy static-score ordering.

### Embeddings and Hybrid Retrieval

Vector backends use the same provider-independent embedding contract. The
default provider lazily loads `all-MiniLM-L6-v2` through sentence-transformers;
applications can inject another `EmbeddingProvider` for hosted or local models.

Every stored vector records its provider, model, package/revision version,
dimension, and a SHA-256 hash of the embedded text. Use `reembed_all()` after a
provider or model change; `stale_only=True` updates only incompatible records.

Semantic results expose both normalized `similarity_score` and `hybrid_score`.
Hybrid ranking combines semantic similarity, component importance, recency, and
feedback using configurable `HybridWeights`. Both scores use a `[0, 1]` scale
across ChromaDB and PostgreSQL/pgvector.

Call `get_semantic_status()` on `SemanticContextManager` to detect missing or
failed embedding providers. Embedding failures are explicit and never replaced
with zero vectors.

### Memory Consolidation

Every component is categorized as an episode, durable fact, preference, goal,
derived summary, or generic memory. Lifecycle metadata records provenance,
supersession, contradictions, expiry, and confidence.

`ConsolidationEngine` can derive or merge memories, expire temporary knowledge,
and record contradictions without discarding either claim. Contradictions are
only resolved through an explicit winner/superseded decision. Normal retrieval
excludes superseded and expired memories while explaining each exclusion;
`include_inactive=True` remains available for audits.

See `examples/memory_consolidation_example.py` for a completely offline flow
that consolidates task episodes into a durable rule, resolves a changed user
preference, inspects retrieval decisions, and evaluates the result.

### Retrieval Evaluation

`RetrievalEvaluator` measures precision, recall, reciprocal rank, NDCG, token
efficiency, expected-exclusion accuracy, context stability, and downstream task
utility. A custom utility scorer can run an application-specific task check;
without one, graded NDCG is used as the utility proxy.

The checked-in `evaluations/agent_memory_retrieval.json` fixture provides an
initial repeatable dataset with relevant memories and distractors.

### Storage Concurrency

- JSON uses atomic file replacement, preventing partial files after interrupted
  writes. It does not coordinate simultaneous writers; use one writer per file.
- SQLite serializes transactional writes. A `SQLiteMemoryStore` instance owns a
  connection and should not be shared across threads; create one per thread.
- ChromaDB and PostgreSQL follow their respective backend concurrency models.
- Storage read/write failures raise typed exceptions instead of appearing as an
  empty memory set.

### Network Scenarios

**Automatic Fallback (Recommended):**
- Use `type = "auto_fallback"` in config.toml
- Automatically tries Ollama when available, falls back to naive when not
- Perfect for switching between networks
- Set `OLLAMA_HOST=http://localhost:11434` in `.env` for a local Ollama service

**Offline/Local Development:**
- Use `type = "naive"` in config.toml
- No external dependencies required
- Perfect for development and testing

**On Your Local Network:**
- Set `type = "ollama"` in config.toml
- Set `OLLAMA_HOST=http://localhost:11434` in `.env`
- Requires Ollama running on the local machine

**Internet Access:**
- Set `type = "openai"` in config.toml
- Set `OPENAI_API_KEY=your_key` in .env
- Requires OpenAI API access

## Security

- API keys are loaded from environment variables
- No sensitive data is stored in configuration files
- Local-first approach with Ollama support

## Running Tests

```bash
python test_runner.py
```

## CLI Usage

The AI Context Manager includes a command-line interface for easy management:

```bash
# Initialize a new project
ai-context init

# Show system status
ai-context status

# Search for content
ai-context search "AI trends"

# Get context for a query
ai-context context "research findings"

# Add content
ai-context add task --id t1 --name "Research" --content "Found insights"
ai-context add learning --id l1 --content "Vector DBs are faster" --source "testing"

# Manage configuration
ai-context config show
ai-context config optimize --use-case agent
```

## Performance Benchmarking

Run performance benchmarks to test your system:

```bash
python benchmark_performance.py
```

## Bourbon Research Desk demo

`bourbon-research` is a persistent, evidence-first research agent built to exercise
the context manager over multiple sessions. It discovers and snapshots sources,
extracts claims with provenance, records possible contradictions, consolidates
durable memory, and emits a cited Markdown report.

```bash
# Start a project (the workspace defaults to .bourbon-research/)
bourbon-research project create "Bourbon whiskey" \
  --objective "Document its history, law, production, finishes, recipes, and comparisons; separate evidence from folklore."

# Build the research questions, preview discovery, then ingest a small batch
bourbon-research research plan
bourbon-research --json research run --max-sources 5 --dry-run
bourbon-research --json research run --max-sources 3

# Inspect what persisted between sessions
bourbon-research status
bourbon-research sources
bourbon-research claims
bourbon-research contradictions
bourbon-research memory trace --token-budget 1200 \
  --query "Document bourbon's federal definition and disputed origins"
bourbon-research session changes

# Create durable derived memory and a cited working report
bourbon-research research consolidate
bourbon-research report --output bourbon-report.md
```

With `OPENAI_API_KEY`, discovery defaults to the Responses API web-search tool;
set `RESEARCH_SEARCH_PROVIDER=brave` to use `BRAVE_SEARCH_API_KEY` instead.
Without credentials, discovery uses Wikipedia and claim extraction uses a clearly
marked heuristic fallback, which is useful for testing the pipeline rather than
producing finished scholarship. `RESEARCH_SEARCH_MODEL` controls the OpenAI search
model, while `RESEARCH_MODEL` controls planning, extraction, and contradiction
detection. Source policy can be
selected per run with `--source-policy authoritative`, `exclude-community`, or
`all`; the default excludes community and retail sources.

## Quick Examples

Try the quick start examples:

```bash
python examples/quick_start.py
```

## Architecture

```
ai_context_manager/
├── components/          # Context component types
├── store/              # Storage backends
├── summarizers/        # Summarization engines
├── config.py           # Configuration management
├── context_manager.py  # Main context manager
├── feedback.py         # Feedback learning system
└── utils.py            # Utility functions
```

## Vector Database Benefits

**Development (ChromaDB):**
- ✅ **Easy setup** - No database server required
- ✅ **Fast prototyping** - Perfect for local development
- ✅ **Semantic search** - Natural language queries
- ✅ **Lightweight** - Minimal dependencies

**Production (PostgreSQL + pgvector):**
- ✅ **Enterprise-grade** - ACID transactions, backup/recovery
- ✅ **Horizontal scaling** - Read replicas, connection pooling
- ✅ **Advanced indexing** - HNSW/IVFFlat for sub-millisecond queries
- ✅ **Full-text search** - Combined with vector similarity
- ✅ **Monitoring** - Built-in observability and metrics

**Performance:**
- 🚀 **10x faster** than traditional keyword search
- 🚀 **Sub-millisecond** vector similarity queries
- 🚀 **Concurrent access** with connection pooling
- 🚀 **Memory efficient** with advanced indexing

**Installation:**
```bash
# Development
pip install ai-context-manager[vector]

# Production
pip install ai-context-manager[production]
```

## Recent Improvements

- ✅ **PostgreSQL + pgvector**: Added enterprise-grade vector database support
- ✅ **Production Setup**: Complete production deployment guide
- ✅ **Vector Database**: Added ChromaDB-based semantic similarity search
- ✅ **Semantic Context Manager**: Enhanced context retrieval for agents
- ✅ **Security**: Moved API keys to environment variables
- ✅ **Code Quality**: Consolidated duplicate code and improved error handling
- ✅ **Data Structures**: Standardized component storage with Dict-based approach
- ✅ **Validation**: Added comprehensive configuration validation
- ✅ **Error Handling**: Implemented consistent exception management throughout
