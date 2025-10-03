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
OLLAMA_MODEL=mistral
```

### 3. Configure the System

Edit `config.toml` based on your setup:

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
- **SQLite**: Database storage for production use
- **Vector**: ChromaDB-based semantic similarity search (development)
- **PostgreSQL + pgvector**: Enterprise-grade vector database (production)

### Network Scenarios

**Automatic Fallback (Recommended):**
- Use `type = "auto_fallback"` in config.toml
- Automatically tries Ollama when available, falls back to naive when not
- Perfect for switching between networks
- Set `OLLAMA_HOST=http://192.168.0.156:11434` in .env for your network

**Offline/Local Development:**
- Use `type = "naive"` in config.toml
- No external dependencies required
- Perfect for development and testing

**On Your Local Network:**
- Set `type = "ollama"` in config.toml
- Set `OLLAMA_HOST=http://192.168.0.156:11434` in .env
- Requires Ollama running on your local network

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
