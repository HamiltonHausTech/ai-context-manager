# Production Setup Guide - PostgreSQL + pgvector

This guide covers setting up the AI Context Manager with PostgreSQL + pgvector for production deployments.

## Why PostgreSQL + pgvector?

**Enterprise Features:**
- ✅ **ACID transactions** - Data consistency and reliability
- ✅ **Horizontal scaling** - Read replicas and connection pooling
- ✅ **Advanced indexing** - HNSW and IVFFlat vector indexes
- ✅ **Full-text search** - Combined with vector similarity
- ✅ **Backup & recovery** - Enterprise-grade data protection
- ✅ **Monitoring** - Built-in observability and metrics

**Performance Benefits:**
- 🚀 **10x faster** than traditional keyword search
- 🚀 **Sub-millisecond** vector similarity queries
- 🚀 **Concurrent access** with connection pooling
- 🚀 **Memory efficient** with advanced indexing

## Prerequisites

1. **PostgreSQL 15+** with pgvector extension
2. **Python 3.8+**
3. **Embedding model** (optional, can use OpenAI embeddings)

## Installation

### 1. Install PostgreSQL with pgvector

**Ubuntu/Debian:**
```bash
# Install PostgreSQL
sudo apt update
sudo apt install postgresql postgresql-contrib

# Install pgvector extension
sudo apt install postgresql-15-pgvector
```

**macOS (Homebrew):**
```bash
# Install PostgreSQL with pgvector
brew install pgvector
```

**Docker:**
```bash
# Run PostgreSQL with pgvector
docker run --name postgres-pgvector \
  -e POSTGRES_PASSWORD=your_password \
  -e POSTGRES_DB=ai_context \
  -p 5432:5432 \
  -d pgvector/pgvector:pg15
```

### 2. Setup Database

```sql
-- Connect to PostgreSQL
psql -U postgres -d ai_context

-- Enable pgvector extension
CREATE EXTENSION vector;

-- Create database user (optional)
CREATE USER ai_context_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE ai_context TO ai_context_user;
```

### 3. Install Python Dependencies

```bash
# Production installation with PostgreSQL support
pip install -e .[production]

# Or install manually
pip install psycopg2-binary numpy sentence-transformers
```

## Configuration

### 1. Environment Variables

Create `.env` file:
```bash
# PostgreSQL Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=ai_context
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_TABLE_NAME=agent_memory
POSTGRES_EMBEDDING_DIM=384
POSTGRES_MAX_CONNECTIONS=20

# Optional: OpenAI for embeddings
OPENAI_API_KEY=your_openai_api_key
```

### 2. Configuration File

Update `config.toml`:
```toml
[summarizer]
type = "auto_fallback"
model = "mistral"

[feedback_store]
type = "sqlite"
db_path = "feedback.db"

[memory_store]
type = "postgres_vector"
host = "localhost"
port = 5432
database = "ai_context"
user = "postgres"
password = "your_password"
table_name = "agent_memory"
embedding_dimension = 384
max_connections = 20
index_type = "hnsw"
index_parameters = { m = 16, ef_construction = 64 }
```

## Index Configuration

### HNSW Index (Recommended)

**High Performance, Higher Memory:**
```toml
index_type = "hnsw"
index_parameters = { 
    m = 16,                    # Number of bi-directional links (16-48)
    ef_construction = 64       # Size of dynamic candidate list (64-200)
}
```

### IVFFlat Index

**Lower Memory, Good Performance:**
```toml
index_type = "ivfflat"
index_parameters = { 
    lists = 100               # Number of clusters (rows/1000)
}
```

## Performance Tuning

### 1. Connection Pooling

```toml
# In config.toml
max_connections = 20  # Adjust based on load
```

### 2. PostgreSQL Configuration

Edit `postgresql.conf`:
```conf
# Memory settings
shared_buffers = 256MB
work_mem = 4MB
maintenance_work_mem = 64MB

# Connection settings
max_connections = 100
shared_preload_libraries = 'vector'

# Checkpoint settings
checkpoint_completion_target = 0.9
wal_buffers = 16MB
```

### 3. Index Maintenance

```sql
-- Analyze table for query planning
ANALYZE agent_memory;

-- Reindex if needed
REINDEX INDEX agent_memory_embedding_hnsw_idx;
```

## Monitoring

### 1. Database Statistics

```python
from ai_context_manager.store.postgres_vector_memory import PostgreSQLVectorMemoryStore

store = PostgreSQLVectorMemoryStore(...)
stats = store.get_stats()
print(f"Total components: {stats['total_components']}")
print(f"Table size: {stats['table_size']}")
print(f"Indexes: {len(stats['indexes'])}")
```

### 2. Query Performance

```sql
-- Check index usage
EXPLAIN (ANALYZE, BUFFERS) 
SELECT id, 1 - (embedding <=> '[0.1,0.2,0.3]') as similarity
FROM agent_memory 
ORDER BY embedding <=> '[0.1,0.2,0.3]' 
LIMIT 10;
```

### 3. System Monitoring

```bash
# Monitor PostgreSQL
pg_stat_activity
pg_stat_database
pg_stat_user_tables
```

## Scaling

### 1. Read Replicas

```bash
# Setup streaming replica
pg_basebackup -h primary_host -D /var/lib/postgresql/replica -U replicator -v -P -W
```

### 2. Connection Pooling

Use PgBouncer:
```ini
[databases]
ai_context = host=localhost port=5432 dbname=ai_context

[pgbouncer]
pool_mode = transaction
max_client_conn = 100
default_pool_size = 20
```

### 3. Horizontal Partitioning

```sql
-- Partition by date
CREATE TABLE agent_memory_2025_01 PARTITION OF agent_memory
FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
```

## Backup & Recovery

### 1. Database Backup

```bash
# Full backup
pg_dump -h localhost -U postgres ai_context > backup.sql

# With compression
pg_dump -h localhost -U postgres -Fc ai_context > backup.dump
```

### 2. Point-in-Time Recovery

```bash
# Enable WAL archiving
archive_mode = on
archive_command = 'cp %p /backup/wal/%f'
```

## Security

### 1. SSL/TLS

```toml
# In config.toml
sslmode = "require"
sslcert = "/path/to/client-cert.pem"
sslkey = "/path/to/client-key.pem"
sslrootcert = "/path/to/ca-cert.pem"
```

### 2. Row Level Security

```sql
-- Enable RLS
ALTER TABLE agent_memory ENABLE ROW LEVEL SECURITY;

-- Create policy
CREATE POLICY agent_isolation ON agent_memory
FOR ALL TO ai_context_user
USING (agent_id = current_setting('app.current_agent_id'));
```

## Troubleshooting

### Common Issues

**1. pgvector extension not found:**
```bash
sudo apt install postgresql-15-pgvector
```

**2. Connection refused:**
```bash
# Check PostgreSQL status
sudo systemctl status postgresql
sudo systemctl start postgresql
```

**3. Index not being used:**
```sql
-- Check index exists
\d agent_memory

-- Analyze table
ANALYZE agent_memory;
```

**4. Memory issues:**
```bash
# Increase work_mem in postgresql.conf
work_mem = 8MB
```

## Migration from ChromaDB

```python
# Migration script
from ai_context_manager.store.vector_memory import VectorMemoryStore
from ai_context_manager.store.postgres_vector_memory import PostgreSQLVectorMemoryStore

# Load from ChromaDB
chroma_store = VectorMemoryStore()
components = chroma_store.load_all()

# Save to PostgreSQL
pg_store = PostgreSQLVectorMemoryStore()
for component in components:
    pg_store.save_component(component)

print(f"Migrated {len(components)} components to PostgreSQL")
```

## Production Checklist

- ✅ **PostgreSQL 15+** with pgvector installed
- ✅ **Connection pooling** configured
- ✅ **Indexes** created (HNSW or IVFFlat)
- ✅ **Backup strategy** implemented
- ✅ **Monitoring** set up
- ✅ **SSL/TLS** enabled
- ✅ **Row-level security** configured
- ✅ **Performance tuning** completed
- ✅ **Load testing** performed

## Support

For production support:
- 📧 **Email**: support@ai-context-manager.com
- 📚 **Documentation**: https://docs.ai-context-manager.com
- 🐛 **Issues**: https://github.com/yourusername/ai-context-manager/issues
