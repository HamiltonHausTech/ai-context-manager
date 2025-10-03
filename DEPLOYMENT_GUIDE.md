# Deployment Guide - AI Context Manager

This guide covers deploying the AI Context Manager and its demo applications.

## Quick Start

### 1. Local Development

```bash
# Install the package
pip install -e .

# Run the research assistant demo
python demo_apps/research_assistant/app.py

# Run the web demo
cd demo_apps/web_demo
pip install -r requirements.txt
python app.py
```

### 2. Docker Deployment

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Demo Applications

### Research Assistant CLI Demo

**Features:**
- Persistent memory across sessions
- Semantic search through research history
- Goal tracking and progress monitoring
- Learning from research patterns

**Usage:**
```bash
python demo_apps/research_assistant/app.py
```

**What it demonstrates:**
- Long-term memory persistence
- Semantic similarity search
- Agent goal management
- Research pattern learning

### Web Demo Application

**Features:**
- Interactive web interface
- Real-time research capabilities
- Semantic search demonstration
- Agent statistics and monitoring

**Usage:**
```bash
cd demo_apps/web_demo
pip install -r requirements.txt
python app.py
```

**Access:** http://localhost:5000

**What it demonstrates:**
- Web-based agent interaction
- Real-time context management
- Semantic search through web UI
- Agent performance monitoring

## Production Deployment

### 1. Environment Setup

**Required Services:**
- PostgreSQL 15+ with pgvector
- Python 3.8+
- Redis (optional, for session storage)

**Environment Variables:**
```bash
# Database
POSTGRES_HOST=your-postgres-host
POSTGRES_PORT=5432
POSTGRES_DB=ai_context
POSTGRES_USER=your-username
POSTGRES_PASSWORD=your-password

# Optional: OpenAI for embeddings
OPENAI_API_KEY=your-openai-key

# Optional: Ollama for local LLM
OLLAMA_HOST=http://your-ollama-host:11434
```

### 2. Database Setup

**PostgreSQL with pgvector:**
```sql
-- Create database
CREATE DATABASE ai_context;

-- Enable pgvector extension
CREATE EXTENSION vector;

-- Create user
CREATE USER ai_context_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE ai_context TO ai_context_user;
```

### 3. Application Deployment

**Using Docker:**
```bash
# Build production image
docker build -t ai-context-manager:latest .

# Run with production database
docker run -d \
  --name ai-context-manager \
  -p 5000:5000 \
  -e POSTGRES_HOST=your-postgres-host \
  -e POSTGRES_PASSWORD=your-password \
  ai-context-manager:latest
```

**Using Python directly:**
```bash
# Install package
pip install ai-context-manager[production]

# Configure environment
export POSTGRES_HOST=your-postgres-host
export POSTGRES_PASSWORD=your-password

# Run application
python your_app.py
```

### 4. Load Balancing

**Nginx Configuration:**
```nginx
upstream ai_context_manager {
    server 127.0.0.1:5000;
    server 127.0.0.1:5001;
    server 127.0.0.1:5002;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://ai_context_manager;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 5. Monitoring

**Health Check Endpoint:**
```python
@app.route('/health')
def health_check():
    try:
        # Check database connection
        stats = assistant.agent.get_stats()
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "stats": stats
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500
```

**Prometheus Metrics:**
```python
from prometheus_client import Counter, Histogram, generate_latest

# Define metrics
research_requests = Counter('research_requests_total', 'Total research requests')
research_duration = Histogram('research_duration_seconds', 'Research duration')

@app.route('/metrics')
def metrics():
    return generate_latest()
```

## Scaling Considerations

### 1. Database Scaling

**Read Replicas:**
```python
# Configure multiple database connections
POSTGRES_READ_REPLICA_1=replica1.example.com
POSTGRES_READ_REPLICA_2=replica2.example.com
```

**Connection Pooling:**
```python
# In config.toml
max_connections = 50
```

### 2. Application Scaling

**Horizontal Scaling:**
- Use multiple application instances
- Load balance with nginx/HAProxy
- Session storage in Redis
- Stateless application design

**Vertical Scaling:**
- Increase memory for vector operations
- Use faster CPUs for embedding generation
- SSD storage for database

### 3. Vector Database Optimization

**Index Tuning:**
```toml
# For high-performance queries
index_type = "hnsw"
index_parameters = { m = 32, ef_construction = 128 }

# For memory efficiency
index_type = "ivfflat"
index_parameters = { lists = 1000 }
```

**Query Optimization:**
- Use similarity thresholds
- Limit result sets
- Cache frequent queries
- Batch operations

## Security Considerations

### 1. Database Security

**SSL/TLS:**
```toml
# In config.toml
sslmode = "require"
sslcert = "/path/to/client-cert.pem"
sslkey = "/path/to/client-key.pem"
sslrootcert = "/path/to/ca-cert.pem"
```

**Row Level Security:**
```sql
-- Enable RLS
ALTER TABLE agent_memory ENABLE ROW LEVEL SECURITY;

-- Create policy
CREATE POLICY agent_isolation ON agent_memory
FOR ALL TO ai_context_user
USING (agent_id = current_setting('app.current_agent_id'));
```

### 2. Application Security

**API Keys:**
- Store in environment variables
- Rotate regularly
- Use different keys for different environments

**Input Validation:**
- Validate all user inputs
- Sanitize search queries
- Limit query complexity

**Rate Limiting:**
```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)

@app.route('/api/research')
@limiter.limit("10 per minute")
def research_topic():
    # Research endpoint with rate limiting
    pass
```

## Troubleshooting

### Common Issues

**1. Database Connection Issues:**
```bash
# Check PostgreSQL status
pg_isready -h localhost -p 5432

# Check pgvector extension
psql -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
```

**2. Memory Issues:**
```bash
# Monitor memory usage
htop
free -h

# Check PostgreSQL memory settings
psql -c "SHOW shared_buffers;"
psql -c "SHOW work_mem;"
```

**3. Performance Issues:**
```bash
# Check slow queries
psql -c "SELECT query, mean_time FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;"

# Analyze table statistics
psql -c "ANALYZE agent_memory;"
```

**4. Index Issues:**
```bash
# Check index usage
psql -c "SELECT indexname, idx_scan FROM pg_stat_user_indexes WHERE relname = 'agent_memory';"

# Rebuild indexes if needed
psql -c "REINDEX INDEX agent_memory_embedding_hnsw_idx;"
```

## Backup and Recovery

### 1. Database Backup

**Automated Backup:**
```bash
#!/bin/bash
# backup.sh
pg_dump -h localhost -U postgres -Fc ai_context > backup_$(date +%Y%m%d_%H%M%S).dump
```

**Cron Job:**
```bash
# Daily backup at 2 AM
0 2 * * * /path/to/backup.sh
```

### 2. Application Backup

**Configuration Backup:**
```bash
# Backup configuration
tar -czf config_backup_$(date +%Y%m%d).tar.gz config.toml .env
```

**Data Export:**
```python
# Export agent data
import json
from ai_context_manager.simple_api import create_agent_context_manager

agent = create_agent_context_manager("backup-agent")
stats = agent.get_stats()

with open('agent_backup.json', 'w') as f:
    json.dump(stats, f, indent=2)
```

## Performance Monitoring

### 1. Application Metrics

**Key Metrics to Monitor:**
- Research request rate
- Average response time
- Database connection pool usage
- Memory usage
- CPU usage
- Error rates

### 2. Database Metrics

**PostgreSQL Metrics:**
- Connection count
- Query performance
- Index usage
- Cache hit ratio
- Lock contention

### 3. Vector Database Metrics

**Specific Metrics:**
- Vector similarity query performance
- Index efficiency
- Embedding generation time
- Memory usage for vectors

## Support

For deployment support:
- 📧 **Email**: support@ai-context-manager.com
- 📚 **Documentation**: https://docs.ai-context-manager.com
- 🐛 **Issues**: https://github.com/ai-context-manager/ai-context-manager/issues
- 💬 **Discord**: https://discord.gg/ai-context-manager
