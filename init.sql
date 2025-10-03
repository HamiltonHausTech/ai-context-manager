-- Initialize PostgreSQL database for AI Context Manager Demo

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create demo database user
CREATE USER demo_user WITH PASSWORD 'demo_password';
GRANT ALL PRIVILEGES ON DATABASE ai_context TO demo_user;

-- Create additional indexes for demo
CREATE INDEX IF NOT EXISTS agent_memory_created_at_desc_idx ON agent_memory (created_at DESC);
CREATE INDEX IF NOT EXISTS agent_memory_score_desc_idx ON agent_memory (score DESC);
