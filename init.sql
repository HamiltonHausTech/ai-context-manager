-- Initialize PostgreSQL database for AI Context Manager Demo

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create demo database user
CREATE USER demo_user WITH PASSWORD 'demo_password';
GRANT ALL PRIVILEGES ON DATABASE ai_context TO demo_user;
