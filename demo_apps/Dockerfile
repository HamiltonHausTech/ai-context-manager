# Multi-stage build for faster builds and smaller images
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY pyproject.toml ./
COPY demo_apps/requirements.txt ./demo_apps/

# Install Python dependencies with minimal set for demo
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -e .[cli]
RUN pip install --no-cache-dir -r demo_apps/requirements.txt

# Production stage
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY ai_context_manager/ ./ai_context_manager/
COPY demo_apps/ ./demo_apps/

# Create default configuration files
RUN echo "# AI Context Manager Configuration" > ./config.toml && \
    echo "[summarizer]" >> ./config.toml && \
    echo "type = \"naive\"" >> ./config.toml && \
    echo "" >> ./config.toml && \
    echo "[feedback_store]" >> ./config.toml && \
    echo "type = \"json\"" >> ./config.toml && \
    echo "filepath = \"feedback.json\"" >> ./config.toml && \
    echo "" >> ./config.toml && \
    echo "[memory_store]" >> ./config.toml && \
    echo "type = \"json\"" >> ./config.toml && \
    echo "filepath = \"memory.json\"" >> ./config.toml

# Create default environment file
RUN echo "# AI Context Manager Environment" > ./.env && \
    echo "# Add your environment variables here" >> ./.env

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 5000

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

# Default command (can be overridden)
CMD ["python", "demo_apps/web_demo/app.py"]
