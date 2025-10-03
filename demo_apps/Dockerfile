FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY pyproject.toml ./
COPY demo_apps/requirements.txt ./demo_apps/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -e .[production]
RUN pip install --no-cache-dir -r demo_apps/requirements.txt

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

# Default command (can be overridden)
CMD ["python", "demo_apps/web_demo/app.py"]
