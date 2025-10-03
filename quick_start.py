#!/usr/bin/env python3
"""
AI Context Manager Quick Start Script

This script helps you get started with the AI Context Manager quickly.
It demonstrates the key features and helps you choose the right setup.
"""

import os
import sys
import subprocess
import platform

def print_banner():
    """Print welcome banner."""
    print("=" * 60)
    print("🤖 AI Context Manager - Quick Start")
    print("=" * 60)
    print()
    print("Enterprise-grade context management for AI agents")
    print("with vector database support and semantic search.")
    print()

def check_requirements():
    """Check system requirements."""
    print("🔍 Checking system requirements...")
    
    # Check Python version
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    
    # Check if we're in the right directory
    if not os.path.exists("pyproject.toml"):
        print("❌ Please run this script from the AI Context Manager root directory")
        return False
    
    print("✅ AI Context Manager project found")
    return True

def choose_setup():
    """Let user choose setup type."""
    print("\n🚀 Choose your setup:")
    print()
    print("1. 🏠 Local Development (ChromaDB + JSON)")
    print("   - Easy setup, no database server required")
    print("   - Perfect for prototyping and development")
    print()
    print("2. 🏢 Production Setup (PostgreSQL + pgvector)")
    print("   - Enterprise-grade with full database features")
    print("   - Requires PostgreSQL with pgvector extension")
    print()
    print("3. 🌐 Web Demo (Flask + PostgreSQL)")
    print("   - Interactive web interface")
    print("   - Full demo application with UI")
    print()
    print("4. 📊 Research Assistant Demo (CLI)")
    print("   - Command-line research assistant")
    print("   - Demonstrates agent capabilities")
    print()
    
    while True:
        choice = input("Enter your choice (1-4): ").strip()
        if choice in ["1", "2", "3", "4"]:
            return int(choice)
        print("Please enter 1, 2, 3, or 4")

def setup_development():
    """Setup development environment."""
    print("\n🏠 Setting up local development environment...")
    
    # Install basic dependencies
    print("📦 Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".", "[vector]"], check=True)
    
    # Create example configuration
    config_content = '''[summarizer]
type = "auto_fallback"
model = "mistral"

[feedback_store]
type = "json"
filepath = "feedback.json"

[memory_store]
type = "vector"
collection_name = "agent_memory"
persist_directory = "./chroma_db"
embedding_model = "all-MiniLM-L6-v2"
'''
    
    with open("config.toml", "w") as f:
        f.write(config_content)
    
    print("✅ Development environment ready!")
    print("\n📝 Next steps:")
    print("1. Run: python examples/quick_start.py")
    print("2. Or: python demo_apps/research_assistant/app.py")

def setup_production():
    """Setup production environment."""
    print("\n🏢 Setting up production environment...")
    
    print("📋 Prerequisites:")
    print("- PostgreSQL 15+ with pgvector extension")
    print("- Database credentials")
    print()
    
    # Check if PostgreSQL is available
    try:
        result = subprocess.run(["pg_isready"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ PostgreSQL is running")
        else:
            print("⚠️  PostgreSQL not detected. Please install and start PostgreSQL.")
    except FileNotFoundError:
        print("⚠️  PostgreSQL not found. Please install PostgreSQL first.")
    
    # Install production dependencies
    print("📦 Installing production dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".", "[production]"], check=True)
    
    # Get database configuration
    print("\n🔧 Database Configuration:")
    host = input("PostgreSQL host [localhost]: ").strip() or "localhost"
    port = input("PostgreSQL port [5432]: ").strip() or "5432"
    database = input("Database name [ai_context]: ").strip() or "ai_context"
    user = input("Database user [postgres]: ").strip() or "postgres"
    password = input("Database password: ").strip()
    
    # Create production configuration
    config_content = f'''[summarizer]
type = "auto_fallback"
model = "mistral"

[feedback_store]
type = "sqlite"
db_path = "feedback.db"

[memory_store]
type = "postgres_vector"
host = "{host}"
port = {port}
database = "{database}"
user = "{user}"
password = "{password}"
table_name = "agent_memory"
embedding_dimension = 384
index_type = "hnsw"
index_parameters = {{ m = 16, ef_construction = 64 }}
'''
    
    with open("config.toml", "w") as f:
        f.write(config_content)
    
    print("✅ Production environment ready!")
    print("\n📝 Next steps:")
    print("1. Ensure PostgreSQL is running with pgvector extension")
    print("2. Run: python examples/quick_start.py")
    print("3. Or: python demo_apps/research_assistant/app.py")

def setup_web_demo():
    """Setup web demo."""
    print("\n🌐 Setting up web demo...")
    
    # Install web dependencies
    print("📦 Installing web dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".", "[production]"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "demo_apps/requirements.txt"], check=True)
    
    print("✅ Web demo ready!")
    print("\n📝 Next steps:")
    print("1. Start PostgreSQL: docker-compose up -d postgres")
    print("2. Run web demo: python demo_apps/web_demo/app.py")
    print("3. Open: http://localhost:5000")

def setup_research_demo():
    """Setup research assistant demo."""
    print("\n📊 Setting up research assistant demo...")
    
    # Install dependencies
    print("📦 Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".", "[vector]"], check=True)
    
    print("✅ Research assistant ready!")
    print("\n📝 Next steps:")
    print("1. Run: python demo_apps/research_assistant/app.py")
    print("2. Watch the agent research topics and build knowledge!")

def show_examples():
    """Show usage examples."""
    print("\n📚 Usage Examples:")
    print()
    print("Basic Usage:")
    print("```python")
    print("from ai_context_manager.simple_api import create_agent_context_manager")
    print()
    print("# Create agent")
    print("agent = create_agent_context_manager('my-agent')")
    print()
    print("# Add goals and tasks")
    print("agent.add_goal('goal-1', 'Research AI trends', priority=2.0)")
    print("agent.add_task('task-1', 'Market Research', 'Found interesting insights')")
    print()
    print("# Get context")
    print("context = agent.get_context('AI research trends', token_budget=1000)")
    print("```")
    print()
    print("Advanced Usage:")
    print("```python")
    print("from ai_context_manager.simple_api import create_context_manager")
    print()
    print("# Create context manager with custom config")
    print("ctx = create_context_manager('config.toml')")
    print()
    print("# Search similar content")
    print("results = ctx.search_similar('vector databases', limit=5)")
    print("```")

def main():
    """Main quick start function."""
    print_banner()
    
    if not check_requirements():
        sys.exit(1)
    
    choice = choose_setup()
    
    if choice == 1:
        setup_development()
    elif choice == 2:
        setup_production()
    elif choice == 3:
        setup_web_demo()
    elif choice == 4:
        setup_research_demo()
    
    show_examples()
    
    print("\n" + "=" * 60)
    print("🎉 Setup Complete!")
    print("=" * 60)
    print()
    print("📚 Additional Resources:")
    print("- Documentation: README.md")
    print("- Production Guide: PRODUCTION_SETUP.md")
    print("- Deployment Guide: DEPLOYMENT_GUIDE.md")
    print("- Examples: examples/ directory")
    print("- Demo Apps: demo_apps/ directory")
    print()
    print("🚀 Ready to build amazing AI agents!")

if __name__ == "__main__":
    main()
