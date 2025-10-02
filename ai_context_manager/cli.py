"""
Command-line interface for AI Context Manager
"""

import argparse
import json
import logging
import sys
from typing import Dict, Any

from .simple_api import create_context_manager, create_agent_context_manager
from .auto_config import auto_detect_and_setup, create_optimal_config

def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='[%(levelname)s] %(message)s'
    )

def cmd_init(args):
    """Initialize a new context manager project."""
    print("🚀 Initializing AI Context Manager...")
    
    # Auto-detect environment
    config = auto_detect_and_setup()
    
    # Create config file
    import toml
    with open("config.toml", 'w') as f:
        toml.dump(config, f)
    
    print("✅ Configuration created: config.toml")
    print("📝 Edit config.toml to customize settings")
    
    # Create example usage
    example_code = '''
# Example usage
from ai_context_manager.simple_api import create_agent_context_manager

# Create agent context manager
agent = create_agent_context_manager("my-agent")

# Add some tasks
agent.add_task("task-1", "Research Task", "Found interesting insights about AI trends")
agent.add_learning("learning-1", "Vector databases are 10x faster than traditional search", "research")

# Get context
context = agent.get_context("AI research trends", token_budget=1000)
print(context)
'''
    
    with open("example.py", 'w') as f:
        f.write(example_code)
    
    print("📄 Example created: example.py")
    print("🎯 Run: python example.py")

def cmd_status(args):
    """Show system status and statistics."""
    try:
        if args.agent_id:
            ctx = create_agent_context_manager(args.agent_id)
        else:
            ctx = create_context_manager()
        
        stats = ctx.get_stats()
        
        print("📊 Context Manager Status")
        print("=" * 40)
        print(f"Total Components: {stats.get('total_components', 0)}")
        
        if 'component_types' in stats:
            print("\\nComponent Types:")
            for comp_type, count in stats['component_types'].items():
                print(f"  {comp_type}: {count}")
        
        if 'semantic_search_enabled' in stats:
            print(f"\\nSemantic Search: {'✅ Enabled' if stats['semantic_search_enabled'] else '❌ Disabled'}")
        
        if 'uptime_minutes' in stats:
            print(f"\\nAgent Uptime: {stats['uptime_minutes']:.1f} minutes")
        
    except Exception as e:
        print(f"❌ Error getting status: {e}")

def cmd_search(args):
    """Search for similar content."""
    try:
        if args.agent_id:
            ctx = create_agent_context_manager(args.agent_id)
        else:
            ctx = create_context_manager()
        
        results = ctx.search_similar(args.query, limit=args.limit)
        
        print(f"🔍 Search Results for: '{args.query}'")
        print("=" * 50)
        
        if not results:
            print("No similar content found.")
            return
        
        for i, result in enumerate(results, 1):
            similarity = result.get('similarity_score', 0.0)
            comp_type = result.get('type', 'Unknown')
            content = result.get('content', '')[:100]
            
            print(f"{i}. [{comp_type}] (similarity: {similarity:.3f})")
            print(f"   {content}...")
            print()
    
    except Exception as e:
        print(f"❌ Error searching: {e}")

def cmd_context(args):
    """Get context for a query."""
    try:
        if args.agent_id:
            ctx = create_agent_context_manager(args.agent_id)
        else:
            ctx = create_context_manager()
        
        context = ctx.get_context(args.query, token_budget=args.tokens)
        
        print(f"📝 Context for: '{args.query}'")
        print("=" * 50)
        print(context)
    
    except Exception as e:
        print(f"❌ Error getting context: {e}")

def cmd_add(args):
    """Add content to the context manager."""
    try:
        if args.agent_id:
            ctx = create_agent_context_manager(args.agent_id)
        else:
            ctx = create_context_manager()
        
        if args.type == "task":
            ctx.add_task(args.id, args.name, args.content, args.tags.split(',') if args.tags else None)
            print(f"✅ Added task: {args.name}")
        
        elif args.type == "learning":
            ctx.add_learning(args.id, args.content, args.source, args.importance, 
                           args.tags.split(',') if args.tags else None)
            print(f"✅ Added learning: {args.content[:50]}...")
        
        elif args.type == "goal" and args.agent_id:
            ctx.add_goal(args.id, args.content, args.priority)
            print(f"✅ Added goal: {args.content}")
        
        else:
            print("❌ Invalid type or missing agent_id for goals")
    
    except Exception as e:
        print(f"❌ Error adding content: {e}")

def cmd_config(args):
    """Manage configuration."""
    if args.action == "show":
        try:
            with open("config.toml", 'r') as f:
                content = f.read()
            print("📄 Current Configuration:")
            print("=" * 30)
            print(content)
        except FileNotFoundError:
            print("❌ No config.toml found. Run 'ai-context init' first.")
    
    elif args.action == "optimize":
        use_case = args.use_case or "agent"
        config = create_optimal_config(use_case)
        
        import toml
        with open("config.toml", 'w') as f:
            toml.dump(config, f)
        
        print(f"✅ Optimized configuration for {use_case} use case")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AI Context Manager CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ai-context init                    # Initialize new project
  ai-context status --agent my-agent # Show agent status
  ai-context search "AI trends"      # Search for similar content
  ai-context context "research"      # Get context for query
  ai-context add task --id t1 --name "Research" --content "Found insights"
        """
    )
    
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--agent-id", help="Agent ID for agent-specific operations")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Init command
    subparsers.add_parser("init", help="Initialize new project")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show system status")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search for similar content")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--limit", "-l", type=int, default=5, help="Number of results")
    
    # Context command
    context_parser = subparsers.add_parser("context", help="Get context for query")
    context_parser.add_argument("query", help="Context query")
    context_parser.add_argument("--tokens", "-t", type=int, default=1000, help="Token budget")
    
    # Add command
    add_parser = subparsers.add_parser("add", help="Add content")
    add_parser.add_argument("type", choices=["task", "learning", "goal"], help="Content type")
    add_parser.add_argument("--id", required=True, help="Content ID")
    add_parser.add_argument("--name", help="Content name (for tasks)")
    add_parser.add_argument("--content", required=True, help="Content text")
    add_parser.add_argument("--source", help="Source (for learnings)")
    add_parser.add_argument("--importance", type=float, default=1.0, help="Importance score")
    add_parser.add_argument("--priority", type=float, default=1.0, help="Priority (for goals)")
    add_parser.add_argument("--tags", help="Comma-separated tags")
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Manage configuration")
    config_parser.add_argument("action", choices=["show", "optimize"], help="Config action")
    config_parser.add_argument("--use-case", choices=["agent", "simple", "production"], 
                             help="Use case for optimization")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging("DEBUG" if args.verbose else "INFO")
    
    # Route to command handlers
    if args.command == "init":
        cmd_init(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "context":
        cmd_context(args)
    elif args.command == "add":
        cmd_add(args)
    elif args.command == "config":
        cmd_config(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
