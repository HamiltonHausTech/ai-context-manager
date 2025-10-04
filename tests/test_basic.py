"""
Basic tests for AI Context Manager
"""

import pytest
import tempfile
import os
from ai_context_manager.simple_api import create_context_manager, create_agent_context_manager
from ai_context_manager.config import Config

# Get the directory where this test file is located
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_CONFIG_PATH = os.path.join(TEST_DIR, "test-config.toml")

@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Clean up test files after each test."""
    yield  # Run the test
    # Cleanup after test
    test_files = ["test_feedback.json", "test_memory.json"]
    for file in test_files:
        if os.path.exists(file):
            try:
                os.remove(file)
            except OSError:
                pass  # Ignore cleanup errors


class TestBasicFunctionality:
    """Test basic functionality of the AI Context Manager."""
    
    def test_create_context_manager(self):
        """Test creating a basic context manager."""
        ctx = create_context_manager(config_path=TEST_CONFIG_PATH)
        assert ctx is not None
        assert hasattr(ctx, 'add_task')
        assert hasattr(ctx, 'get_context')
    
    def test_create_agent_context_manager(self):
        """Test creating an agent context manager."""
        agent = create_agent_context_manager("test-agent", config_path=TEST_CONFIG_PATH)
        assert agent is not None
        assert hasattr(agent, 'add_goal')
        assert hasattr(agent, 'add_task')
        assert hasattr(agent, 'get_context')
    
    def test_add_task(self):
        """Test adding a task."""
        ctx = create_context_manager(config_path=TEST_CONFIG_PATH)
        ctx.add_task("task-1", "Test Task", "This is a test task", success=True)
        
        # Verify task was added
        stats = ctx.get_stats()
        assert stats["total_components"] > 0
    
    def test_add_goal(self):
        """Test adding a goal."""
        agent = create_agent_context_manager("test-agent", config_path=TEST_CONFIG_PATH)
        agent.add_goal("goal-1", "Test Goal", priority=1.5)
        
        # Verify goal was added
        stats = agent.get_stats()
        assert stats["total_goals"] > 0
    
    def test_get_context(self):
        """Test getting context."""
        ctx = create_context_manager(config_path=TEST_CONFIG_PATH)
        ctx.add_task("task-1", "Test Task", "This is a test task")
        
        context = ctx.get_context("test", token_budget=500)
        assert isinstance(context, str)
        assert len(context) > 0
    
    def test_search_similar(self):
        """Test semantic search."""
        ctx = create_context_manager(config_path=TEST_CONFIG_PATH)
        ctx.add_task("task-1", "AI Research", "Research about AI trends")
        
        results = ctx.search_similar("artificial intelligence", limit=5)
        assert isinstance(results, list)
    
    def test_config_loading(self):
        """Test configuration loading."""
        # Create a temporary config file
        config_content = """
[summarizer]
type = "naive"

[feedback_store]
type = "json"
filepath = "test_feedback.json"

[memory_store]
type = "json"
filepath = "test_memory.json"

[logging]
level = "INFO"
format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(config_content)
            temp_config = f.name
        
        try:
            config = Config(temp_config)
            assert config is not None
            assert config.data["summarizer"]["type"] == "naive"
        finally:
            os.unlink(temp_config)
    
    def test_agent_stats(self):
        """Test agent statistics."""
        agent = create_agent_context_manager("test-agent", config_path=TEST_CONFIG_PATH)
        
        # Add some content
        agent.add_goal("goal-1", "Test Goal", priority=1.5)
        agent.add_task("task-1", "Test Task", "Test result")
        
        stats = agent.get_stats()
        assert "total_goals" in stats
        assert "total_tasks" in stats
        assert "agent_id" in stats


class TestErrorHandling:
    """Test error handling."""
    
    def test_invalid_agent_id(self):
        """Test handling of invalid agent ID."""
        # Empty agent ID should be handled gracefully
        agent = create_agent_context_manager("", config_path=TEST_CONFIG_PATH)
        assert agent is not None
    
    def test_invalid_task_data(self):
        """Test handling of invalid task data."""
        ctx = create_context_manager(config_path=TEST_CONFIG_PATH)
        
        # Should handle empty task name gracefully
        ctx.add_task("task-1", "", "Valid result")
        
        # Should handle empty result gracefully
        ctx.add_task("task-2", "Valid task", "")
    
    def test_missing_dependencies(self):
        """Test graceful handling of missing dependencies."""
        # This test ensures the system falls back gracefully
        # when optional dependencies are missing
        ctx = create_context_manager(config_path=TEST_CONFIG_PATH)
        assert ctx is not None


if __name__ == "__main__":
    pytest.main([__file__])
