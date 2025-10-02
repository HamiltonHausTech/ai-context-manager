import logging
from ai_context_manager.context_manager import ContextManager
from ai_context_manager.feedback import Feedback
from ai_context_manager.utils import load_summarizer
from ai_context_manager.components.task_summary import TaskSummaryComponent
from ai_context_manager.config import Config
from ai_context_manager.utils import load_stores_from_config

logging.basicConfig(level=logging.INFO)

def test_memory_store_usage(ctx: ContextManager):
    comp = TaskSummaryComponent(
        id="memory-test-001",
        task_name="Memory Store Test",
        summary="This is a test to persist component state.",
        score=1.2,
        tags=["task", "persistence"]
    )
    print("[TEST] Registering and saving component to memory...")
    ctx.register_component(comp)
    ctx.save_component_to_memory(comp)
    print("[OK] Saved. Re-run to check if it reloads from memory.\n")

def test_dry_run(ctx: ContextManager):
    print("[TEST] Running dry run:")
    ctx.get_context(token_budget=100, dry_run=True)

def test_feedback_summary(feedback: Feedback):
    print("[TEST] Feedback Summary:\n")
    print(feedback.render_feedback())

def run_test_from_config():
    try:
        config_obj = Config("config.toml")
        config = config_obj.data
        feedback_store, memory_store = load_stores_from_config(config)
        feedback = Feedback(store=feedback_store)

        ctx = ContextManager(
            feedback=feedback,
            summarizer=load_summarizer(config_obj),  # Pass config object instead of dict
            memory_store=memory_store,
            config=config
        )
    except Exception as e:
        logging.error(f"Failed to initialize from config: {e}")
        raise

    # --- Run tests ---
    test_memory_store_usage(ctx)
    test_dry_run(ctx)
    test_feedback_summary(feedback)

if __name__ == "__main__":
    run_test_from_config()
