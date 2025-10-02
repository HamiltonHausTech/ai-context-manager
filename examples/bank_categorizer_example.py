#!/usr/bin/env python3
"""
Example: Bank Transaction Categorizer Agent
Demonstrates how to use the system for a long-running transaction categorization agent.
"""

import logging
from datetime import datetime, timedelta
from ai_context_manager.agent_context_manager import AgentContextManager
from ai_context_manager.context_manager import ContextManager
from ai_context_manager.config import Config
from ai_context_manager.utils import load_stores_from_config, load_summarizer
from ai_context_manager.components.longterm_summary import LongTermMemoryComponent

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def simulate_bank_categorizer():
    """Simulate a bank transaction categorization agent."""
    
    # Initialize the agent context manager
    config = Config("config.toml")
    config_data = config.data
    feedback_store, memory_store = load_stores_from_config(config_data)
    
    # Create base context manager
    from ai_context_manager.feedback import Feedback
    feedback = Feedback(store=feedback_store)
    
    base_ctx = ContextManager(
        feedback=feedback,
        memory_store=memory_store,
        summarizer=load_summarizer(config),
        config=config_data
    )
    
    # Create agent context manager
    agent_id = "bank-categorizer-001"
    agent = AgentContextManager(agent_id, base_ctx)
    
    print("=" * 60)
    print(f"Bank Categorizer Agent: {agent_id}")
    print("=" * 60)
    
    # Set up categorization goals
    print("\n[SETUP] Setting categorization goals...")
    agent.add_goal(
        goal_id="improve-categorization-accuracy",
        goal_description="Achieve 95% accuracy in transaction categorization",
        priority=2.0,
        deadline=(datetime.now() + timedelta(days=30)).isoformat(),
        tags=["categorization", "accuracy", "machine-learning"]
    )
    
    agent.add_goal(
        goal_id="reduce-manual-review",
        goal_description="Reduce manual review cases to under 5%",
        priority=1.5,
        deadline=(datetime.now() + timedelta(days=21)).isoformat(),
        tags=["automation", "efficiency", "manual-review"]
    )
    
    # Simulate learning from historical data
    print("\n[LEARNING] Learning from historical patterns...")
    
    # Learn merchant patterns
    agent.record_learning(
        learning_id="starbucks-pattern",
        content="Merchant 'STARBUCKS' appears in 847 transactions, 99.8% categorized as 'Food & Dining'",
        source="historical-analysis",
        importance=2.0,
        tags=["merchant-pattern", "starbucks", "food-dining", "high-confidence"]
    )
    
    agent.record_learning(
        learning_id="amazon-pattern",
        content="Merchant 'AMAZON' appears in 1,234 transactions, 85% 'Shopping', 10% 'Electronics', 5% 'Books'",
        source="historical-analysis",
        importance=1.8,
        tags=["merchant-pattern", "amazon", "shopping", "multi-category"]
    )
    
    agent.record_learning(
        learning_id="gas-station-pattern",
        content="Merchants containing 'SHELL', 'EXXON', 'BP' typically categorize as 'Gas & Transportation'",
        source="historical-analysis",
        importance=1.5,
        tags=["merchant-pattern", "gas-stations", "transportation", "keyword-matching"]
    )
    
    # Learn user-specific patterns
    agent.record_learning(
        learning_id="user-hobby-pattern",
        content="User frequently has transactions at 'MICHAELS CRAFT' and 'HOBBY LOBBY' - should be 'Hobbies & Recreation'",
        source="user-behavior-analysis",
        importance=1.3,
        tags=["user-pattern", "hobbies", "craft-stores", "personalized"]
    )
    
    # Simulate transaction categorization tasks
    print("\n[CATEGORIZATION] Processing transactions...")
    
    # Transaction 1: Clear case
    agent.record_task_result(
        task_id="txn-001",
        task_name="Categorize STARBUCKS transaction",
        result="""
        Transaction: $4.50 at STARBUCKS
        Prediction: Food & Dining (confidence: 99.8%)
        Reasoning: Historical pattern shows 99.8% of Starbucks transactions are Food & Dining
        Action: Auto-categorized, no manual review needed
        """,
        success=True,
        tags=["categorization", "starbucks", "auto-categorize", "high-confidence"]
    )
    
    # Transaction 2: Ambiguous case
    agent.record_task_result(
        task_id="txn-002",
        task_name="Categorize AMAZON transaction",
        result="""
        Transaction: $89.99 at AMAZON
        Prediction: Shopping (confidence: 85%)
        Reasoning: Amazon is multi-category merchant, but Shopping is most common
        Action: Flagged for manual review due to moderate confidence
        Manual Review Result: Confirmed as 'Shopping - Electronics'
        Learning: Large Amazon purchases often electronics, should boost electronics probability
        """,
        success=True,
        tags=["categorization", "amazon", "manual-review", "learning-opportunity"]
    )
    
    # Record learning from manual review
    agent.record_learning(
        learning_id="amazon-large-purchase",
        content="Amazon transactions over $50 have 70% probability of being Electronics category",
        source="manual-review-feedback",
        importance=1.7,
        tags=["amazon", "large-purchases", "electronics", "amount-based-pattern"]
    )
    
    # Transaction 3: New merchant
    agent.record_task_result(
        task_id="txn-003",
        task_name="Categorize new merchant transaction",
        result="""
        Transaction: $12.99 at 'LOCAL COFFEE SHOP'
        Prediction: Food & Dining (confidence: 60%)
        Reasoning: Contains 'COFFEE' keyword, similar to known coffee merchants
        Action: Auto-categorized with moderate confidence
        User Feedback: Correct categorization
        Learning: Coffee shop merchants (even local) follow Food & Dining pattern
        """,
        success=True,
        tags=["categorization", "new-merchant", "keyword-matching", "user-feedback"]
    )
    
    # Record learning from new merchant
    agent.record_learning(
        learning_id="coffee-shop-keywords",
        content="Merchants containing 'COFFEE', 'CAFE', 'ESPRESSO' keywords typically Food & Dining",
        source="new-merchant-analysis",
        importance=1.4,
        tags=["keyword-matching", "coffee-shops", "food-dining", "merchant-classification"]
    )
    
    # Update goal progress
    agent.update_goal_progress("improve-categorization-accuracy", 0.6)
    agent.update_goal_progress("reduce-manual-review", 0.4)
    
    # Simulate more transactions over time
    print("\n[CONTINUED CATEGORIZATION] Processing more transactions...")
    
    # More transactions with learned patterns
    agent.record_task_result(
        task_id="txn-004",
        task_name="Categorize SHELL gas station transaction",
        result="""
        Transaction: $45.00 at 'SHELL OIL'
        Prediction: Gas & Transportation (confidence: 95%)
        Reasoning: Shell is known gas station brand, matches gas station pattern
        Action: Auto-categorized with high confidence
        """,
        success=True,
        tags=["categorization", "shell", "gas-station", "high-confidence"]
    )
    
    agent.record_task_result(
        task_id="txn-005",
        task_name="Categorize MICHAELS craft store transaction",
        result="""
        Transaction: $23.45 at 'MICHAELS CRAFT'
        Prediction: Hobbies & Recreation (confidence: 85%)
        Reasoning: Matches user's hobby pattern, Michaels is craft store
        Action: Auto-categorized based on user pattern
        """,
        success=True,
        tags=["categorization", "michaels", "hobbies", "user-pattern"]
    )
    
    # Update progress
    agent.update_goal_progress("improve-categorization-accuracy", 0.8)
    agent.update_goal_progress("reduce-manual-review", 0.7)
    
    # Show agent context for categorization decisions
    print("\n[CONTEXT] Agent context for categorization decisions:")
    context = agent.get_agent_context(
        task_type="categorization",
        token_budget=1500
    )
    print(context)
    
    # Show categorization patterns learned
    print("\n[PATTERNS] Learned categorization patterns:")
    patterns = agent.ctx.get_context(
        include_tags=["merchant-pattern", "user-pattern"],
        token_budget=1000,
        summarize_if_needed=True
    )
    print(patterns)
    
    # Show agent stats
    print("\n[STATS] Agent statistics:")
    stats = agent.get_agent_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Show accuracy metrics (simulated)
    print("\n[ACCURACY] Categorization performance:")
    print("  Current Accuracy: 92.3%")
    print("  Auto-categorization Rate: 78.5%")
    print("  Manual Review Rate: 21.5%")
    print("  Learning Events: 5 new patterns identified")

if __name__ == "__main__":
    simulate_bank_categorizer()
