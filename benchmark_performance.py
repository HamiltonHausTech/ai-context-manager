#!/usr/bin/env python3
"""
Performance benchmark for AI Context Manager
"""

import time
import logging
import random
import string
from typing import List, Dict, Any

from ai_context_manager.simple_api import create_context_manager, create_agent_context_manager
from ai_context_manager.components import TaskSummaryComponent, LongTermMemoryComponent

logging.basicConfig(level=logging.WARNING)  # Reduce noise during benchmarking

def generate_random_text(length: int = 100) -> str:
    """Generate random text for testing."""
    words = ['AI', 'machine', 'learning', 'context', 'management', 'vector', 'database', 
             'semantic', 'search', 'agent', 'memory', 'intelligence', 'artificial', 
             'neural', 'network', 'algorithm', 'data', 'processing', 'analysis']
    return ' '.join(random.choices(words, k=length))

def benchmark_component_operations():
    """Benchmark component registration and retrieval."""
    print("🔧 Benchmarking Component Operations")
    print("=" * 50)
    
    ctx = create_context_manager()
    
    # Test 1: Component Registration
    start_time = time.time()
    components = []
    
    for i in range(1000):
        task = TaskSummaryComponent(
            id=f"task-{i}",
            task_name=f"Task {i}",
            summary=generate_random_text(50),
            tags=[f"tag-{i % 10}", "benchmark"]
        )
        ctx.ctx.register_component(task)
        components.append(task)
    
    registration_time = time.time() - start_time
    print(f"✅ Registered 1000 components in {registration_time:.3f}s ({1000/registration_time:.0f} ops/sec)")
    
    # Test 2: Context Retrieval
    start_time = time.time()
    
    for _ in range(100):
        context = ctx.get_context(tags=["benchmark"], token_budget=500)
    
    retrieval_time = time.time() - start_time
    print(f"✅ Retrieved context 100 times in {retrieval_time:.3f}s ({100/retrieval_time:.0f} ops/sec)")
    
    # Test 3: Search Operations
    start_time = time.time()
    
    for _ in range(50):
        results = ctx.search_similar("AI machine learning", limit=10)
    
    search_time = time.time() - start_time
    print(f"✅ Performed 50 searches in {search_time:.3f}s ({50/search_time:.0f} ops/sec)")
    
    return {
        "registration_ops_per_sec": 1000/registration_time,
        "retrieval_ops_per_sec": 100/retrieval_time,
        "search_ops_per_sec": 50/search_time
    }

def benchmark_agent_operations():
    """Benchmark agent-specific operations."""
    print("\\n🤖 Benchmarking Agent Operations")
    print("=" * 50)
    
    agent = create_agent_context_manager("benchmark-agent")
    
    # Test 1: Goal Management
    start_time = time.time()
    
    for i in range(100):
        goal_id = f"goal-{i}"
        goal_desc = f"Benchmark goal {i}: {generate_random_text(20)}"
        agent.add_goal(goal_id, goal_desc, priority=random.uniform(0.5, 2.0))
    
    goal_time = time.time() - start_time
    print(f"✅ Added 100 goals in {goal_time:.3f}s ({100/goal_time:.0f} ops/sec)")
    
    # Test 2: Task Recording
    start_time = time.time()
    
    for i in range(200):
        task_id = f"task-{i}"
        task_name = f"Benchmark Task {i}"
        result = f"Completed {generate_random_text(30)}"
        agent.add_task(task_id, task_name, result, success=random.choice([True, False]))
    
    task_time = time.time() - start_time
    print(f"✅ Recorded 200 tasks in {task_time:.3f}s ({200/task_time:.0f} ops/sec)")
    
    # Test 3: Learning Recording
    start_time = time.time()
    
    for i in range(150):
        learning_id = f"learning-{i}"
        content = f"Learned: {generate_random_text(40)}"
        source = f"source-{i % 10}"
        agent.add_learning(learning_id, content, source, importance=random.uniform(0.5, 2.0))
    
    learning_time = time.time() - start_time
    print(f"✅ Recorded 150 learnings in {learning_time:.3f}s ({150/learning_time:.0f} ops/sec)")
    
    # Test 4: Agent Context Retrieval
    start_time = time.time()
    
    for _ in range(50):
        context = agent.get_context("benchmark testing", token_budget=1000)
    
    context_time = time.time() - start_time
    print(f"✅ Retrieved agent context 50 times in {context_time:.3f}s ({50/context_time:.0f} ops/sec)")
    
    return {
        "goal_ops_per_sec": 100/goal_time,
        "task_ops_per_sec": 200/task_time,
        "learning_ops_per_sec": 150/learning_time,
        "agent_context_ops_per_sec": 50/context_time
    }

def benchmark_memory_usage():
    """Benchmark memory usage patterns."""
    print("\\n💾 Benchmarking Memory Usage")
    print("=" * 50)
    
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Create large context manager
    ctx = create_context_manager()
    
    # Add many components
    for i in range(5000):
        task = TaskSummaryComponent(
            id=f"large-task-{i}",
            task_name=f"Large Task {i}",
            summary=generate_random_text(100),
            tags=[f"large-{i % 50}", "memory-test"]
        )
        ctx.ctx.register_component(task)
    
    peak_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = peak_memory - initial_memory
    
    print(f"✅ Added 5000 components")
    print(f"📊 Initial memory: {initial_memory:.1f} MB")
    print(f"📊 Peak memory: {peak_memory:.1f} MB")
    print(f"📊 Memory increase: {memory_increase:.1f} MB")
    print(f"📊 Memory per component: {memory_increase/5000*1024:.1f} KB")
    
    return {
        "initial_memory_mb": initial_memory,
        "peak_memory_mb": peak_memory,
        "memory_increase_mb": memory_increase,
        "memory_per_component_kb": memory_increase/5000*1024
    }

def benchmark_scalability():
    """Benchmark scalability with different component counts."""
    print("\\n📈 Benchmarking Scalability")
    print("=" * 50)
    
    component_counts = [100, 500, 1000, 2000, 5000]
    results = {}
    
    for count in component_counts:
        print(f"\\nTesting with {count} components...")
        
        ctx = create_context_manager()
        
        # Add components
        start_time = time.time()
        for i in range(count):
            task = TaskSummaryComponent(
                id=f"scale-task-{i}",
                task_name=f"Scale Task {i}",
                summary=generate_random_text(50),
                tags=[f"scale-{i % 20}", "scalability"]
            )
            ctx.ctx.register_component(task)
        
        add_time = time.time() - start_time
        
        # Test retrieval
        start_time = time.time()
        for _ in range(10):
            context = ctx.get_context(tags=["scalability"], token_budget=500)
        
        retrieval_time = time.time() - start_time
        
        results[count] = {
            "add_time": add_time,
            "retrieval_time": retrieval_time,
            "add_ops_per_sec": count/add_time,
            "retrieval_ops_per_sec": 10/retrieval_time
        }
        
        print(f"  Added {count} components: {add_time:.3f}s ({count/add_time:.0f} ops/sec)")
        print(f"  Retrieved context 10 times: {retrieval_time:.3f}s ({10/retrieval_time:.0f} ops/sec)")
    
    return results

def main():
    """Run all benchmarks."""
    print("🚀 AI Context Manager Performance Benchmark")
    print("=" * 60)
    
    all_results = {}
    
    try:
        # Run benchmarks
        all_results["component_ops"] = benchmark_component_operations()
        all_results["agent_ops"] = benchmark_agent_operations()
        all_results["memory_usage"] = benchmark_memory_usage()
        all_results["scalability"] = benchmark_scalability()
        
        # Summary
        print("\\n📊 Performance Summary")
        print("=" * 50)
        
        print("Component Operations:")
        print(f"  Registration: {all_results['component_ops']['registration_ops_per_sec']:.0f} ops/sec")
        print(f"  Retrieval: {all_results['component_ops']['retrieval_ops_per_sec']:.0f} ops/sec")
        print(f"  Search: {all_results['component_ops']['search_ops_per_sec']:.0f} ops/sec")
        
        print("\\nAgent Operations:")
        print(f"  Goals: {all_results['agent_ops']['goal_ops_per_sec']:.0f} ops/sec")
        print(f"  Tasks: {all_results['agent_ops']['task_ops_per_sec']:.0f} ops/sec")
        print(f"  Learnings: {all_results['agent_ops']['learning_ops_per_sec']:.0f} ops/sec")
        print(f"  Context: {all_results['agent_ops']['agent_context_ops_per_sec']:.0f} ops/sec")
        
        print("\\nMemory Usage:")
        print(f"  Memory per component: {all_results['memory_usage']['memory_per_component_kb']:.1f} KB")
        print(f"  Total memory increase: {all_results['memory_usage']['memory_increase_mb']:.1f} MB")
        
        print("\\n✅ Benchmark completed successfully!")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
