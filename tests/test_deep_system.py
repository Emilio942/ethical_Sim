import pytest
import gc
import psutil
import time
import threading
import random
import math
import sys
import os

# Add root directory and src directory to sys.path to allow imports
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "src"))

from society.neural_society import NeuralEthicalSociety
from agents.neural_agent import NeuralEthicalAgent
from scenarios.scenarios import get_trolley_problem

def test_memory_leaks():
    """Test for memory leaks during repeated operations."""
    gc.collect()
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    societies = []

    # Create and destroy 100 Societies
    for i in range(100):
        society = NeuralEthicalSociety()
        for j in range(5):
            agent = NeuralEthicalAgent(f"agent_{i}_{j}")
            society.add_agent(agent)
        societies.append(society)

        # Cleanup every 20 iterations
        if i % 20 == 0:
            societies.clear()
            gc.collect()

    # Final Memory Check
    societies.clear()
    gc.collect()
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory

    # Memory Leak Test: Not more than 10MB increase
    assert memory_increase < 10, f"Excessive memory increase: {memory_increase:.2f}MB"

def test_concurrent_access():
    """Test for thread safety and concurrent access."""
    society = NeuralEthicalSociety()

    # Add initial agents
    for i in range(5):
        agent = NeuralEthicalAgent(f"concurrent_agent_{i}")
        society.add_agent(agent)

    scenario = get_trolley_problem()
    errors = []
    results = []

    def worker_thread(thread_id: int):
        """Worker Thread for Concurrent Testing"""
        try:
            # Various concurrent operations
            for i in range(10):
                # Make decisions
                agent_keys = list(society.agents.keys())
                if agent_keys:
                    agent = society.agents[random.choice(agent_keys)]
                    decision = agent.make_decision(scenario)
                    results.append((thread_id, i, decision))

                # Short pause to encourage race conditions
                time.sleep(0.001)

        except Exception as e:
            errors.append((thread_id, e))

    # Start 5 concurrent threads
    threads = []
    for i in range(5):
        thread = threading.Thread(target=worker_thread, args=(i,))
        threads.append(thread)
        thread.start()

    # Wait for all threads
    for thread in threads:
        thread.join()

    # Evaluation
    assert len(errors) == 0, f"Errors occurred: {errors}"
    assert len(results) == 50, f"Expected 50 results, got {len(results)}"

def test_edge_cases_data_corruption():
    """Test edge cases and data corruption."""
    
    # Test 1: Empty/Null Inputs
    with pytest.raises(Exception):
         NeuralEthicalAgent("")

    # Test 2: Extreme Values
    agent = NeuralEthicalAgent("test_agent")

    class MockBelief:
        def __init__(self):
            self.name = "extreme_belief"
            self.strength = float("inf")
            self.confidence = -999999
            self.certainty = 0.5
            self.activation = 0.0
            self.connections = {}
            self.associated_concepts = {}

        def activate(self, level, time):
            self.activation = max(0.0, min(1.0, level))

        def update_certainty(self, new_certainty):
            self.certainty = max(0.0, min(1.0, new_certainty))

        def update_strength(self, new_strength):
            self.strength = (
                max(0.0, min(1.0, new_strength))
                if not math.isinf(new_strength)
                else 1.0
            )

    agent.add_belief(MockBelief())
    scenario = get_trolley_problem()
    decision = agent.make_decision(scenario)

    confidence = decision.get("confidence", 0)
    assert 0 <= confidence <= 1, f"Invalid confidence: {confidence}"

    # Test 3: Invalid Data Types
    society = NeuralEthicalSociety()
    with pytest.raises(Exception):
        society.add_agent("not_an_agent")

    # Test 4: Circular References
    agent1 = NeuralEthicalAgent("agent1")
    agent2 = NeuralEthicalAgent("agent2")

    agent1.social_connections = {"agent2": 1.0}
    agent2.social_connections = {"agent1": 1.0}

    society.add_agent(agent1)
    society.add_agent(agent2)

    decision1 = agent1.make_decision(scenario)
    decision2 = agent2.make_decision(scenario)
    
    assert decision1 is not None
    assert decision2 is not None

def test_performance_under_load():
    """Test performance under load."""
    society = NeuralEthicalSociety()

    # Test 1: Mass Agent Creation
    start_time = time.time()
    for i in range(100):
        agent = NeuralEthicalAgent(f"load_agent_{i}")
        agent.personality_traits = {
            "openness": random.random(),
            "conscientiousness": random.random(),
            "extraversion": random.random(),
            "agreeableness": random.random(),
            "neuroticism": random.random(),
        }
        society.add_agent(agent)
    
    creation_time = time.time() - start_time
    assert creation_time < 5.0, f"Agent creation too slow: {creation_time:.2f}s"

    # Test 2: Mass Decision Making
    scenario = get_trolley_problem()
    decisions = []
    start_time = time.time()
    
    for i in range(1000):
        agent_keys = list(society.agents.keys())
        agent = society.agents[random.choice(agent_keys)]
        decision = agent.make_decision(scenario)
        decisions.append(decision)

    decision_time = time.time() - start_time
    decisions_per_second = 1000 / decision_time if decision_time > 0 else 1000
    
    assert decisions_per_second > 100, f"Decision making too slow: {decisions_per_second:.1f} decisions/second"

    # Test 3: Memory Under Load
    process = psutil.Process()
    memory_before = process.memory_info().rss / 1024 / 1024

    for i in range(500):
        agent = NeuralEthicalAgent(f"stress_agent_{i}")
        society.add_agent(agent)
        if i % 50 == 0:
            for j in range(10):
                decision = agent.make_decision(scenario)

    memory_after = process.memory_info().rss / 1024 / 1024
    memory_increase = memory_after - memory_before
    
    assert memory_increase < 100, f"Excessive memory use: {memory_increase:.1f}MB"

def test_data_consistency():
    """Test data integrity and consistency."""
    society = NeuralEthicalSociety()
    agent = NeuralEthicalAgent("consistency_agent")
    society.add_agent(agent)
    scenario = get_trolley_problem()

    # Test 1: Decision Consistency
    decisions = []
    for i in range(10):
        decision = agent.make_decision(scenario)
        decisions.append(decision)

    for i, decision in enumerate(decisions):
        assert isinstance(decision, dict), f"Decision {i} is not dict"
        assert "chosen_option" in decision, f"Decision {i} missing chosen_option"
        assert "confidence" in decision, f"Decision {i} missing confidence"
        confidence = decision.get("confidence", -1)
        assert 0 <= confidence <= 1, f"Invalid confidence {confidence} in decision {i}"

    # Test 2: Agent State Consistency
    # Helper to capture state
    def get_agent_state(a):
        return {
            "agent_id": a.agent_id,
            "personality_traits": dict(a.personality_traits) if hasattr(a, "personality_traits") else {},
            "beliefs_count": len(a.beliefs.beliefs) if hasattr(a.beliefs, "beliefs") else 0,
        }

    agent_state_1 = get_agent_state(agent)

    for i in range(5):
        agent.make_decision(scenario)

    agent_state_2 = get_agent_state(agent)

    assert agent_state_1["agent_id"] == agent_state_2["agent_id"], "Agent ID changed"

def test_error_handling_resilience():
    """Test error handling and system resilience."""
    
    # Test 1: Corrupted Scenario Handling
    agent = NeuralEthicalAgent("resilience_agent")
    corrupted_scenario = type(
        "CorruptedScenario",
        (),
        {
            "scenario_id": None,
            "options": [],
            "description": None,
        },
    )()

    try:
        decision = agent.make_decision(corrupted_scenario)
        # If it returns, it should be handled gracefully (e.g. not None, or specific error dict)
        # The original test accepts decision is not None OR exception.
        # If it returns None, it fails.
        if decision is None:
             pytest.fail("Returned None for corrupted scenario")
    except Exception:
        # Exception is also OK
        pass

    # Test 2: Invalid Agent State Recovery
    society = NeuralEthicalSociety()
    agent = NeuralEthicalAgent("recovery_agent")

    # Corrupt Agent Data
    agent.beliefs = None

    try:
        society.add_agent(agent)
        assert len(society.agents) == 1, "Society failed to accept agent with corrupted beliefs"
    except Exception:
        # Exception is also OK (handled invalid state)
        pass

    # Test 3: Resource Exhaustion Simulation
    big_data = []
    society = NeuralEthicalSociety()

    for i in range(50):
        agent = NeuralEthicalAgent(f"memory_agent_{i}")
        agent.large_data = [random.random() for _ in range(1000)]
        big_data.append(agent.large_data)
        society.add_agent(agent)

    assert len(society.agents) == 50, f"Lost agents under memory pressure: {len(society.agents)}/50"
    
    del big_data
    gc.collect()
