import pytest
import os
import random
import numpy as np
import matplotlib
import sys
import shutil

# Add root directory and src directory to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "src"))

from agents.neural_agent import NeuralEthicalAgent
from scenarios.scenarios import get_trolley_problem
from society.neural_society import NeuralEthicalSociety
from visualization.visualization import EthicalSimulationVisualizer

def test_visualizations(tmp_path):
    """Test visualization generation."""
    matplotlib.use("Agg")
    
    # Create Test Data
    agents = []
    for i in range(6):
        agent = NeuralEthicalAgent(f"test_agent_{i+1}")
        agents.append(agent)
        
    society = NeuralEthicalSociety()
    for agent in agents:
        society.add_agent(agent)

    scenario = get_trolley_problem()
    society.add_scenario(scenario)

    # Simulate Decisions
    decisions = {}
    options = list(scenario.options.keys())

    for agent in agents:
        decision = random.choice(options)
        confidence = random.uniform(0.3, 0.9)

        decisions[agent.agent_id] = {
            "decision": decision,
            "confidence": confidence,
            "processing_type": agent.cognitive_architecture.primary_processing,
            "personality": agent.personality_traits,
        }

    visualizer = EthicalSimulationVisualizer()
    
    # Use tmp_path for output
    output_dir = str(tmp_path)
    
    # Test 1: Personality Plot
    visualizer.plot_agent_personalities(agents, os.path.join(output_dir, "test_personalities.png"))
    assert os.path.exists(os.path.join(output_dir, "test_personalities.png"))

    # Test 2: Network Plot
    visualizer.plot_social_network(society, os.path.join(output_dir, "test_network.png"))
    assert os.path.exists(os.path.join(output_dir, "test_network.png"))

    # Test 3: Decision Plot
    visualizer.plot_scenario_decisions(scenario, decisions, os.path.join(output_dir, "test_decisions.png"))
    assert os.path.exists(os.path.join(output_dir, "test_decisions.png"))

    # Test 4: Dashboard
    visualizer.create_simulation_dashboard(society, scenario, decisions, os.path.join(output_dir, "test_dashboard.png"))
    assert os.path.exists(os.path.join(output_dir, "test_dashboard.png"))

    # Test 5: Batch Save
    # save_all_plots creates a directory in the current working directory usually.
    # We can try to run it and check if it returns a path.
    saved_dir = visualizer.save_all_plots(society, scenario, decisions)
    assert os.path.exists(saved_dir)
    assert os.path.isdir(saved_dir)
    
    # Cleanup the created directory
    if os.path.exists(saved_dir):
        shutil.rmtree(saved_dir)
