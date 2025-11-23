import os
import sys
import random
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from agents.neural_agent import NeuralEthicalAgent
from core.logger import logger

def run_agent_integration_demo():
    print("=== Agent Auditory Integration Demo ===")
    
    # 1. Create Agent
    agent = NeuralEthicalAgent(agent_id="TestAgent_01")
    print(f"Agent created: {agent.agent_id}")
    
    # 2. Initialize Auditory System
    print("Initializing Auditory System...")
    agent.initialize_auditory_system()
    
    if not agent.emotion_model:
        print("Error: Emotion model not loaded. Run demo_full_pipeline.py first.")
        return

    # 3. Pick a random audio file
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/emodb'))
    if not os.path.exists(data_dir):
        print("Error: Data directory not found.")
        return
    
    # Walk through subdirectories to find wav files
    all_wavs = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.wav'):
                all_wavs.append(os.path.join(root, file))
                
    if not all_wavs:
        print("Error: No wav files found.")
        return
        
    test_file = random.choice(all_wavs)
    print(f"\nTesting with file: {os.path.basename(test_file)}")
    print(f"True Label (from folder): {os.path.basename(os.path.dirname(test_file))}")
    
    # 4. Agent Perceives Audio
    print("Agent is listening...")
    perception = agent.perceive_audio(test_file)
    
    print("\n=== Perception Result ===")
    print(f"Emotion: {perception['emotion']}")
    print(f"Confidence: {perception['confidence']:.4f}")
    print("\nProbabilities:")
    for emo, prob in perception['probabilities'].items():
        print(f"  {emo}: {prob:.4f}")

if __name__ == "__main__":
    run_agent_integration_demo()
