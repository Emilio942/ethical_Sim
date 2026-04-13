
import numpy as np
import networkx as nx
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from core.cognitive_architecture import QuantumZenoArchitecture, EthicalScheme
from core.beliefs import NeuralEthicalBelief
from agents.neural_agent import NeuralEthicalAgent
from society.neural_society import NeuralSociety
from analysis.metrics import SocialDynamicsMetrics

def test_math_frontiers():
    print("🚀 Starting Mathematical Frontiers Test...")
    
    # 1. Test Quantum Zeno Architecture
    print("\n--- 1. Quantum Zeno Test ---")
    qze = QuantumZenoArchitecture(num_moral_dimensions=3, measurement_frequency=0.8)
    inputs = {f"opt_{i}": 0.1 for i in range(8)}
    context = {"forbidden_indices": [2, 3]}
    results = qze.process_information(inputs, context)
    print(f"  - QZE Probability for forbidden opt_2: {results['opt_2']:.4f}")
    
    # 2. Test Society & RMT Leader Detection
    print("\n--- 2. RMT & Society Test ---")
    soc = NeuralSociety(num_agents=10)
    for aid, agent in soc.agents.items():
        agent.add_belief(NeuralEthicalBelief("Justice", "Value", 0.7, 0.8, 0.5))
    
    # Add some structure to the network
    soc.social_network.add_edges_from([(0, i) for i in range(1, 6)]) # Node 0 is a hub
    leaders = soc.detect_moral_leaders_rmt()
    print(f"  - RMT Detected Leaders: {leaders}")
    
    # 3. Test Entropy & Work
    print("\n--- 3. Thermodynamics Test ---")
    entropy = SocialDynamicsMetrics.calculate_ethical_entropy(soc)
    b_init = np.array([0.5, 0.5])
    b_final = np.array([0.9, 0.1])
    work = SocialDynamicsMetrics.calculate_ethical_work(b_init, b_final)
    print(f"  - Society Ethical Entropy: {entropy:.4f}")
    print(f"  - Ethical Work (Jarzynski): {work:.4f}")
    
    # 4. Test p-adic Distance
    print("\n--- 4. p-adic Analysis Test ---")
    d_padic = SocialDynamicsMetrics.padic_moral_distance(b_init, b_final, p=2)
    print(f"  - p-adic Moral Distance (p=2): {d_padic:.4f}")
    
    # 5. Test Ethical Schemes & Localization
    print("\n--- 5. Scheme Theory Test ---")
    scheme = EthicalScheme(axioms=["Non-Harm", "Fairness"])
    local_truths = scheme.localize_at("Fairness")
    print(f"  - Localization at 'Fairness': {local_truths}")
    
    # 6. Test Mapping Cone Synthesis
    print("\n--- 6. Derived Categories Test ---")
    norm_a = np.array([1.0, 0.0])
    norm_b = np.array([0.0, 1.0])
    synthesis = scheme.resolve_conflict_via_cone(norm_a, norm_b)
    print(f"  - Mapping Cone Synthesis (Norm + Meta-Norm): {synthesis}")
    
    # 7. Test Graph De-noising (GFT)
    print("\n--- 7. Graph Signal Processing Test ---")
    denoised = SocialDynamicsMetrics.graph_belief_denoising(soc)
    print(f"  - Denoised Belief for Agent 0: {denoised.get(0, 'N/A')}")
    
    # 8. Test Cohomological Obstructions
    print("\n--- 8. Topology Test ---")
    # Force a 'Moral Twist' in a cycle
    # (Simplified test: checking if the logic executes)
    obstructions = SocialDynamicsMetrics.detect_cohomological_obstruction(soc)
    print(f"  - H1 Obstruction Count: {obstructions}")

    print("\n✅ All Mathematical Frontier Tests Completed Successfully!")

if __name__ == "__main__":
    test_math_frontiers()
