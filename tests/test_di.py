import sys
import os
import unittest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agents.neural_agent import NeuralEthicalAgent
from core.cognitive_architecture import CognitiveArchitecture
from core.neural_types import NeuralProcessingType

class TestDependencyInjection(unittest.TestCase):
    def test_cognitive_architecture_injection(self):
        # Create a custom architecture
        custom_arch = CognitiveArchitecture(
            primary_processing=NeuralProcessingType.SYSTEMATIC,
            secondary_processing=NeuralProcessingType.EMOTIONAL,
            processing_balance=0.9
        )
        
        # Inject it into the agent
        agent = NeuralEthicalAgent("test_agent", cognitive_architecture=custom_arch)
        
        # Verify that the agent uses the injected instance
        self.assertIs(agent.cognitive_architecture, custom_arch)
        self.assertEqual(agent.cognitive_architecture.processing_balance, 0.9)
        print("✅ Dependency Injection Test Passed!")

if __name__ == "__main__":
    unittest.main()
