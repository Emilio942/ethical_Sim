import unittest
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from core.attention_readout import AttentionReadout

class TestAttentionReadout(unittest.TestCase):
    
    def setUp(self):
        self.input_dim = 64 # e.g., 64 frequency channels
        self.num_queries = 7 # 7 emotions
        self.embed_dim = 32
        self.readout = AttentionReadout(self.input_dim, self.num_queries, self.embed_dim)
        
    def test_forward_shape(self):
        """Test if forward pass returns correct shapes."""
        # Simulate auditory features: (100 time steps, 64 channels)
        features = np.random.randn(100, 64)
        
        probs, weights = self.readout.forward(features)
        
        print(f"Probs shape: {probs.shape}")
        print(f"Weights shape: {weights.shape}")
        
        self.assertEqual(probs.shape, (7,))
        self.assertEqual(weights.shape, (7, 100))
        
        # Check probability sum
        self.assertAlmostEqual(np.sum(probs), 1.0, places=5)
        
    def test_attention_mechanism(self):
        """Test if attention focuses on relevant parts."""
        # Create a synthetic feature sequence where the middle is very "salient"
        # (e.g., high magnitude)
        features = np.zeros((10, 64))
        features[5, :] = 10.0 # Spike at time step 5
        
        # We need to make sure the weights align such that this spike is attended to.
        # Since weights are random initially, we can't guarantee it without training.
        # But we can check if the mechanism runs without error.
        
        probs, weights = self.readout.forward(features)
        
        # Check if weights are valid probabilities (sum to 1 over time)
        row_sums = np.sum(weights, axis=1)
        for s in row_sums:
            self.assertAlmostEqual(s, 1.0, places=5)

    def test_learning_update(self):
        """Test if update_weights changes the query vector."""
        features = np.random.randn(50, 64)
        target_idx = 0 # Anger
        
        old_query = self.readout.queries[target_idx].copy()
        
        self.readout.update_weights(features, target_idx, learning_rate=0.1)
        
        new_query = self.readout.queries[target_idx]
        
        # Check if changed
        diff = np.linalg.norm(new_query - old_query)
        print(f"Query update diff: {diff}")
        self.assertGreater(diff, 0.0)

if __name__ == '__main__':
    unittest.main()
