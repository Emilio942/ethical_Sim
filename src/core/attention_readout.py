import numpy as np
from typing import Dict, List, Tuple, Optional
from core.logger import logger

class AttentionReadout:
    """
    Implements an Attention-Based Readout mechanism for the Auditory Cortex.
    Inspired by Brain-IT (Zalcher et al., 2025), this module uses learnable query tokens
    (representing emotion prototypes) to attend to specific spectro-temporal features
    from the auditory expert modules.
    
    Phase 5.3 of the Migration Plan.
    """
    def __init__(self, input_dim: int, num_queries: int, embed_dim: int = 64):
        """
        Args:
            input_dim: Dimension of the input features (e.g., number of frequency channels).
            num_queries: Number of emotion classes (e.g., 7 for EMO-DB).
            embed_dim: Dimension of the internal attention space.
        """
        self.input_dim = input_dim
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        
        # Learnable Query Tokens (Emotion Prototypes)
        # Shape: (num_queries, embed_dim)
        # e.g., Row 0 = "Anger Prototype", Row 1 = "Sadness Prototype"
        self.queries = np.random.randn(num_queries, embed_dim) * 0.01
        
        # Linear Projections for Keys and Values (from Auditory Features)
        # We assume input features are projected to embed_dim
        self.W_k = np.random.randn(input_dim, embed_dim) * 0.01
        self.W_v = np.random.randn(input_dim, embed_dim) * 0.01
        
        # Output projection (optional, usually to logits)
        # Here we just use the attention output directly or project to scalar score
        self.W_o = np.random.randn(embed_dim, 1) * 0.01
        
    def softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Stable softmax implementation."""
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)
        
    def forward(self, auditory_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs Multi-Head Attention (simplified to Single-Head here for simulation).
        
        Args:
            auditory_features: Shape (time_steps, input_dim) or (num_modules * time, input_dim)
                               The "Context" or "Memory" for attention.
                               
        Returns:
            scores: Shape (num_queries,) - Probability/Score for each emotion.
            attention_weights: Shape (num_queries, time_steps) - What the model looked at.
        """
        # 1. Project Input to Keys and Values
        # K = X * W_k
        K = np.dot(auditory_features, self.W_k) # (time, embed_dim)
        V = np.dot(auditory_features, self.W_v) # (time, embed_dim)
        
        # 2. Calculate Attention Scores
        # Score = Q * K^T / sqrt(d)
        # Q: (num_queries, embed_dim)
        # K^T: (embed_dim, time)
        # Result: (num_queries, time)
        d_k = self.embed_dim
        scores_raw = np.dot(self.queries, K.T) / np.sqrt(d_k)
        
        # 3. Apply Softmax to get Attention Weights
        attention_weights = self.softmax(scores_raw, axis=1) # (num_queries, time)
        
        # 4. Weighted Sum of Values
        # Context = Weights * V
        # (num_queries, time) * (time, embed_dim) -> (num_queries, embed_dim)
        context = np.dot(attention_weights, V)
        
        # 5. Final Prediction (Project to scalar score/logit)
        # (num_queries, embed_dim) * (embed_dim, 1) -> (num_queries, 1)
        logits = np.dot(context, self.W_o).flatten()
        
        # Optional: Apply softmax over emotions if we want probabilities
        emotion_probs = self.softmax(logits)
        
        return emotion_probs, attention_weights

    def update_weights(self, auditory_features: np.ndarray, target_idx: int, learning_rate: float = 0.01):
        """
        Simple gradient descent step (simulation) to train the readout.
        In a real framework like PyTorch, autograd handles this.
        Here we implement a simple analytic update or Hebbian-like rule.
        
        For simulation purposes, we'll just nudge the Query vector of the target emotion
        towards the average Key vector of the input (Prototype Learning).
        """
        # Project input to embedding space
        projected_input = np.dot(auditory_features, self.W_k)
        
        # Average feature vector (Global Average Pooling equivalent)
        avg_feature = np.mean(projected_input, axis=0)
        
        # Move the target query closer to this input representation
        # Q_target = Q_target + lr * (Input_Avg - Q_target)
        self.queries[target_idx] += learning_rate * (avg_feature - self.queries[target_idx])
        
        # Normalize to prevent explosion
        norm = np.linalg.norm(self.queries[target_idx])
        if norm > 0:
            self.queries[target_idx] /= norm
