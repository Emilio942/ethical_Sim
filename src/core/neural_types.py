import random


class NeuralProcessingType:
    """Repräsentiert unterschiedliche neuronale Verarbeitungsstile."""

    SYSTEMATIC = "systematic"  # Stark analytisch, schrittweise
    INTUITIVE = "intuitive"  # Schnell, ganzheitlich, gefühlsbasiert
    ASSOCIATIVE = "associative"  # Netzwerkartig, assoziativ
    ANALOGICAL = "analogical"  # Basierend auf Analogien und Metaphern
    EMOTIONAL = "emotional"  # Stark emotionsgesteuert
    NARRATIVE = "narrative"  # Informationsverarbeitung durch Geschichten

# --- MATHEMATICAL INQUIRY 14: GRAPH SIGNAL PROCESSING ---
# Question: If we treat belief strengths as signals on a graph, can we use 
# the Graph Fourier Transform (GFT) to identify 'low-frequency' (consensus) 
# and 'high-frequency' (conflict) components? How would a 'Graph Spectral Filter' 
# effectively 'de-noise' a population during a propaganda event?

    @staticmethod
    def get_random():
        """Liefert einen zufälligen Verarbeitungstyp."""
        types = [
            NeuralProcessingType.SYSTEMATIC,
            NeuralProcessingType.INTUITIVE,
            NeuralProcessingType.ASSOCIATIVE,
            NeuralProcessingType.ANALOGICAL,
            NeuralProcessingType.EMOTIONAL,
            NeuralProcessingType.NARRATIVE,
        ]
        return random.choice(types)
