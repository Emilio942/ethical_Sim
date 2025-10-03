import random


class NeuralProcessingType:
    """Repräsentiert unterschiedliche neuronale Verarbeitungsstile."""

    SYSTEMATIC = "systematic"  # Stark analytisch, schrittweise
    INTUITIVE = "intuitive"  # Schnell, ganzheitlich, gefühlsbasiert
    ASSOCIATIVE = "associative"  # Netzwerkartig, assoziativ
    ANALOGICAL = "analogical"  # Basierend auf Analogien und Metaphern
    EMOTIONAL = "emotional"  # Stark emotionsgesteuert
    NARRATIVE = "narrative"  # Informationsverarbeitung durch Geschichten

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
