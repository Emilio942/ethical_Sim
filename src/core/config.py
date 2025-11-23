from dataclasses import dataclass, field

@dataclass
class CognitiveConfig:
    """Konfiguration für die kognitive Architektur."""
    # Systematic Processing
    BIAS_REDUCTION_FACTOR: float = 0.7

    # Intuitive Processing
    AVAILABILITY_BIAS_WEIGHT: float = 0.5

    # Associative Processing
    ACTIVATION_THRESHOLD: float = 0.3
    ASSOCIATION_DAMPING_FACTOR: float = 0.4

    # Analogical Processing
    ANALOGY_STRENGTH: float = 0.5
    ANALOGY_CONTEXT_WEIGHT: float = 0.1

    # Emotional Processing
    DEFAULT_EMOTIONAL_REACTIVITY: float = 0.5
    DEFAULT_EMOTIONAL_REGULATION: float = 0.5
    DEFAULT_NEGATIVITY_BIAS: float = 0.6
    EMOTIONAL_REGULATION_DAMPING: float = 0.5

    # Narrative Processing
    COHERENCE_BIAS: float = 0.3

    # Bayesian Update
    SYSTEMATIC_EVIDENCE_WEIGHT: float = 0.7
    EMOTIONAL_EVIDENCE_WEIGHT: float = 0.3
    NEUTRAL_EVIDENCE_WEIGHT: float = 0.5
    ANCHORING_EFFECT_WEIGHT: float = 0.2

@dataclass
class SimulationConfig:
    """Zentrale Konfiguration für die Simulation."""
    cognitive: CognitiveConfig = field(default_factory=CognitiveConfig)
    
    # Globale Simulationseinstellungen
    RANDOM_SEED: int = 42
    LOG_LEVEL: str = "INFO"

# Globale Instanz
config = SimulationConfig()
