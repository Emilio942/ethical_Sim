import numpy as np
from typing import Dict, Tuple

class NeuralEthicalBelief:
    """Repräsentiert eine ethische Überzeugung mit neuronaler Modellierung."""
    
    def __init__(self, name: str, category: str, initial_strength: float = 0.5,
                certainty: float = 0.5, emotional_valence: float = 0.0):
        """
        Initialisiert eine ethische Überzeugung.
        
        Args:
            name: Beschreibender Name der Überzeugung
            category: Kategorie der Überzeugung (z.B. 'Gerechtigkeit', 'Freiheit')
            initial_strength: Anfängliche Stärke der Überzeugung (0-1)
            certainty: Gewissheit über die Überzeugung (0-1)
            emotional_valence: Emotionale Ladung der Überzeugung (-1 bis +1)
        """
        self.name = name
        self.category = category
        self.strength = np.clip(initial_strength, 0.0, 1.0)
        self.certainty = np.clip(certainty, 0.0, 1.0)
        self.emotional_valence = np.clip(emotional_valence, -1.0, 1.0)
        
        # Verbindungen zu anderen Überzeugungen (Belief -> (Einfluss, Polarität))
        self.connections = {}
        
        # Aktivierungsniveau (für spreading activation)
        self.activation = 0.0
        
        # Letzter Aktivierungszeitpunkt
        self.last_activation_time = 0
        
        # Assoziative Konzepte (für associative processing)
        self.associated_concepts = {}  # concept_name -> strength
        
    def add_connection(self, belief_name: str, influence_strength: float, polarity: int):
        """Fügt eine Verbindung zu einer anderen Überzeugung hinzu."""
        self.connections[belief_name] = (np.clip(influence_strength, 0.0, 1.0), np.sign(polarity))
        
    def add_associated_concept(self, concept_name: str, association_strength: float):
        """Fügt ein assoziiertes Konzept hinzu."""
        self.associated_concepts[concept_name] = np.clip(association_strength, 0.0, 1.0)
    
    def update_strength(self, new_strength: float):
        """Aktualisiert die Stärke der Überzeugung."""
        self.strength = np.clip(new_strength, 0.0, 1.0)
        
    def update_certainty(self, new_certainty: float):
        """Aktualisiert die Gewissheit über die Überzeugung."""
        self.certainty = np.clip(new_certainty, 0.0, 1.0)
        
    def update_emotional_valence(self, new_valence: float):
        """Aktualisiert die emotionale Ladung der Überzeugung."""
        self.emotional_valence = np.clip(new_valence, -1.0, 1.0)
        
    def activate(self, activation_level: float, current_time: int):
        """Aktiviert die Überzeugung für spreading activation."""
        # Zeitlichen Abfall modellieren (je länger nicht aktiviert, desto stärker der Effekt)
        time_since_last = current_time - self.last_activation_time
        decay_factor = np.exp(-0.1 * time_since_last) if time_since_last > 0 else 1.0
        
        # Aktualisieren mit Decay
        self.activation = decay_factor * self.activation + activation_level
        self.last_activation_time = current_time
        self.last_activation_time = current_time