from typing import Dict

class EthicalScenario:
    """Repräsentiert ein ethisches Szenario oder Dilemma."""
    
    def __init__(self, scenario_id: str, description: str, 
                 relevant_beliefs: Dict[str, float],
                 options: Dict[str, Dict[str, float]],
                 option_attributes: Dict[str, Dict[str, float]] = None,
                 outcome_feedback: Dict[str, Dict[str, float]] = None,
                 moral_implications: Dict[str, Dict[str, float]] = None):
        """
        Initialisiert ein ethisches Szenario.
        
        Args:
            scenario_id: Eindeutige ID des Szenarios
            description: Beschreibung des Szenarios
            relevant_beliefs: Dictionary mit relevanten Überzeugungen und ihrer Relevanz
            options: Dictionary mit Optionen und ihren Auswirkungen auf Überzeugungen
            option_attributes: Zusätzliche Attribute für jede Option (z.B. Risiko)
            outcome_feedback: Feedback für jede Option, wie sie Überzeugungen beeinflusst
            moral_implications: Implikationen für moralische Grundlagen pro Option
        """
        self.scenario_id = scenario_id
        self.description = description
        self.relevant_beliefs = relevant_beliefs
        self.options = options
        self.option_attributes = option_attributes or {}
        self.outcome_feedback = outcome_feedback or {}
        self.moral_implications = moral_implications or {}
        
        # Emotionale Valenz des Szenarios (falls relevant)
        self.emotional_valence = 0.0
        
        # Narrative Elemente (für narrative Verarbeitung)
        self.narrative_elements = {
            "characters": [],
            "conflict": "",
            "context": "",
            "coherence": 0.7  # Wie kohärent/nachvollziehbar das Szenario ist (0-1)
        }