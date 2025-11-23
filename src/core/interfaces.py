from typing import Protocol, Dict, List, Optional, Any, runtime_checkable

@runtime_checkable
class EthicalAgent(Protocol):
    """
    Protokoll für einen ethischen Agenten.
    Definiert die minimale Schnittstelle, die ein Agent implementieren muss.
    """
    agent_id: str
    beliefs: Dict[str, Any]
    decision_history: List[Any]
    
    def evaluate_scenario(self, scenario: Any) -> Dict[str, Any]:
        """Bewertet ein ethisches Szenario."""
        ...
        
    def make_decision(self, scenario: Any) -> str:
        """Trifft eine Entscheidung in einem Szenario."""
        ...
        
    def update_belief(self, belief_name: str, new_strength: float, **kwargs):
        """Aktualisiert eine Überzeugung."""
        ...
