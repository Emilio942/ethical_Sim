"""
Utility-Funktionen für die ethische Agenten-Simulation
======================================================

Dieses Modul enthält Hilfsfunktionen für Datenvalidierung, 
mathematische Berechnungen und Debugging.
"""

import json
import csv
import os
import logging
import sys
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime


# Logging Setup
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Konfiguriert das Logging-System."""
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    
    level = level_map.get(log_level.upper(), logging.INFO)
    
    # Basis-Konfiguration
    handlers = [logging.StreamHandler(sys.stdout)]
    
    # Optional: Log-Datei
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    return logging.getLogger(__name__)


# Datenvalidierung
def validate_probability(value: Union[int, float], name: str = "value") -> float:
    """Validiert, dass ein Wert eine gültige Wahrscheinlichkeit (0-1) ist."""
    try:
        val = float(value)
        if not 0.0 <= val <= 1.0:
            raise ValueError(f"{name} muss zwischen 0 und 1 liegen, erhielt: {val}")
        return val
    except (TypeError, ValueError) as e:
        raise ValueError(f"Ungültiger Wahrscheinlichkeitswert für {name}: {value}") from e


def validate_agent_id(agent_id: str) -> str:
    """Validiert eine Agent-ID."""
    if not isinstance(agent_id, str):
        raise TypeError("Agent-ID muss ein String sein")
    
    if not agent_id.strip():
        raise ValueError("Agent-ID darf nicht leer sein")
    
    # Erlaubte Zeichen: Buchstaben, Zahlen, Unterstrich, Bindestrich
    allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    if not all(c in allowed_chars for c in agent_id):
        raise ValueError("Agent-ID enthält ungültige Zeichen. Erlaubt: a-z, A-Z, 0-9, _, -")
    
    return agent_id.strip()


def validate_personality_traits(traits: Dict[str, float]) -> Dict[str, float]:
    """Validiert Persönlichkeitsmerkmale (Big Five)."""
    required_traits = ["openness", "conscientiousness", "extroversion", "agreeableness", "neuroticism"]
    
    if not isinstance(traits, dict):
        raise TypeError("Persönlichkeitsmerkmale müssen ein Dictionary sein")
    
    # Prüfe, ob alle erforderlichen Traits vorhanden sind
    missing_traits = [trait for trait in required_traits if trait not in traits]
    if missing_traits:
        raise ValueError(f"Fehlende Persönlichkeitsmerkmale: {missing_traits}")
    
    # Validiere jeden Wert
    validated_traits = {}
    for trait in required_traits:
        validated_traits[trait] = validate_probability(traits[trait], f"personality.{trait}")
    
    return validated_traits


def validate_belief_connections(connections: Dict[str, Tuple[float, int]]) -> Dict[str, Tuple[float, int]]:
    """Validiert Überzeugungsverbindungen."""
    if not isinstance(connections, dict):
        raise TypeError("Verbindungen müssen ein Dictionary sein")
    
    validated_connections = {}
    for belief_name, (strength, polarity) in connections.items():
        if not isinstance(belief_name, str) or not belief_name.strip():
            raise ValueError("Überzeugungsname muss ein nicht-leerer String sein")
        
        validated_strength = validate_probability(strength, f"connection.{belief_name}.strength")
        
        if polarity not in [-1, 0, 1]:
            raise ValueError(f"Polarität muss -1, 0 oder 1 sein, erhielt: {polarity}")
        
        validated_connections[belief_name.strip()] = (validated_strength, int(polarity))
    
    return validated_connections


# Mathematische Hilfsfunktionen
def sigmoid(x: float, steepness: float = 1.0) -> float:
    """Sigmoid-Funktion für neuronale Aktivierung."""
    import math
    try:
        return 1.0 / (1.0 + math.exp(-steepness * x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def tanh_activation(x: float, steepness: float = 1.0) -> float:
    """Tanh-Aktivierungsfunktion."""
    import math
    try:
        return math.tanh(steepness * x)
    except OverflowError:
        return -1.0 if x < 0 else 1.0


def linear_interpolation(x: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """Lineare Interpolation zwischen zwei Punkten."""
    if x2 == x1:
        return y1
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)


def weighted_average(values: List[float], weights: List[float]) -> float:
    """Berechnet den gewichteten Durchschnitt."""
    if len(values) != len(weights):
        raise ValueError("Anzahl der Werte und Gewichte muss gleich sein")
    
    if not values:
        return 0.0
    
    total_weight = sum(weights)
    if total_weight == 0:
        return sum(values) / len(values)  # Fallback: ungewichteter Durchschnitt
    
    return sum(v * w for v, w in zip(values, weights)) / total_weight


def normalize_dict_values(data: Dict[str, float]) -> Dict[str, float]:
    """Normalisiert die Werte eines Dictionaries so, dass sie zwischen 0 und 1 liegen."""
    if not data:
        return {}
    
    min_val = min(data.values())
    max_val = max(data.values())
    
    if max_val == min_val:
        return {k: 0.5 for k in data.keys()}  # Alle Werte gleich
    
    return {k: (v - min_val) / (max_val - min_val) for k, v in data.items()}


def calculate_entropy(probabilities: List[float]) -> float:
    """Berechnet die Entropie einer Wahrscheinlichkeitsverteilung."""
    import math
    
    if not probabilities:
        return 0.0
    
    # Entferne Nullwerte (log(0) ist undefiniert)
    non_zero_probs = [p for p in probabilities if p > 0]
    
    if not non_zero_probs:
        return 0.0
    
    return -sum(p * math.log2(p) for p in non_zero_probs)


# Datenexport und -import
def export_agent_data(agents: List[Any], filename: str, format: str = "json") -> bool:
    """Exportiert Agent-Daten in verschiedenen Formaten."""
    try:
        if format.lower() == "json":
            return export_agents_to_json(agents, filename)
        elif format.lower() == "csv":
            return export_agents_to_csv(agents, filename)
        else:
            raise ValueError(f"Unbekanntes Format: {format}")
    except Exception as e:
        logging.error(f"Fehler beim Exportieren: {e}")
        return False


def export_agents_to_json(agents: List[Any], filename: str) -> bool:
    """Exportiert Agent-Daten als JSON."""
    try:
        data = []
        for agent in agents:
            agent_data = {
                "agent_id": agent.agent_id,
                "personality_traits": agent.personality_traits,
                "beliefs": {name: {
                    "strength": belief.strength,
                    "certainty": belief.certainty,
                    "emotional_valence": belief.emotional_valence
                } for name, belief in agent.beliefs.items()},
                "decision_history": agent.decision_history
            }
            data.append(agent_data)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Agent-Daten erfolgreich nach {filename} exportiert")
        return True
        
    except Exception as e:
        logging.error(f"Fehler beim JSON-Export: {e}")
        return False


def export_agents_to_csv(agents: List[Any], filename: str) -> bool:
    """Exportiert Agent-Daten als CSV."""
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            header = ["agent_id", "openness", "conscientiousness", "extroversion", 
                     "agreeableness", "neuroticism", "num_beliefs", "num_decisions"]
            writer.writerow(header)
            
            # Daten
            for agent in agents:
                row = [
                    agent.agent_id,
                    agent.personality_traits.get("openness", 0),
                    agent.personality_traits.get("conscientiousness", 0),
                    agent.personality_traits.get("extroversion", 0),
                    agent.personality_traits.get("agreeableness", 0),
                    agent.personality_traits.get("neuroticism", 0),
                    len(agent.beliefs),
                    len(agent.decision_history)
                ]
                writer.writerow(row)
        
        logging.info(f"Agent-Daten erfolgreich nach {filename} exportiert")
        return True
        
    except Exception as e:
        logging.error(f"Fehler beim CSV-Export: {e}")
        return False


def load_agents_from_json(filename: str) -> Optional[List[Dict]]:
    """Lädt Agent-Daten aus einer JSON-Datei."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logging.info(f"Agent-Daten erfolgreich aus {filename} geladen")
        return data
        
    except Exception as e:
        logging.error(f"Fehler beim JSON-Import: {e}")
        return None


# Debugging und Analyse
def print_agent_summary(agent: Any, detailed: bool = False):
    """Druckt eine Zusammenfassung eines Agenten."""
    print(f"\n🤖 Agent: {agent.agent_id}")
    print("-" * 40)
    
    # Persönlichkeitsmerkmale
    print("Persönlichkeit:")
    for trait, value in agent.personality_traits.items():
        bar = "█" * int(value * 10) + "░" * (10 - int(value * 10))
        print(f"  {trait:<15} [{bar}] {value:.2f}")
    
    # Kognitive Architektur
    print(f"\nKognitive Architektur:")
    print(f"  Primär: {agent.cognitive_architecture.primary_processing}")
    print(f"  Sekundär: {agent.cognitive_architecture.secondary_processing}")
    print(f"  Balance: {agent.cognitive_architecture.processing_balance:.2f}")
    
    # Überzeugungen
    print(f"\nÜberzeugungen: {len(agent.beliefs)}")
    if detailed and agent.beliefs:
        for name, belief in list(agent.beliefs.items())[:5]:  # Zeige nur die ersten 5
            print(f"  {name}: Stärke={belief.strength:.2f}, Gewissheit={belief.certainty:.2f}")
        if len(agent.beliefs) > 5:
            print(f"  ... und {len(agent.beliefs) - 5} weitere")
    
    # Entscheidungshistorie
    print(f"Entscheidungen: {len(agent.decision_history)}")


def print_society_summary(society: Any):
    """Druckt eine Zusammenfassung einer Gesellschaft."""
    print(f"\n🏛️ Gesellschaft")
    print("=" * 40)
    print(f"Agenten: {len(society.agents)}")
    print(f"Szenarien: {len(society.scenarios)}")
    print(f"Netzwerk-Knoten: {society.social_network.number_of_nodes()}")
    print(f"Netzwerk-Kanten: {society.social_network.number_of_edges()}")
    
    # Verarbeitungstypen-Verteilung
    processing_types = {}
    for agent in society.agents.values():
        ptype = agent.cognitive_architecture.primary_processing
        processing_types[ptype] = processing_types.get(ptype, 0) + 1
    
    print(f"\nVerarbeitungstypen:")
    for ptype, count in processing_types.items():
        percentage = (count / len(society.agents)) * 100
        print(f"  {ptype:<12} {count:>2} ({percentage:>5.1f}%)")


def create_simulation_report(society: Any, filename: str = None) -> str:
    """Erstellt einen detaillierten Simulationsbericht."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
Ethische Agenten Simulation - Bericht
=====================================
Erstellt: {timestamp}

GESELLSCHAFTSSTATISTIKEN
------------------------
Anzahl Agenten: {len(society.agents)}
Anzahl Szenarien: {len(society.scenarios)}
Netzwerk-Dichte: {society.social_network.number_of_edges() / max(1, society.social_network.number_of_nodes())}

AGENT-ANALYSE
------------
"""
    
    # Analysiere Agenten
    personality_stats = {trait: [] for trait in ["openness", "conscientiousness", "extroversion", "agreeableness", "neuroticism"]}
    
    for agent in society.agents.values():
        for trait, value in agent.personality_traits.items():
            if trait in personality_stats:
                personality_stats[trait].append(value)
    
    for trait, values in personality_stats.items():
        if values:
            avg = sum(values) / len(values)
            report += f"{trait.capitalize():<15}: Durchschnitt = {avg:.3f}\n"
    
    # Speichere Bericht falls Dateiname angegeben
    if filename:
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report)
            logging.info(f"Bericht gespeichert: {filename}")
        except Exception as e:
            logging.error(f"Fehler beim Speichern des Berichts: {e}")
    
    return report


# Konfiguration und Einstellungen
def load_config(config_file: str = "config.json") -> Dict[str, Any]:
    """Lädt Konfigurationseinstellungen."""
    default_config = {
        "simulation": {
            "random_seed": 42,
            "max_agents": 100,
            "max_scenarios": 50
        },
        "logging": {
            "level": "INFO",
            "file": None
        },
        "export": {
            "auto_export": False,
            "export_format": "json"
        }
    }
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
            
            # Merge mit Default-Konfiguration
            def merge_config(default, user):
                for key, value in user.items():
                    if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                        merge_config(default[key], value)
                    else:
                        default[key] = value
            
            merge_config(default_config, user_config)
            logging.info(f"Konfiguration aus {config_file} geladen")
            
        except Exception as e:
            logging.warning(f"Fehler beim Laden der Konfiguration: {e}. Verwende Standard-Konfiguration.")
    
    return default_config


def save_config(config: Dict[str, Any], config_file: str = "config.json") -> bool:
    """Speichert Konfigurationseinstellungen."""
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        logging.info(f"Konfiguration gespeichert: {config_file}")
        return True
    except Exception as e:
        logging.error(f"Fehler beim Speichern der Konfiguration: {e}")
        return False


# Hauptfunktion für Tests
if __name__ == "__main__":
    # Test der Utility-Funktionen
    print("🔧 Teste Utility-Funktionen...")
    
    # Validierung testen
    try:
        validate_probability(0.5, "test")
        validate_agent_id("agent_123")
        print("✅ Validierungsfunktionen funktionieren")
    except Exception as e:
        print(f"❌ Validierungsfehler: {e}")
    
    # Mathematische Funktionen testen
    print(f"Sigmoid(0): {sigmoid(0):.3f}")
    print(f"Tanh(1): {tanh_activation(1):.3f}")
    print(f"Gewichteter Durchschnitt: {weighted_average([1, 2, 3], [0.5, 0.3, 0.2]):.3f}")
    
    # Konfiguration testen
    config = load_config()
    print(f"✅ Konfiguration geladen: {len(config)} Bereiche")
    
    print("🔧 Utility-Tests abgeschlossen")
