import numpy as np
import random
from typing import Dict, List, Optional, Tuple, Any
from core.logger import logger


class EthicalScenario:
    """Repräsentiert ein ethisches Szenario oder Dilemma."""

    def __init__(
        self,
        scenario_id: str,
        description: str,
        relevant_beliefs: Dict[str, float],
        options: Dict[str, Dict[str, float]],
        option_attributes: Optional[Dict[str, Dict[str, float]]] = None,
        outcome_feedback: Optional[Dict[str, Dict[str, float]]] = None,
        moral_implications: Optional[Dict[str, Dict[str, float]]] = None,
    ):
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

        # Schwierigkeit/Komplexität des Szenarios
        self.complexity = 0.5

        # Narrative Elemente (für narrative Verarbeitung)
        self.narrative_elements: Dict[str, Any] = {
            "characters": [],
            "conflict": "",
            "context": "",
            "coherence": 0.7,  # Wie kohärent/nachvollziehbar das Szenario ist (0-1)
        }

        # Zeitkritikalität des Szenarios
        self.time_pressure = 0.5  # 0 = kein Zeitdruck, 1 = extreme Zeitkritikalität

        # Unsicherheit über Konsequenzen
        self.uncertainty = 0.5  # 0 = sicher, 1 = sehr unsicher

    def get_option_impact(self, option_name: str, belief_name: str) -> float:
        """Gibt die Auswirkung einer Option auf eine spezifische Überzeugung zurück."""
        if option_name in self.options and belief_name in self.options[option_name]:
            return self.options[option_name][belief_name]
        
        # Logging für fehlende Daten (Silent Failure Prevention)
        if option_name not in self.options:
            logger.debug(f"Option '{option_name}' nicht in Szenario '{self.scenario_id}' gefunden.")
        elif belief_name not in self.options[option_name]:
            # Dies ist normal, wenn eine Option keinen Einfluss auf eine bestimmte Überzeugung hat
            pass
            
        return 0.0

    def get_moral_foundation_impact(self, option_name: str, foundation: str) -> float:
        """Gibt die Auswirkung einer Option auf eine moralische Grundlage zurück."""
        if (
            option_name in self.moral_implications
            and foundation in self.moral_implications[option_name]
        ):
            return self.moral_implications[option_name][foundation]
            
        if option_name not in self.moral_implications:
             logger.debug(f"Keine moralischen Implikationen für Option '{option_name}' in Szenario '{self.scenario_id}'.")
             
        return 0.0

    def add_narrative_element(self, element_type: str, content: str):
        """Fügt ein narratives Element hinzu."""
        if element_type in self.narrative_elements:
            if isinstance(self.narrative_elements[element_type], list):
                self.narrative_elements[element_type].append(content)
            else:
                self.narrative_elements[element_type] = content

    def __str__(self):
        return f"EthicalScenario({self.scenario_id}): {self.description[:50]}..."


class ScenarioGenerator:
    """Generator für ethische Szenarien."""

    def __init__(self):
        """Initialisiert den Szenario-Generator."""
        self.templates = self._initialize_templates()
        self.belief_categories = [
            "Gerechtigkeit",
            "Freiheit",
            "Sicherheit",
            "Wohlfahrt",
            "Autonomie",
            "Würde",
            "Gleichheit",
            "Solidarität",
        ]

    def _initialize_templates(self) -> Dict[str, Dict]:
        """Initialisiert vordefinierte Szenario-Templates."""
        return {
            "trolley_problem": {
                "description": "Ein außer Kontrolle geratener Zug rast auf fünf Personen zu. Sie können eine Weiche umlegen und den Zug auf ein Nebengleis lenken, wodurch eine Person stirbt, aber fünf gerettet werden.",
                "options": {
                    "weiche_umlegen": {
                        "Utilitarismus": 0.8,
                        "Deontologie": -0.3,
                        "Tugendethik": 0.1,
                    },
                    "nichts_tun": {"Utilitarismus": -0.8, "Deontologie": 0.3, "Tugendethik": -0.1},
                },
                "moral_implications": {
                    "weiche_umlegen": {
                        "care": -0.2,  # Direkter Schaden
                        "fairness": 0.6,  # Mehr Leben gerettet
                        "authority": -0.1,
                    },
                    "nichts_tun": {
                        "care": -0.6,  # Mehr Schaden durch Untätigkeit
                        "fairness": -0.2,
                        "authority": 0.1,  # Respekt vor bestehenden Verhältnissen
                    },
                },
                "emotional_valence": -0.7,
                "complexity": 0.6,
                "time_pressure": 0.9,
                "uncertainty": 0.3,
            },
            "autonomous_vehicle": {
                "description": "Ein autonomes Fahrzeug muss in einer unvermeidbaren Unfallsituation entscheiden: Soll es geradeaus fahren und zwei Fußgänger gefährden oder ausweichen und den Fahrer gefährden?",
                "options": {
                    "geradeaus_fahren": {
                        "Schutz_der_Insassen": -0.8,
                        "Schutz_der_Öffentlichkeit": -0.6,
                        "Programmierte_Ethik": 0.2,
                    },
                    "ausweichen": {
                        "Schutz_der_Insassen": -0.9,
                        "Schutz_der_Öffentlichkeit": 0.7,
                        "Programmierte_Ethik": 0.5,
                    },
                },
                "moral_implications": {
                    "geradeaus_fahren": {
                        "care": -0.4,
                        "fairness": -0.3,
                        "loyalty": 0.6,  # Loyalität zum Käufer
                    },
                    "ausweichen": {"care": 0.3, "fairness": 0.5, "loyalty": -0.4},
                },
                "emotional_valence": -0.5,
                "complexity": 0.8,
                "time_pressure": 1.0,
                "uncertainty": 0.7,
            },
            "privacy_vs_security": {
                "description": "Eine Regierung möchte umfassende Überwachungsmaßnahmen einführen, um Terroranschläge zu verhindern. Dies würde die Privatsphäre der Bürger erheblich einschränken.",
                "options": {
                    "überwachung_einführen": {
                        "Sicherheit": 0.7,
                        "Privatsphäre": -0.8,
                        "Staatliche_Kontrolle": 0.6,
                    },
                    "überwachung_ablehnen": {
                        "Sicherheit": -0.4,
                        "Privatsphäre": 0.8,
                        "Staatliche_Kontrolle": -0.5,
                    },
                },
                "moral_implications": {
                    "überwachung_einführen": {
                        "care": 0.3,  # Schutz vor Terror
                        "liberty": -0.8,  # Einschränkung der Freiheit
                        "authority": 0.5,
                    },
                    "überwachung_ablehnen": {"care": -0.2, "liberty": 0.8, "authority": -0.3},
                },
                "emotional_valence": -0.3,
                "complexity": 0.9,
                "time_pressure": 0.4,
                "uncertainty": 0.8,
            },
            "environmental_dilemma": {
                "description": "Ein Unternehmen kann durch den Bau einer neuen Fabrik 1000 Arbeitsplätze schaffen, würde aber ein ökologisch wertvolles Gebiet zerstören und zur Klimakrise beitragen.",
                "options": {
                    "fabrik_bauen": {
                        "Wirtschaftliches_Wohl": 0.8,
                        "Umweltschutz": -0.9,
                        "Lokale_Gemeinschaft": 0.6,
                    },
                    "fabrik_verhindern": {
                        "Wirtschaftliches_Wohl": -0.6,
                        "Umweltschutz": 0.9,
                        "Lokale_Gemeinschaft": -0.4,
                    },
                },
                "moral_implications": {
                    "fabrik_bauen": {
                        "care": 0.2,  # Arbeitsplätze helfen Menschen
                        "fairness": -0.3,  # Umweltkosten für alle
                        "loyalty": 0.4,  # Zur lokalen Gemeinschaft
                    },
                    "fabrik_verhindern": {
                        "care": 0.5,  # Schutz zukünftiger Generationen
                        "fairness": 0.6,
                        "loyalty": -0.2,
                    },
                },
                "emotional_valence": -0.2,
                "complexity": 0.8,
                "time_pressure": 0.5,
                "uncertainty": 0.6,
            },
        }

    def create_scenario_from_template(
        self, template_name: str, scenario_id: Optional[str] = None
    ) -> EthicalScenario:
        """Erstellt ein Szenario basierend auf einem Template."""
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' nicht gefunden")

        template = self.templates[template_name]
        scenario_id = scenario_id or f"{template_name}_{random.randint(1000, 9999)}"

        scenario = EthicalScenario(
            scenario_id=scenario_id,
            description=template["description"],
            relevant_beliefs=template.get("relevant_beliefs", {}),
            options=template["options"],
            option_attributes=template.get("option_attributes", {}),
            outcome_feedback=template.get("outcome_feedback", {}),
            moral_implications=template.get("moral_implications", {}),
        )

        # Zusätzliche Eigenschaften setzen
        scenario.emotional_valence = template.get("emotional_valence", 0.0)
        scenario.complexity = template.get("complexity", 0.5)
        scenario.time_pressure = template.get("time_pressure", 0.5)
        scenario.uncertainty = template.get("uncertainty", 0.5)

        return scenario

    def generate_random_scenario(self, scenario_id: Optional[str] = None) -> EthicalScenario:
        """Generiert ein zufälliges Szenario."""
        scenario_id = scenario_id or f"random_{random.randint(1000, 9999)}"

        # Zufällige Beschreibung generieren
        descriptions = [
            "Ein schwieriges ethisches Dilemma erfordert eine Entscheidung zwischen widersprüchlichen Werten.",
            "Verschiedene Stakeholder haben konfliktäre Interessen in dieser ethischen Situation.",
            "Eine komplexe Situation erfordert die Abwägung verschiedener moralischer Prinzipien.",
        ]

        description = random.choice(descriptions)

        # Zufällige Überzeugungen und Optionen
        num_beliefs = random.randint(2, 4)
        selected_beliefs = random.sample(self.belief_categories, num_beliefs)

        relevant_beliefs = {belief: random.uniform(0.3, 0.9) for belief in selected_beliefs}

        # Zwei Optionen mit zufälligen Auswirkungen
        options: Dict[str, Dict[str, float]] = {}
        for i, option_name in enumerate([f"Option_A", f"Option_B"]):
            options[option_name] = {}
            for belief in selected_beliefs:
                # Eine Option positiv, eine negativ für Konflikt
                impact = random.uniform(-0.8, 0.8)
                if i == 1:  # Zweite Option tendenziell umgekehrt
                    impact *= -0.7
                options[option_name][belief] = impact

        scenario = EthicalScenario(
            scenario_id=scenario_id,
            description=description,
            relevant_beliefs=relevant_beliefs,
            options=options,
        )

        # Zufällige Eigenschaften
        scenario.emotional_valence = random.uniform(-0.5, 0.5)
        scenario.complexity = random.uniform(0.3, 0.9)
        scenario.time_pressure = random.uniform(0.1, 0.8)
        scenario.uncertainty = random.uniform(0.2, 0.8)

        return scenario

    def get_available_templates(self) -> List[str]:
        """Gibt eine Liste aller verfügbaren Templates zurück."""
        return list(self.templates.keys())

    def create_custom_scenario(
        self, scenario_id: str, description: str, options: Dict[str, Dict[str, float]], **kwargs
    ) -> EthicalScenario:
        """Erstellt ein benutzerdefiniertes Szenario."""
        return EthicalScenario(
            scenario_id=scenario_id,
            description=description,
            relevant_beliefs=kwargs.get("relevant_beliefs", {}),
            options=options,
            option_attributes=kwargs.get("option_attributes", {}),
            outcome_feedback=kwargs.get("outcome_feedback", {}),
            moral_implications=kwargs.get("moral_implications", {}),
        )


# Vordefinierte Szenarien für einfache Verwendung
def get_trolley_problem() -> EthicalScenario:
    """Gibt das klassische Trolley-Problem zurück."""
    generator = ScenarioGenerator()
    return generator.create_scenario_from_template("trolley_problem")


def get_autonomous_vehicle_dilemma() -> EthicalScenario:
    """Gibt das autonome Fahrzeug-Dilemma zurück."""
    generator = ScenarioGenerator()
    return generator.create_scenario_from_template("autonomous_vehicle")


def get_privacy_security_dilemma() -> EthicalScenario:
    """Gibt das Privatsphäre vs. Sicherheit-Dilemma zurück."""
    generator = ScenarioGenerator()
    return generator.create_scenario_from_template("privacy_vs_security")


def get_environmental_dilemma() -> EthicalScenario:
    """Gibt das Umwelt-Dilemma zurück."""
    generator = ScenarioGenerator()
    return generator.create_scenario_from_template("environmental_dilemma")


if __name__ == "__main__":
    # Test der Szenarien
    generator = ScenarioGenerator()

    print("Verfügbare Templates:")
    for template in generator.get_available_templates():
        print(f"- {template}")

    print("\nTrolley Problem Test:")
    trolley = get_trolley_problem()
    print(f"ID: {trolley.scenario_id}")
    print(f"Beschreibung: {trolley.description}")
    print(f"Optionen: {list(trolley.options.keys())}")

    print("\nZufälliges Szenario:")
    random_scenario = generator.generate_random_scenario()
    print(f"ID: {random_scenario.scenario_id}")
    print(f"Beschreibung: {random_scenario.description}")
