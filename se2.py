# Block 1: Konstanten und Hilfsfunktionen (z.B. in constants.py)
# Purpose: Zentralisierung von Konfigurationswerten und magischen Zahlen.

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from scipy.spatial.distance import cosine
import random
import pickle
import logging
import os
from typing import List, Dict, Tuple, Optional, Set, Union, Callable, Any

# --- Verarbeitungstypen ---
class NeuralProcessingType:
    SYSTEMATIC = "systematic"
    INTUITIVE = "intuitive"
    ASSOCIATIVE = "associative"
    ANALOGICAL = "analogical"
    EMOTIONAL = "emotional"
    NARRATIVE = "narrative"

    ALL_TYPES = [SYSTEMATIC, INTUITIVE, ASSOCIATIVE, ANALOGICAL, EMOTIONAL, NARRATIVE]

    @staticmethod
    def get_random():
        return np.random.choice(NeuralProcessingType.ALL_TYPES)

# --- Kognitive Architektur Parameter ---
# Default Beta Verteilungsparameter (alpha, beta)
# Hohe alpha/beta -> engere Verteilung um 0.5
# Niedrige alpha/beta -> breitere Verteilung
BETA_DIST_CENTERED = (5, 5) # Mittelwert um 0.5
BETA_DIST_HIGH_SKEW = (3, 7) # Tendenz zu niedrigen Werten (z.B. confirmation_bias)
BETA_DIST_LOW_SKEW = (7, 3) # Tendenz zu hohen Werten (z.B. negativity_bias)

# Einflussfaktoren in Aktivierungsfunktionen
SYSTEMATIC_BIAS_REDUCTION = 0.7
INTUITIVE_AVAILABILITY_EFFECT_SCALE = 0.5
ASSOCIATIVE_ACTIVATION_THRESHOLD = 0.3
ASSOCIATIVE_SPREAD_DAMPING = 0.4
ASSOCIATIVE_CONCEPT_SPREAD_DAMPING = 0.3
ANALOGICAL_STRENGTH = 0.5
ANALOGICAL_CONTEXT_WEIGHT = 0.1
EMOTIONAL_REGULATION_DAMPING = 0.5
NARRATIVE_COHERENCE_BIAS = 0.3

# Bayes'sche Update Parameter
BAYESIAN_DEFAULT_EVIDENCE_WEIGHT = 0.5
BAYESIAN_SYSTEMATIC_EVIDENCE_WEIGHT_BOOST = 0.2
BAYESIAN_EMOTIONAL_EVIDENCE_WEIGHT_REDUCTION = 0.2
BAYESIAN_ANCHORING_EFFECT_SCALE = 0.2

# --- Belief Parameter ---
BELIEF_ACTIVATION_DECAY_RATE = 0.8 # Pro spreading_activation Runde
BELIEF_ACTIVATION_TIME_DECAY_RATE = 0.1 # Für Zeit zwischen Aktivierungen

# --- Agenten Parameter ---
# Persönlichkeitsmerkmale (Big Five)
PERSONALITY_DEFAULT_DIST = BETA_DIST_CENTERED

# Kognitive Architektur Generierung
COGNITIVE_ARCH_OPENNESS_THRESHOLD = 0.7
COGNITIVE_ARCH_CONSCIENTIOUSNESS_THRESHOLD = 0.6
COGNITIVE_ARCH_AGREEABLENESS_THRESHOLD = 0.7
COGNITIVE_ARCH_EXTROVERSION_THRESHOLD = 0.6
COGNITIVE_ARCH_NEUROTICISM_THRESHOLD = 0.7
COGNITIVE_ARCH_DEFAULT_BALANCE_RANGE = (0.6, 0.8)
COGNITIVE_ARCH_CONSC_BALANCE_MOD = 0.3
COGNITIVE_ARCH_NEUR_BALANCE_MOD = 0.3

# Morale Grundlagen
MORAL_FOUNDATIONS_DEFAULT_DIST = BETA_DIST_CENTERED

# Arbeitsspeicher
WORKING_MEMORY_BASE_CAPACITY = 5
WORKING_MEMORY_CONSC_CAPACITY_MOD = 2
WORKING_MEMORY_BASE_RETENTION = 0.7
WORKING_MEMORY_CONSC_RETENTION_MOD = 0.3

# Entscheidungsfindung
DECISION_MORAL_WEIGHT = 0.3
DECISION_GROUP_NORM_WEIGHT = 0.2
DECISION_RISK_AVERSION_BASE = 0.7
DECISION_RISK_OPENNESS_MOD = 0.4
DECISION_INGROUP_BIAS_MOD = 1.0 # Multiplikator: 1.0 + bias
DECISION_NOISE_STD_DEV_DEFAULT = 0.2
DECISION_NOISE_STD_DEV_SYSTEMATIC = 0.1
DECISION_NOISE_STD_DEV_INTUITIVE = 0.3
DECISION_CONSISTENCY_BOOST_FACTOR = 0.2
DECISION_RECENCY_BOOST_FACTOR = 0.15
DECISION_EMOTION_BOOST_FACTOR = 0.3
DECISION_SIGNIFICANT_DISSONANCE_THRESHOLD = 0.3
DECISION_HIGH_CONFIDENCE_THRESHOLD = 0.7
DECISION_UPDATE_DURING_DELIBERATION = False # Deaktiviert zur Klarheit
DECISION_DELIBERATION_EVIDENCE_STRENGTH = 0.1 # Falls oben aktiviert

# Lernen & Anpassung
EXPERIENCE_LEARNING_RATE_BASE = 0.05
EXPERIENCE_LEARNING_RATE_SYS_MOD = 1.2
EXPERIENCE_LEARNING_RATE_ASSOC_MOD = 1.1
EXPERIENCE_LEARNING_OPENNESS_MOD_RANGE = (0.8, 1.2) # Skaliert von 0.8 bis 1.2
EXPERIENCE_CERTAINTY_CHANGE_BASE = 0.05
EXPERIENCE_CERTAINTY_CHANGE_INCONSISTENT_PENALTY = 0.1
EXPERIENCE_VALENCE_CHANGE_THRESHOLD = 0.3
EXPERIENCE_VALENCE_CHANGE_RATE = 0.1
EXPERIENCE_SIGNIFICANT_CHANGE_THRESHOLD = 0.1

# Propagation
PROPAGATION_STRENGTH_DEFAULT = 0.5
PROPAGATION_STRENGTH_ASSOC_MOD = 0.6
PROPAGATION_STRENGTH_SYS_MOD = 0.4
PROPAGATION_ASSOC_CONCEPT_BOOST = 0.3

# Sozialer Einfluss
SOCIAL_LEARNING_RATE_BASE = 0.02
SOCIAL_LEARNING_NARRATIVE_FACTOR = 1.3
SOCIAL_LEARNING_SYSTEMATIC_FACTOR = 0.7
SOCIAL_LEARNING_AGREEABLENESS_FACTOR_RANGE = (0.7, 1.3) # Skaliert von 0.7 bis 1.3
SOCIAL_LEARNING_AUTHORITY_BIAS_MOD = 1.0 # Multiplikator: 1.0 + bias * perceived_authority
SOCIAL_LEARNING_CERTAINTY_FACTOR_SELF_MOD = 0.5 # Reduktion basierend auf eigener Gewissheit
SOCIAL_LEARNING_CERTAINTY_FACTOR_OTHER_MOD = 0.5 # Boost basierend auf Gewissheit des anderen
SOCIAL_LEARNING_INGROUP_BIAS_MOD = 1.0 # Multiplikator: 1.0 + bias * shared_identity
SOCIAL_LEARNING_DOGMATISM_FACTOR = 0.7 # Reduktion bei großen Differenzen
SOCIAL_LEARNING_MIN_CHANGE_THRESHOLD = 0.01
SOCIAL_LEARNING_SIGNIFICANT_VALENCE_CHANGE_THRESHOLD = 0.05
SOCIAL_LEARNING_VALENCE_CHANGE_RATE = 0.5
SOCIAL_LEARNING_CERTAINTY_CHANGE_RATE = 0.3
SOCIAL_LEARNING_SIGNIFICANT_INFLUENCE_THRESHOLD = 0.1

# Reflexion
REFLECTION_ENABLED_STYLES = {NeuralProcessingType.SYSTEMATIC, NeuralProcessingType.ANALOGICAL} # Erweiterbar
REFLECTION_STRENGTH_BASE = 0.3
REFLECTION_STRENGTH_SYS_MOD = 0.5
REFLECTION_OPENNESS_MOD_RANGE = (0.7, 1.3)
REFLECTION_MEMORY_FRACTION = 0.3
REFLECTION_MIN_MEMORY_ITEMS = 5
REFLECTION_CONSISTENCY_THRESHOLD = 0.6
REFLECTION_CONSOLIDATION_RATE = 0.3
REFLECTION_CERTAINTY_BOOST_RATE = 0.05

# --- Society Parameter ---
# Agent Generation
AGENT_GENERATION_BELIEF_DIST = "beta" # normal, uniform, beta
AGENT_GENERATION_MIN_BELIEFS = 5
AGENT_GENERATION_MAX_BELIEFS = 15
AGENT_GENERATION_CONN_PROB = 0.3
AGENT_GENERATION_CONN_STRENGTH_RANGE = (0.1, 0.5)
AGENT_GENERATION_SIMILARITY_VARIATION_FACTOR = 0.2 # Multiplikator für (1 - similarity)
AGENT_GENERATION_BELIEF_STRENGTH_VARIATION_FACTOR = 0.3
AGENT_GENERATION_BELIEF_CERTAINTY_VARIATION_FACTOR = 0.3
AGENT_GENERATION_BELIEF_VALENCE_VARIATION_FACTOR = 0.3
AGENT_GENERATION_GROUP_ID_VARIATION_FACTOR = 0.3
AGENT_GENERATION_SIMILARITY_POLARITY_FLIP_PROB = 0.1

# Netzwerk Generierung
NETWORK_DEFAULT_DENSITY = 0.1
NETWORK_GROUP_CONN_BOOST = 0.3
NETWORK_BELIEF_SIM_FACTOR = 0.5
NETWORK_COGNITIVE_SIM_FACTOR = 0.2 # War vorher cognitive_style_influence
NETWORK_SIMILARITY_CONN_PROB_MOD = 0.5
NETWORK_EXTROVERSION_CONN_PROB_MOD = 0.2
NETWORK_NARRATIVE_CONN_PROB_BOOST = 0.2
NETWORK_EMOTIONAL_CONN_PROB_BOOST = 0.15
NETWORK_MAX_CONN_PROB = 0.95
NETWORK_STRENGTH_FROM_SIMILARITY_RANGE = (0.3, 1.0) # Skaliert von 0.3 bis 1.0

# Simulation
SIMULATION_DEFAULT_ENSEMBLE_SIZE = 3
SIMULATION_VALIDATION_INTERVAL = 10 # Validierung alle X Schritte
SIMULATION_DEFAULT_SCENARIO_PROB = 0.2
SIMULATION_DEFAULT_SOCIAL_PROB = 0.3
SIMULATION_DEFAULT_REFLECTION_PROB = 0.1

# Validierung & Analyse
VALIDATION_LARGE_BELIEF_CHANGE_WARNING = 0.3
VALIDATION_HIGH_DISSONANCE_WARNING = 1.0
VALIDATION_EXTREME_POLARIZATION_THRESHOLD = 0.8 # Bimodalitätsindex
ANALYSIS_BELIEF_CLUSTER_THRESHOLD = 0.7
ANALYSIS_STRONG_BELIEF_THRESHOLD = 0.7
ANALYSIS_CLUSTER_MIN_SIZE = 2
ANALYSIS_DECISION_CONSENSUS_STEPS = 3 # Anzahl Schritte zurück für Konsensberechnung
ANALYSIS_HIGH_ENSEMBLE_VARIANCE_THRESHOLD = 0.05
ANALYSIS_TOP_HIGH_VARIANCE_BELIEFS = 5

# Visualisierung
VIS_BELIEF_NETWORK_MIN_CONN_STRENGTH = 0.2
VIS_BELIEF_NODE_BASE_SIZE = 300
VIS_BELIEF_NODE_STRENGTH_SCALING = 700
VIS_BELIEF_NODE_ACTIVATION_SCALING = 700
VIS_BELIEF_NODE_CERTAINTY_ALPHA_RANGE = (0.3, 1.0)
VIS_BELIEF_EDGE_WEIGHT_SCALING = 2
VIS_SOCIAL_NODE_DEGREE_SCALING = 300
VIS_SOCIAL_EDGE_WEIGHT_SCALING = 2
VIS_STYLE_COMPARISON_NUM_BELIEFS = 5
VIS_STYLE_COMPARISON_BAR_WIDTH = 0.15

# --- Hilfsfunktionen ---
def np_random_beta(a, b):
    """Wrapper für Beta-Verteilung."""
    return np.random.beta(a, b)

def clip(value, min_val=0.0, max_val=1.0):
    """Wrapper für np.clip."""
    return np.clip(value, min_val, max_val)

def scale_value(value, old_min, old_max, new_min, new_max):
    """Skaliert einen Wert von einem Bereich in einen anderen."""
    if old_max == old_min:
        return new_min
    return (((value - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min


# Block 2: Kernklassen - NeuralEthicalBelief und CognitiveArchitecture
# Purpose: Definition der Bausteine für Überzeugungen und kognitive Verarbeitung.
# Änderungen: Konstanten verwendet, Logik in apply_bayesian_update überarbeitet,
#              Dokumentation verbessert, Aktivierungsfunktionen leicht angepasst.

import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Set, Union, Callable
import logging

# Importiere Konstanten (Annahme: constants.py existiert im selben Verzeichnis)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class NeuralEthicalBelief:
    """
    Repräsentiert eine ethische Überzeugung mit neuronaler Modellierung.

    Attributes:
        name (str): Beschreibender Name der Überzeugung.
        category (str): Kategorie der Überzeugung (z.B. 'Gerechtigkeit', 'Freiheit').
        strength (float): Aktuelle Stärke der Überzeugung (0-1).
        certainty (float): Gewissheit über die Überzeugung (0-1).
        emotional_valence (float): Emotionale Ladung der Überzeugung (-1 bis +1).
        connections (Dict[str, Tuple[float, int]]): Verbindungen zu anderen Überzeugungen
                                                      (belief_name -> (Einflussstärke, Polarität [-1, 1])).
        associated_concepts (Dict[str, float]): Konzepte, die mit dieser Überzeugung assoziiert sind
                                               (concept_name -> Assoziationsstärke).
        activation (float): Aktuelles Aktivierungsniveau für Spreading Activation.
        last_activation_time (int): Zeitpunkt der letzten Aktivierung.
    """
    def __init__(self, name: str, category: str, initial_strength: float = 0.5,
                 certainty: float = 0.5, emotional_valence: float = 0.0):
        """Initialisiert eine ethische Überzeugung."""
        self.name = name
        self.category = category
        self._strength = clip(initial_strength)
        self._certainty = clip(certainty)
        self._emotional_valence = clip(emotional_valence, -1.0, 1.0)

        self.connections: Dict[str, Tuple[float, int]] = {}
        self.associated_concepts: Dict[str, float] = {}
        self.activation: float = 0.0
        self.last_activation_time: int = 0

    # --- Properties mit Validierung ---
    @property
    def strength(self) -> float:
        return self._strength

    @strength.setter
    def strength(self, value: float):
        self._strength = clip(value)

    @property
    def certainty(self) -> float:
        return self._certainty

    @certainty.setter
    def certainty(self, value: float):
        self._certainty = clip(value)

    @property
    def emotional_valence(self) -> float:
        return self._emotional_valence

    @emotional_valence.setter
    def emotional_valence(self, value: float):
        self._emotional_valence = clip(value, -1.0, 1.0)

    # --- Methoden ---
    def add_connection(self, belief_name: str, influence_strength: float, polarity: int):
        """Fügt eine Verbindung zu einer anderen Überzeugung hinzu."""
        self.connections[belief_name] = (clip(influence_strength), np.sign(polarity))

    def add_associated_concept(self, concept_name: str, association_strength: float):
        """Fügt ein assoziiertes Konzept hinzu."""
        self.associated_concepts[concept_name] = clip(association_strength)

    def activate(self, activation_level: float, current_time: int):
        """
        Aktiviert die Überzeugung für Spreading Activation.

        Berücksichtigt zeitlichen Zerfall seit der letzten Aktivierung.
        Akkumuliert Aktivierung, begrenzt sie aber nach oben, um unendliches Wachstum zu vermeiden.
        """
        # Zeitlichen Abfall modellieren
        time_since_last = current_time - self.last_activation_time
        # Verwende die Konstante für den Zeit-basierten Decay
        decay_factor = np.exp(-BELIEF_ACTIVATION_TIME_DECAY_RATE * time_since_last) if time_since_last > 0 else 1.0

        # Aktualisieren mit Decay und additiver neuer Aktivierung
        new_activation = decay_factor * self.activation + activation_level
        # Begrenzung der Aktivierung (z.B. auf einen Maximalwert von 2.0)
        self.activation = clip(new_activation, 0.0, 2.0) # Obergrenze hinzugefügt
        self.last_activation_time = current_time

    def __str__(self):
        return (f"Belief({self.name}, Cat: {self.category}, Str: {self.strength:.2f}, "
                f"Cert: {self.certainty:.2f}, Val: {self.emotional_valence:.2f}, Act: {self.activation:.2f})")


class CognitiveArchitecture:
    """
    Modelliert die kognitive Architektur eines Agenten, inklusive Verarbeitungsstilen,
    Biases, emotionalen Parametern und Bayes'scher Update-Logik.
    """
    def __init__(self,
                 primary_processing: str = NeuralProcessingType.SYSTEMATIC,
                 secondary_processing: str = NeuralProcessingType.INTUITIVE,
                 processing_balance: float = 0.5):
        """
        Initialisiert die kognitive Architektur.

        Args:
            primary_processing: Primärer Verarbeitungstyp.
            secondary_processing: Sekundärer Verarbeitungstyp.
            processing_balance: Balance zwischen primärer und sekundärer Verarbeitung (0-1).
                                Wert > 0.5 bedeutet mehr Gewicht auf primärem Stil.
        """
        if primary_processing not in NeuralProcessingType.ALL_TYPES:
            raise ValueError(f"Ungültiger primärer Verarbeitungstyp: {primary_processing}")
        if secondary_processing is not None and secondary_processing not in NeuralProcessingType.ALL_TYPES:
            raise ValueError(f"Ungültiger sekundärer Verarbeitungstyp: {secondary_processing}")
        if primary_processing == secondary_processing:
             logging.warning(f"Primärer und sekundärer Verarbeitungstyp sind identisch ({primary_processing}).")
             # Optional: Sekundären Stil entfernen oder zufällig neu wählen
             # secondary_processing = None


        self.primary_processing = primary_processing
        self.secondary_processing = secondary_processing
        self.processing_balance = clip(processing_balance)

        # Kognitive Verzerrungen (Biases) - Initialisierung mit Beta-Verteilungen
        self.cognitive_biases = {
            "confirmation_bias": np_random_beta(*BETA_DIST_HIGH_SKEW),
            "availability_bias": np_random_beta(*BETA_DIST_CENTERED),
            "anchoring_bias": np_random_beta(*BETA_DIST_CENTERED),
            "authority_bias": np_random_beta(*BETA_DIST_CENTERED),
            "ingroup_bias": np_random_beta(*BETA_DIST_CENTERED),
            # Optional: Dogmatismus oder andere Biases hinzufügen
            "dogmatism": np_random_beta(*BETA_DIST_CENTERED)
        }

        # Emotionale Parameter
        self.emotional_parameters = {
            "emotional_reactivity": np_random_beta(*BETA_DIST_CENTERED),
            "emotional_regulation": np_random_beta(*BETA_DIST_CENTERED),
            "empathy": np_random_beta(*BETA_DIST_CENTERED),
            "negativity_bias": np_random_beta(*BETA_DIST_LOW_SKEW) # Tendenz zu > 0.5
        }

        # Bayes'sche Verarbeitungsparameter
        self.bayesian_parameters = {
            "prior_strength_factor": np_random_beta(*BETA_DIST_CENTERED), # Einfluss des Priors
            "evidence_sensitivity": np_random_beta(*BETA_DIST_CENTERED), # Wie stark auf Evidenz reagiert wird
            "update_rate": np_random_beta(*BETA_DIST_CENTERED) # Lerngeschwindigkeit
            # "evidence_threshold" wurde entfernt, da es in der Update-Logik nicht klar verwendet wurde.
        }

        # Neuronale Aktivierungsfunktionen für verschiedene Verarbeitungstypen
        # Mapping der Typen auf ihre Verarbeitungsfunktionen
        self.activation_functions: Dict[str, Callable] = {
            NeuralProcessingType.SYSTEMATIC: self._systematic_activation,
            NeuralProcessingType.INTUITIVE: self._intuitive_activation,
            NeuralProcessingType.ASSOCIATIVE: self._associative_activation,
            NeuralProcessingType.ANALOGICAL: self._analogical_activation,
            NeuralProcessingType.EMOTIONAL: self._emotional_activation,
            NeuralProcessingType.NARRATIVE: self._narrative_activation
        }

    # --- Private Aktivierungsfunktionen ---
    # Diese Funktionen modellieren, wie ein bestimmter Denkstil die
    # Wahrnehmung oder Bewertung von Inputs (z.B. Überzeugungsstärken) verändert.
    # Sie geben modifizierte Werte zurück.

    def _apply_cognitive_biases(self, inputs: Dict[str, float], context: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Wendet relevante kognitive Biases auf die Inputs an.
        Dies ist eine *vereinfachte* Darstellung, wie Biases wirken könnten.
        Wird *vor* den spezifischen Aktivierungsfunktionen aufgerufen.
        """
        biased_inputs = inputs.copy()
        context = context or {}

        # Beispiel: Ankereffekt (wenn ein relevanter Anker im Kontext ist)
        anchor_value = context.get("anchor_belief_strength")
        if anchor_value is not None:
            anchoring_bias = self.cognitive_biases.get("anchoring_bias", 0)
            for key, value in biased_inputs.items():
                 # Verschiebt den Wert leicht in Richtung des Ankers
                 biased_inputs[key] = value + anchoring_bias * (anchor_value - value) * 0.1 # Kleiner Effekt

        # Beispiel: Verfügbarkeitsheuristik (könnte durch salient_concepts im Kontext beeinflusst werden)
        salient_concepts = context.get("salient_concepts", [])
        if salient_concepts:
             availability_bias = self.cognitive_biases.get("availability_bias", 0)
             for key in salient_concepts:
                 if key in biased_inputs:
                     # Verstärkt die wahrgenommene Stärke salienter Konzepte
                     biased_inputs[key] *= (1.0 + availability_bias * 0.1) # Kleiner Effekt

        # Weitere Biases könnten hier integriert werden (z.B. Ingroup-Bias beeinflusst Wahrnehmung von Gruppeninformationen)

        # Clip, um sicherzustellen, dass Werte im gültigen Bereich bleiben
        return {k: clip(v) for k, v in biased_inputs.items()}


    def _systematic_activation(self, inputs: Dict[str, float], context: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Systematische Verarbeitung: Reduziert den Einfluss von Biases (vereinfacht)."""
        results = {}
        confirmation_bias_level = self.cognitive_biases.get("confirmation_bias", 0)
        # Systematisches Denken reduziert den Einfluss des Bestätigungsfehlers
        reduced_bias_effect = confirmation_bias_level * (1.0 - SYSTEMATIC_BIAS_REDUCTION)

        for key, value in inputs.items():
            # Simuliert Widerstand gegen Bestätigungsfehler: Wenn der Input eine bestehende Neigung
            # (z.B. > 0.5) verstärken würde, wird dieser Effekt leicht gedämpft.
            # Dies ist eine Heuristik!
            if value > 0.5: # Annahme: Verstärkt Tendenz > 0.5
                results[key] = value * (1.0 - reduced_bias_effect * 0.1) # Leichte Dämpfung
            elif value < 0.5: # Annahme: Verstärkt Tendenz < 0.5
                 results[key] = value + reduced_bias_effect * 0.1 * value # Leichte Dämpfung Richtung 0.5
            else:
                 results[key] = value

        return {k: clip(v) for k, v in results.items()} # Clip am Ende

    def _intuitive_activation(self, inputs: Dict[str, float], context: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Intuitive Verarbeitung: Stärker beeinflusst durch leicht verfügbare/saliente Infos."""
        results = {}
        context = context or {}
        availability_bias = self.cognitive_biases.get("availability_bias", 0)
        # Konzepte, die im Kontext als "salient" markiert sind, erhalten mehr Gewicht
        salient_concepts = context.get("salient_concepts", [])

        for key, value in inputs.items():
            boost = 0.0
            if key in salient_concepts:
                boost = availability_bias * INTUITIVE_AVAILABILITY_EFFECT_SCALE

            # Verstärkt den Wert, wenn er salient ist
            results[key] = value * (1.0 + boost)

        return {k: clip(v) for k, v in results.items()}

    def _associative_activation(self, inputs: Dict[str, float], context: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Assoziative Verarbeitung: Aktiviert verbundene Konzepte stärker."""
        results = inputs.copy() # Start mit den Originalwerten
        context = context or {}
        # Annahme: context enthält Aktivierungslevel anderer (möglicherweise assoziierter) Konzepte
        # Beispiel: context = {"belief_A": 0.8, "belief_C": 0.6}

        for key, value in inputs.items():
            if value > ASSOCIATIVE_ACTIVATION_THRESHOLD: # Nur wenn das Konzept selbst aktiv genug ist
                # Suche nach aktiven Konzepten im Kontext
                for other_key, other_activation in context.items():
                    if other_key != key and other_key in inputs and other_activation > ASSOCIATIVE_ACTIVATION_THRESHOLD:
                        # Einfaches assoziatives Spreading: Verstärkung durch aktive Nachbarn
                        # Hier fehlt eigentlich das Belief-Netzwerk! Diese Funktion ist konzeptionell schwierig
                        # ohne Zugriff auf die Verbindungen des Agenten.
                        # Vereinfachung: Wir nehmen an, der Kontext *repräsentiert* irgendwie die Nachbarschaft.
                        association_boost = other_activation * ASSOCIATIVE_SPREAD_DAMPING
                        # Ergebnisse können nur verstärkt werden (keine Hemmung hier)
                        results[key] = max(results[key], value * (1.0 + association_boost))

        # Hier könnte auch Aktivierung basierend auf `associated_concepts` im Belief stattfinden
        # (TODO: Komplexere Implementierung benötigt Agenten-Beliefs hier)

        return {k: clip(v) for k, v in results.items()}

    def _analogical_activation(self, inputs: Dict[str, float], context: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Analoge Verarbeitung: Verstärkt durch Ähnlichkeiten (simuliert)."""
        results = inputs.copy()
        context = context or {}
        # Annahme: context enthält Informationen über ein analoges Szenario/Konzept
        analogy_source_strength = context.get("analogy_source_strength", 0.0) # Stärke der Analogiequelle
        similarity_score = context.get("analogy_similarity", 0.0) # Wie ähnlich ist der Input zur Analogie?

        if analogy_source_strength > 0 and similarity_score > 0:
            analogy_boost = analogy_source_strength * similarity_score * ANALOGICAL_STRENGTH
            for key in results:
                 # Verstärkt alle Inputs basierend auf der Analogie
                 results[key] *= (1.0 + analogy_boost * ANALOGICAL_CONTEXT_WEIGHT)

        return {k: clip(v) for k, v in results.items()}

    def _emotional_activation(self, inputs: Dict[str, float], context: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Emotionale Verarbeitung: Moduliert durch emotionale Parameter und Kontext."""
        results = {}
        context = context or {}
        reactivity = self.emotional_parameters.get("emotional_reactivity", 0.5)
        regulation = self.emotional_parameters.get("emotional_regulation", 0.5)
        negativity_bias = self.emotional_parameters.get("negativity_bias", 0.6)
        # Annahme: Kontext enthält emotionale Valenz des Szenarios/der Situation
        context_valence = context.get("emotional_valence", 0.0) # Valenz von -1 bis +1

        for key, value in inputs.items():
            base_activation = value
            emotional_effect = 0.0

            # Emotionale Modulation basierend auf Kontextvalenz
            if context_valence != 0:
                # Negativitätsverzerrung: Negative Valenz hat stärkeren Effekt
                if context_valence < 0:
                    raw_effect = abs(context_valence) * reactivity * (0.5 + negativity_bias) # Stärker bei Bias > 0.5
                else:
                    raw_effect = context_valence * reactivity * (1.0 - (0.5 + negativity_bias)) # Schwächer bei Bias > 0.5

                # Emotionsregulation dämpft den Effekt
                regulated_effect = raw_effect * (1.0 - regulation * EMOTIONAL_REGULATION_DAMPING)
                emotional_effect = regulated_effect

            # Additiver Effekt der Emotion auf die Aktivierung
            results[key] = base_activation + emotional_effect
            # Alternative: Multiplikativ -> base_activation * (1.0 + emotional_effect)

        return {k: clip(v) for k, v in results.items()}

    def _narrative_activation(self, inputs: Dict[str, float], context: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Narrative Verarbeitung: Verstärkt durch narrative Kohärenz (simuliert)."""
        results = inputs.copy()
        context = context or {}
        # Annahme: Kontext enthält die wahrgenommene Kohärenz der aktuellen Situation/des Szenarios
        narrative_coherence = context.get("narrative_coherence", 0.5) # Kohärenz von 0 bis 1

        if narrative_coherence > 0.5: # Nur wenn halbwegs kohärent
            coherence_boost = (narrative_coherence - 0.5) * 2 * NARRATIVE_COHERENCE_BIAS # Skaliert 0..1
            for key in results:
                 # Verstärkt Inputs proportional zur Kohärenz
                 results[key] *= (1.0 + coherence_boost)

        return {k: clip(v) for k, v in results.items()}

    # --- Kernmethoden ---
    def process_information(self, inputs: Dict[str, float], context: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Verarbeitet Informationen gemäß der kognitiven Architektur.

        Kombiniert primäre und sekundäre Verarbeitungstypen. Wendet zuerst allgemeine Biases an,
        dann die spezifischen Aktivierungsfunktionen.

        Args:
            inputs: Eingabewerte für verschiedene Konzepte (z.B. initiale Belief-Stärken).
            context: Kontextinformationen zur Beeinflussung der Verarbeitung.

        Returns:
            Verarbeitete Werte, die die modifizierte Wahrnehmung oder Bewertung repräsentieren.
        """
        context = context or {}

        # 1. (Optional) Biases auf Inputs anwenden (beeinflusst die "Wahrnehmung")
        # biased_inputs = self._apply_cognitive_biases(inputs, context)
        # Deaktiviert, da die Logik noch sehr heuristisch ist. Biases werden eher
        # in spezifischen Aktivierungsfunktionen oder Update-Regeln berücksichtigt.
        biased_inputs = inputs

        # 2. Primäre Verarbeitung anwenden
        primary_results = self.activation_functions[self.primary_processing](biased_inputs, context)

        # 3. Sekundäre Verarbeitung anwenden (falls vorhanden)
        if self.secondary_processing and self.processing_balance < 1.0:
            secondary_results = self.activation_functions[self.secondary_processing](biased_inputs, context)

            # Gewichtete Kombination der Ergebnisse
            results = {}
            balance = self.processing_balance
            for key in inputs.keys():
                primary_value = primary_results.get(key, 0)
                secondary_value = secondary_results.get(key, 0)

                # Lineare Interpolation
                results[key] = (balance * primary_value + (1.0 - balance) * secondary_value)

                # TODO: Alternative Interaktionen? Z.B. Gating?
                # Beispiel: Wenn Systematisch primär, dämpft es vielleicht Emotional sekundär stärker?
                # if self.primary_processing == NeuralProcessingType.SYSTEMATIC and \
                #    self.secondary_processing == NeuralProcessingType.EMOTIONAL:
                #    results[key] = balance * primary_value + (1.0 - balance) * secondary_value * (1.0 - balance * 0.5)

        else:
            # Nur primäre Verarbeitung
            results = primary_results

        # Sicherstellen, dass Ergebnisse im gültigen Bereich bleiben
        return {k: clip(v) for k, v in results.items()}

    def get_evidence_weight(self) -> float:
        """
        Bestimmt, wie stark neue Evidenz gewichtet wird, basierend auf dem Verarbeitungsstil.
        Systematische Denker gewichten Evidenz stärker, emotionale/intuitive schwächer.
        """
        base_weight = BAYESIAN_DEFAULT_EVIDENCE_WEIGHT
        primary_style_effect = 0.0

        if self.primary_processing == NeuralProcessingType.SYSTEMATIC:
            primary_style_effect = BAYESIAN_SYSTEMATIC_EVIDENCE_WEIGHT_BOOST
        elif self.primary_processing in [NeuralProcessingType.EMOTIONAL, NeuralProcessingType.INTUITIVE]:
            primary_style_effect = -BAYESIAN_EMOTIONAL_EVIDENCE_WEIGHT_REDUCTION

        # Berücksichtige sekundären Stil anteilig
        secondary_style_effect = 0.0
        if self.secondary_processing and self.processing_balance < 1.0:
             if self.secondary_processing == NeuralProcessingType.SYSTEMATIC:
                 secondary_style_effect = BAYESIAN_SYSTEMATIC_EVIDENCE_WEIGHT_BOOST
             elif self.secondary_processing in [NeuralProcessingType.EMOTIONAL, NeuralProcessingType.INTUITIVE]:
                 secondary_style_effect = -BAYESIAN_EMOTIONAL_EVIDENCE_WEIGHT_REDUCTION

        # Gewichteter Durchschnitt der Effekte
        style_effect = (self.processing_balance * primary_style_effect +
                       (1.0 - self.processing_balance) * secondary_style_effect)

        # Modifiziere Basisgewicht durch Stil und individuelle Evidenzsensitivität
        evidence_sensitivity = self.bayesian_parameters.get("evidence_sensitivity", 0.5)
        # Skaliere den Effekt der Evidenzsensitivität (z.B. von -0.3 bis +0.3)
        sensitivity_mod = scale_value(evidence_sensitivity, 0, 1, -0.3, 0.3)

        final_weight = base_weight + style_effect + sensitivity_mod
        return clip(final_weight) # Gewichtung zwischen 0 und 1

    def apply_bayesian_update(self,
                              prior_belief: float,
                              prior_certainty: float,
                              evidence_strength: float,
                              evidence_direction: int, # +1 bestätigend, -1 widersprechend
                              evidence_certainty: float = 0.8 # Wie sicher ist die Evidenz selbst?
                              ) -> Tuple[float, float]:
        """
        Wendet ein vereinfachtes Bayes'sches Update auf eine Überzeugung an.

        Berücksichtigt kognitive Architektur (Verarbeitungsstil, Biases, Parameter).
        Aktualisiert sowohl die Stärke (belief) als auch die Gewissheit (certainty).

        Args:
            prior_belief: Vorherige Stärke der Überzeugung (0-1).
            prior_certainty: Vorherige Gewissheit über die Überzeugung (0-1).
            evidence_strength: Stärke des Beweises (0-1), wie stark die Evidenz ist.
            evidence_direction: Richtung des Beweises (+1 bestätigend, -1 widerlegend).
            evidence_certainty: Gewissheit der Evidenzquelle (0-1).

        Returns:
            Tuple[float, float]: Aktualisierte Überzeugungsstärke, Aktualisierte Gewissheit.
        """
        prior_belief = clip(prior_belief)
        prior_certainty = clip(prior_certainty)
        evidence_strength = clip(evidence_strength)
        evidence_certainty = clip(evidence_certainty)

        # --- Parameter abrufen ---
        prior_strength_factor = self.bayesian_parameters.get("prior_strength_factor", 0.5)
        update_rate = self.bayesian_parameters.get("update_rate", 0.5)
        anchoring_bias = self.cognitive_biases.get("anchoring_bias", 0.5)
        confirmation_bias = self.cognitive_biases.get("confirmation_bias", 0.5)

        # --- Gewichtung der Evidenz ---
        # Evidenz wird stärker gewichtet, wenn Quelle sicher ist und Agent Evidenz gut aufnimmt
        effective_evidence_weight = evidence_certainty * self.get_evidence_weight()

        # Confirmation Bias: Evidenz, die dem Prior widerspricht, wird abgeschwächt.
        is_confirming = (evidence_direction > 0 and prior_belief > 0.5) or \
                        (evidence_direction < 0 and prior_belief < 0.5) or \
                        (prior_belief == 0.5) # Neutral gilt als bestätigend

        if not is_confirming:
            effective_evidence_weight *= (1.0 - confirmation_bias * 0.5) # Abschwächung um bis zu 25%

        # --- Stärke-Update (vereinfachte Logik, inspiriert von Bayes) ---
        # Idee: Verschiebung in Richtung der Evidenz, moduliert durch Gewichte und Raten.
        # Die "Ziel"-Überzeugung, wenn nur die Evidenz zählen würde:
        evidence_target = 0.5 + 0.5 * evidence_direction * evidence_strength

        # Gewichteter Durchschnitt zwischen Prior und Evidenzziel
        # Das Gewicht des Priors hängt von seiner Stärke und Gewissheit ab
        prior_weight = prior_strength_factor * prior_certainty
        # Das Gewicht der Evidenz hängt von ihrer effektiven Gewichtung ab
        evidence_weight = effective_evidence_weight

        # Normalisierung der Gewichte (vereinfacht)
        total_weight = prior_weight + evidence_weight
        if total_weight == 0: total_weight = 1.0 # Vermeide Division durch Null
        norm_prior_weight = prior_weight / total_weight
        norm_evidence_weight = evidence_weight / total_weight

        # Aktualisierte Stärke als gewichteter Mittelwert (vor Update Rate)
        intermediate_belief = norm_prior_weight * prior_belief + norm_evidence_weight * evidence_target

        # Update Rate anwenden: Nur ein Teil der Änderung wird übernommen
        belief_change = (intermediate_belief - prior_belief) * update_rate

        # Anchoring Bias: Zieht die Änderung leicht zurück zum ursprünglichen Prior
        anchor_effect = (prior_belief - intermediate_belief) * anchoring_bias * 0.1 # Kleiner Effekt
        belief_change += anchor_effect

        new_belief = prior_belief + belief_change

        # --- Gewissheits-Update ---
        # Gewissheit steigt, wenn Evidenz den Prior bestätigt oder stark ist.
        # Gewissheit sinkt, wenn Evidenz dem Prior widerspricht.
        certainty_change = 0.0

        agreement_factor = 1.0 - abs(prior_belief - evidence_target) # 1 bei Übereinstimmung, 0 bei max. Unterschied
        conflict_factor = 1.0 - agreement_factor

        if is_confirming:
             # Stärkere Evidenz und höhere Evidenzsicherheit erhöhen Gewissheit stärker
             certainty_change = 0.05 * evidence_strength * effective_evidence_weight * update_rate
        else:
             # Widersprüchliche Evidenz reduziert Gewissheit, stärker bei starkem Konflikt
             certainty_change = -0.1 * conflict_factor * evidence_strength * effective_evidence_weight * update_rate

        # Hohe Gewissheit der Evidenzquelle kann Gewissheit auch bei Konflikt leicht steigern
        certainty_change += 0.02 * evidence_certainty * effective_evidence_weight * update_rate

        new_certainty = prior_certainty + certainty_change

        # Clipping der finalen Werte
        final_belief = clip(new_belief)
        final_certainty = clip(new_certainty, 0.01, 1.0) # Mindestgewissheit von 0.01

        return final_belief, final_certainty


    def __str__(self):
        """String-Repräsentation der kognitiven Architektur."""
        primary = self.primary_processing
        balance = self.processing_balance
        secondary = self.secondary_processing if self.secondary_processing and balance < 1.0 else "None"
        secondary_balance = f"({1-balance:.2f})" if secondary != "None" else ""
        return (f"CognitiveArch({primary} ({balance:.2f}) / {secondary} {secondary_balance} | "
                f"Biases: Conf={self.cognitive_biases['confirmation_bias']:.2f}, Avail={self.cognitive_biases['availability_bias']:.2f} | "
                f"Emo: React={self.emotional_parameters['emotional_reactivity']:.2f}, Regul={self.emotional_parameters['emotional_regulation']:.2f})")

# Block 3: Kernklasse - NeuralEthicalAgent
# Purpose: Definition des Agenten mit Überzeugungen, kognitiver Architektur und Verhalten.
# Änderungen: Konstanten verwendet, zentrales _update_belief_internal implementiert,
#              Logik in Entscheidungsfindung, Lernen, sozialem Einfluss und Reflexion überarbeitet,
#              um das zentrale Update zu verwenden und Konsistenz zu erhöhen.
#              Working Memory / Episodic Memory Einfluss rudimentär integriert.

# Importe wurden an den Anfang der Datei verlegt
# import numpy as np
# import pandas as pd
# import random
# from typing import List, Dict, Tuple, Optional, Set, Union, Any
# import logging

# Importe sind nicht mehr nötig, da die Klassen direkt in dieser Datei definiert sind
# from your_module import NeuralEthicalBelief, CognitiveArchitecture

# Forward declaration for type hinting EthicalScenario
# class EthicalScenario: pass


class NeuralEthicalAgent:
    """
    Repräsentiert einen ethischen Agenten mit neuronalen Verarbeitungsmodellen,
    Überzeugungen, Persönlichkeit und sozialen Verbindungen.
    """
    def __init__(self, agent_id: str, personality_traits: Optional[Dict[str, float]] = None):
        """
        Initialisiert einen neuronalen ethischen Agenten.

        Args:
            agent_id: Eindeutige ID des Agenten.
            personality_traits: Persönlichkeitsmerkmale (Big Five, 0-1). Wenn None, werden zufällige generiert.
        """
        self.agent_id = agent_id
        self.beliefs: Dict[str, NeuralEthicalBelief] = {}

        # --- Persönlichkeit ---
        self.personality_traits = personality_traits or self._generate_random_personality()

        # --- Kognitive Architektur ---
        # Wird basierend auf Persönlichkeit generiert
        self.cognitive_architecture = self._generate_cognitive_architecture()

        # --- Gedächtnis ---
        self.working_memory = {
            "capacity": WORKING_MEMORY_BASE_CAPACITY + int(WORKING_MEMORY_CONSC_CAPACITY_MOD * self.personality_traits["conscientiousness"]),
            "contents": [],  # Aktuelle Inhalte (z.B. kürzlich aktivierte Beliefs/Kontextelemente)
            "retention": WORKING_MEMORY_BASE_RETENTION + WORKING_MEMORY_CONSC_RETENTION_MOD * self.personality_traits["conscientiousness"]
        }
        self.episodic_memory: List[Dict[str, Any]] = [] # Liste von wichtigen Ereignissen/Erfahrungen

        # --- Historie ---
        self.decision_history: List[Dict[str, Any]] = []
        # Historien werden jetzt innerhalb von _update_belief_internal verwaltet
        self.belief_strength_history: Dict[str, List[float]] = {}
        self.belief_certainty_history: Dict[str, List[float]] = {}

        # --- Soziales ---
        self.social_connections: Dict[str, float] = {}  # other_agent_id -> connection_strength (0-1)
        self.group_identities: Dict[str, float] = {}  # group_name -> identification_strength (0-1)

        # --- Moralische Grundlagen ---
        self.moral_foundations = self._generate_random_moral_foundations()

        # --- Simulation State ---
        self.current_time: int = 0 # Wird von der Society gesetzt

    # --- Private Hilfsfunktionen ---
    def _generate_random_personality(self) -> Dict[str, float]:
        """Generiert zufällige Big Five Persönlichkeitsmerkmale."""
        traits = ["openness", "conscientiousness", "extroversion", "agreeableness", "neuroticism"]
        return {trait: np_random_beta(*PERSONALITY_DEFAULT_DIST) for trait in traits}

    def _generate_random_moral_foundations(self) -> Dict[str, float]:
        """Generiert zufällige Ausprägungen der moralischen Grundlagen."""
        foundations = ["care", "fairness", "loyalty", "authority", "purity", "liberty"]
        return {f: np_random_beta(*MORAL_FOUNDATIONS_DEFAULT_DIST) for f in foundations}

    def _generate_cognitive_architecture(self) -> CognitiveArchitecture:
        """
        Generiert eine zur Persönlichkeit passende kognitive Architektur.
        Verwendet Heuristiken, um Stile und Balance basierend auf Traits zuzuordnen.
        """
        personality = self.personality_traits
        primary = None
        secondary = None

        # Primären Stil wählen (basierend auf dominanten Traits - vereinfacht)
        if personality["openness"] > COGNITIVE_ARCH_OPENNESS_THRESHOLD and \
           personality["conscientiousness"] > COGNITIVE_ARCH_CONSCIENTIOUSNESS_THRESHOLD:
            primary = NeuralProcessingType.SYSTEMATIC
        elif personality["agreeableness"] > COGNITIVE_ARCH_AGREEABLENESS_THRESHOLD and \
             personality["extroversion"] > COGNITIVE_ARCH_EXTROVERSION_THRESHOLD:
            primary = random.choice([NeuralProcessingType.EMOTIONAL, NeuralProcessingType.NARRATIVE])
        elif personality["openness"] > COGNITIVE_ARCH_OPENNESS_THRESHOLD:
            primary = random.choice([NeuralProcessingType.ANALOGICAL, NeuralProcessingType.ASSOCIATIVE])
        elif personality["neuroticism"] > COGNITIVE_ARCH_NEUROTICISM_THRESHOLD:
            primary = random.choice([NeuralProcessingType.INTUITIVE, NeuralProcessingType.EMOTIONAL])

        if primary is None: # Fallback
             primary = NeuralProcessingType.get_random()

        # Sekundären Stil wählen (oft komplementär, aber mit Zufall)
        possible_secondary = [t for t in NeuralProcessingType.ALL_TYPES if t != primary]
        if not possible_secondary: # Sollte nicht passieren, aber sicher ist sicher
             secondary = None
        elif primary == NeuralProcessingType.SYSTEMATIC:
            secondary = random.choice([NeuralProcessingType.INTUITIVE, NeuralProcessingType.EMOTIONAL, NeuralProcessingType.ASSOCIATIVE])
        elif primary in [NeuralProcessingType.EMOTIONAL, NeuralProcessingType.INTUITIVE]:
            secondary = random.choice([NeuralProcessingType.SYSTEMATIC, NeuralProcessingType.ASSOCIATIVE, NeuralProcessingType.NARRATIVE])
        else:
            secondary = random.choice(possible_secondary)

        # Balance basierend auf Persönlichkeit
        balance = np.random.uniform(*COGNITIVE_ARCH_DEFAULT_BALANCE_RANGE) # Standardmäßig dominiert primärer Stil
        if primary == NeuralProcessingType.SYSTEMATIC:
            # Gewissenhafte erhöhen Dominanz des systematischen Stils
            balance = 0.5 + COGNITIVE_ARCH_CONSC_BALANCE_MOD * personality["conscientiousness"]
        elif primary in [NeuralProcessingType.EMOTIONAL, NeuralProcessingType.INTUITIVE]:
            # Neurotische erhöhen Dominanz emotionaler/intuitiver Stile
            balance = 0.5 + COGNITIVE_ARCH_NEUR_BALANCE_MOD * personality["neuroticism"]

        return CognitiveArchitecture(primary, secondary, clip(balance))

    def _add_to_history(self, belief_name: str, strength: float, certainty: float):
        """Fügt Werte zu den Belief-Historien hinzu."""
        if belief_name not in self.belief_strength_history:
            self.belief_strength_history[belief_name] = []
            self.belief_certainty_history[belief_name] = []
        self.belief_strength_history[belief_name].append(strength)
        self.belief_certainty_history[belief_name].append(certainty)

    def _update_belief_internal(self,
                                belief_name: str,
                                evidence_strength: float,
                                evidence_direction: int,
                                evidence_certainty: float = 0.8,
                                source: str = "unknown", # Woher kommt das Update? (experience, social, reflection, propagation)
                                context: Optional[Dict[str, Any]] = None
                               ) -> Tuple[float, float, float]:
        """
        Zentrale Methode zur Aktualisierung einer Überzeugung.

        Nutzt apply_bayesian_update der kognitiven Architektur für Stärke und Gewissheit.
        Aktualisiert auch optional die emotionale Valenz basierend auf dem Kontext.
        Protokolliert Änderungen in der Historie.

        Returns:
            Tuple[float, float, float]: Änderung der Stärke, neue Stärke, neue Gewissheit.
        """
        if belief_name not in self.beliefs:
            logging.warning(f"Agent {self.agent_id}: Versuch, nicht existierende Überzeugung '{belief_name}' zu aktualisieren.")
            return 0.0, 0.0, 0.0

        belief = self.beliefs[belief_name]
        old_strength = belief.strength
        old_certainty = belief.certainty
        old_valence = belief.emotional_valence

        # 1. Update von Stärke und Gewissheit via kognitiver Architektur
        new_strength, new_certainty = self.cognitive_architecture.apply_bayesian_update(
            prior_belief=old_strength,
            prior_certainty=old_certainty,
            evidence_strength=evidence_strength,
            evidence_direction=evidence_direction,
            evidence_certainty=evidence_certainty
        )

        belief.strength = new_strength
        belief.certainty = new_certainty
        strength_change = new_strength - old_strength

        # 2. Update der emotionalen Valenz (optional, z.B. bei starken Erfahrungen)
        new_valence = old_valence # Standard: keine Änderung
        if context and context.get("update_valence", False):
            valence_impact = context.get("valence_impact", 0.0) # Wie stark soll Valenz beeinflusst werden?
            if abs(valence_impact) > 0.01:
                # Emotionale Reaktivität beeinflusst, wie stark die Valenz angepasst wird
                reactivity = self.cognitive_architecture.emotional_parameters.get("emotional_reactivity", 0.5)
                # Annahme: Valenz nähert sich dem valence_impact an
                valence_change = (valence_impact - old_valence) * reactivity * EXPERIENCE_VALENCE_CHANGE_RATE
                new_valence = clip(old_valence + valence_change, -1.0, 1.0)
                belief.emotional_valence = new_valence

        # 3. Historie aktualisieren
        self._add_to_history(belief_name, new_strength, new_certainty)

        # 4. Episodisches Gedächtnis bei signifikanten Änderungen
        if abs(strength_change) > EXPERIENCE_SIGNIFICANT_CHANGE_THRESHOLD:
            self._add_to_episodic_memory(
                event_type="belief_change",
                details={
                    "belief": belief_name,
                    "change": strength_change,
                    "new_strength": new_strength,
                    "new_certainty": new_certainty,
                    "source": source,
                    "context": context # Speichere relevanten Kontext (z.B. Szenario ID)
                }
            )

        return strength_change, new_strength, new_certainty

    def _add_to_episodic_memory(self, event_type: str, details: Dict[str, Any]):
        """Fügt ein Ereignis zum episodischen Gedächtnis hinzu."""
        memory_item = {
            "time": self.current_time,
            "type": event_type,
            **details # Fügt alle Details aus dem Dict hinzu
        }
        self.episodic_memory.append(memory_item)
        # Optional: Begrenze die Größe des Gedächtnisses
        # if len(self.episodic_memory) > MAX_EPISODIC_MEMORY_SIZE:
        #     self.episodic_memory.pop(0) # Ältestes entfernen

    def _get_relevant_context(self, scenario: Optional['EthicalScenario'] = None) -> Dict[str, Any]:
        """
        Erstellt ein Kontext-Dictionary für kognitive Prozesse.
        Sammelt relevante Informationen aus Agentenzustand, Gedächtnis und Szenario.
        """
        context = {
            "time": self.current_time,
            # Agenten-interne Zustände
            "cognitive_dissonance": self.calculate_cognitive_dissonance(),
            "dominant_emotion": None, # TODO: Könnte aus Valenzen aktiver Beliefs abgeleitet werden
            # Szenario-spezifische Infos (falls vorhanden)
            "scenario_id": scenario.scenario_id if scenario else None,
            "scenario_description": scenario.description if scenario else None,
            "scenario_emotional_valence": scenario.emotional_valence if scenario else 0.0,
            "scenario_narrative_coherence": scenario.narrative_elements.get("coherence", 0.5) if scenario else 0.5,
            # Gedächtnis-basierte Infos
            "salient_concepts": self._get_salient_concepts_from_memory(),
            "recent_decision_outcomes": self._get_recent_outcomes(),
            # TODO: "anchor_belief_strength", "analogy_source_strength", "analogy_similarity" etc. könnten hier gesetzt werden
        }
        return context

    def _get_salient_concepts_from_memory(self, num_recent: int = 5) -> List[str]:
        """Identifiziert kürzlich aktivierte oder geänderte Überzeugungen aus dem Gedächtnis."""
        salient = set()
        # Kürzliche Aktivierungen (aus Working Memory - hier simuliert über Belief Activation)
        active_beliefs = sorted(self.beliefs.items(), key=lambda item: item[1].activation, reverse=True)
        for name, belief in active_beliefs[:self.working_memory["capacity"]]:
             if belief.activation > 0.1: # Nur wenn signifikant aktiv
                  salient.add(name)

        # Kürzliche Änderungen (aus Episodic Memory)
        for memory in reversed(self.episodic_memory[-num_recent:]):
            if memory["type"] == "belief_change":
                salient.add(memory["belief"])
            elif memory["type"] == "social_influence":
                 salient.update(memory.get("significant_changes", {}).keys())

        # Optional: Konzepte aus `working_memory.contents`, falls dieses aktiv genutzt wird.

        return list(salient)

    def _get_recent_outcomes(self, num_recent: int = 3) -> Dict[str, List[float]]:
         """Sammelt Ergebnisse (z.B. Belief-Änderungen) kürzlicher Entscheidungen."""
         outcomes = {}
         for memory in reversed(self.episodic_memory[-num_recent:]):
             if memory["type"] == "belief_change":
                 scenario_id = memory.get("context", {}).get("scenario_id")
                 if scenario_id:
                     if scenario_id not in outcomes:
                         outcomes[scenario_id] = []
                     outcomes[scenario_id].append(memory["change"])
         return outcomes


    # --- Öffentliche Methoden ---
    def add_belief(self, belief: NeuralEthicalBelief):
        """Fügt eine ethische Überzeugung hinzu und initialisiert Historie."""
        if belief.name in self.beliefs:
            logging.warning(f"Agent {self.agent_id}: Belief '{belief.name}' wird überschrieben.")
        self.beliefs[belief.name] = belief
        # Historie initialisieren
        self._add_to_history(belief.name, belief.strength, belief.certainty)

    def update_belief(self,
                      belief_name: str,
                      evidence_strength: float,
                      evidence_direction: int,
                      evidence_certainty: float = 0.8,
                      source: str = "external",
                      context: Optional[Dict[str, Any]] = None
                      ) -> Tuple[float, float, float]:
        """
        Öffentliche Methode zum Aktualisieren einer Überzeugung durch externe Evidenz.
        Wrapper um _update_belief_internal.
        """
        return self._update_belief_internal(belief_name, evidence_strength, evidence_direction,
                                            evidence_certainty, source, context)


    def add_social_connection(self, agent_id: str, connection_strength: float):
        """Fügt eine soziale Verbindung zu einem anderen Agenten hinzu."""
        self.social_connections[agent_id] = clip(connection_strength)

    def add_group_identity(self, group_name: str, identification_strength: float):
        """Fügt eine Gruppenidentität hinzu."""
        self.group_identities[group_name] = clip(identification_strength)

    def get_belief_vector(self, belief_names: Optional[List[str]] = None) -> np.ndarray:
        """Gibt einen Vektor mit Überzeugungsstärken zurück (optional gefiltert)."""
        if belief_names is None:
            belief_names = sorted(self.beliefs.keys())
        return np.array([self.beliefs[name].strength for name in belief_names if name in self.beliefs])

    def get_belief_names(self) -> List[str]:
        """Gibt die Namen aller Überzeugungen zurück."""
        return list(self.beliefs.keys())

    def calculate_cognitive_dissonance(self) -> float:
        """
        Berechnet die kognitive Dissonanz basierend auf widersprüchlichen direkten Verbindungen im Netzwerk.
        Höhere Dissonanz bei stark gegensätzlich verbundenen, sicheren Überzeugungen.
        """
        dissonance = 0.0
        processed_pairs = set()
        num_connections = 0

        for belief_name, belief in self.beliefs.items():
            for other_name, (influence, polarity) in belief.connections.items():
                # Nur Paare einmal zählen und nur wenn beide Beliefs existieren
                pair = tuple(sorted((belief_name, other_name)))
                if other_name in self.beliefs and pair not in processed_pairs:
                    other_belief = self.beliefs[other_name]
                    num_connections += 1
                    processed_pairs.add(pair)

                    # Dissonanz entsteht bei negativer Verbindung (*-1*)
                    # zwischen zwei starken (*strength1 * strength2*)
                    # und sicheren (*cert1 * cert2*) Überzeugungen.
                    # Der Einfluss der Verbindung skaliert die Dissonanz (*influence*).
                    if polarity < 0:
                        dissonance_contribution = (belief.strength * other_belief.strength *
                                                 belief.certainty * other_belief.certainty *
                                                 influence * abs(polarity))
                        dissonance += dissonance_contribution

        # Normalisierung (optional, hier basierend auf Anzahl der Verbindungen)
        # avg_dissonance = dissonance / num_connections if num_connections > 0 else 0.0

        # Anpassung basierend auf kognitiver Architektur (wie im Original)
        # Systematische Denker *spüren* Dissonanz stärker (oder versuchen eher, sie zu reduzieren)
        # Intuitive Denker tolerieren sie möglicherweise besser.
        if self.cognitive_architecture.primary_processing == NeuralProcessingType.SYSTEMATIC:
            dissonance *= 1.2
        elif self.cognitive_architecture.primary_processing == NeuralProcessingType.INTUITIVE:
            dissonance *= 0.8

        return clip(dissonance) # Dissonanz sollte im Bereich 0-1 liegen (oder zumindest positiv)


    def spreading_activation(self, seed_beliefs: Dict[str, float]):
        """
        Führt Spreading Activation im Überzeugungsnetzwerk durch.

        Args:
            seed_beliefs: Dictionary der Initial-Überzeugungen und ihrer Aktivierungslevel.
        """
        if not self.beliefs: return # Nichts zu aktivieren

        self.current_time += 1 # Interner Zeitschritt für Aktivierungsdynamik

        # 1. Initial-Aktivierung anwenden
        for belief_name, activation_level in seed_beliefs.items():
            if belief_name in self.beliefs:
                self.beliefs[belief_name].activate(activation_level, self.current_time)

        # 2. Spreading activation über mehrere Runden (z.B. 2 Runden)
        num_spread_rounds = 2
        for _ in range(num_spread_rounds):
            # Aktuelle Aktivierungen speichern, um sie in dieser Runde zu verwenden
            current_activations = {name: belief.activation for name, belief in self.beliefs.items()}
            next_activations = current_activations.copy() # Kopie für Updates

            # Aktivierung verbreiten
            for belief_name, belief in self.beliefs.items():
                source_activation = current_activations.get(belief_name, 0.0)

                # Nur von ausreichend aktiven Knoten aus verbreiten
                if source_activation > 0.1:
                    # Verbreitung an verbundene Beliefs
                    for conn_name, (strength, polarity) in belief.connections.items():
                        if conn_name in self.beliefs:
                            # Berechne die zu übertragende Aktivierung
                            # Verwende die gespeicherte source_activation für Konsistenz in der Runde
                            spread_activation = source_activation * strength * ASSOCIATIVE_SPREAD_DAMPING # Dämpfung

                            # Aktivierung weitergeben (akkumuliert in next_activations)
                            # Polarität berücksichtigen: Positive verstärken, negative hemmen (Reduktion)
                            current_target_activation = next_activations.get(conn_name, 0.0)
                            if polarity > 0:
                                next_activations[conn_name] = current_target_activation + spread_activation
                            else:
                                # Hemmung: Reduziert Aktivierung, aber nicht unter 0
                                inhibition_factor = spread_activation * 0.5 # Schwächere Hemmung als Aktivierung
                                next_activations[conn_name] = max(0.0, current_target_activation * (1.0 - inhibition_factor))

                    # Verbreitung an assoziierte Konzepte (falls assoziativer Stil aktiv)
                    if self.cognitive_architecture.primary_processing == NeuralProcessingType.ASSOCIATIVE or \
                       self.cognitive_architecture.secondary_processing == NeuralProcessingType.ASSOCIATIVE:
                        for concept, assoc_strength in belief.associated_concepts.items():
                            # Suche andere Beliefs, die dieses Konzept auch haben
                            for other_name, other_belief in self.beliefs.items():
                                if other_name != belief_name and concept in other_belief.associated_concepts:
                                    # Stärke der assoziativen Verbindung
                                    cross_assoc_strength = assoc_strength * other_belief.associated_concepts[concept]
                                    # Zu übertragende Aktivierung
                                    assoc_spread = source_activation * cross_assoc_strength * ASSOCIATIVE_CONCEPT_SPREAD_DAMPING
                                    # An Ziel-Belief weitergeben
                                    current_target_activation = next_activations.get(other_name, 0.0)
                                    next_activations[other_name] = current_target_activation + assoc_spread

            # Aktivierungen für die nächste Runde/Finale Aktivierung aktualisieren
            for name, next_act in next_activations.items():
                 # Begrenze Aktivierung nach oben
                 self.beliefs[name].activation = clip(next_act, 0.0, 2.0)

        # 3. Decay nach jeder Runde Spreading (simuliert Vergessen/Beruhigung)
        for belief in self.beliefs.values():
            belief.activation *= BELIEF_ACTIVATION_DECAY_RATE


    def make_decision(self, scenario: 'EthicalScenario') -> Dict[str, Any]:
        """
        Trifft eine Entscheidung in einem ethischen Szenario.

        Prozess:
        1. Kontext erstellen.
        2. Relevante Überzeugungen aktivieren (Spreading Activation).
        3. Aktuelle Überzeugungsstärken durch kognitive Architektur verarbeiten ("Wahrnehmung").
        4. Optionen basierend auf verarbeiteten Überzeugungen, Moral, Risiko etc. bewerten.
        5. Finale Entscheidung treffen (ggf. mit Rauschen/Bias).
        6. Entscheidung und Zustand protokollieren.

        Returns:
            Dict mit der Entscheidung, Scores, Begründungen und Agentenzustand.
        """
        # --- 1. Kontext erstellen ---
        context = self._get_relevant_context(scenario)
        # Füge spezifische Infos für dieses Szenario hinzu
        context["relevant_beliefs_for_scenario"] = list(scenario.relevant_beliefs.keys())

        # --- 2. Spreading Activation ---
        # Aktiviere die für das Szenario relevanten Überzeugungen
        seed_activations = {name: relevance for name, relevance in scenario.relevant_beliefs.items() if name in self.beliefs}
        if seed_activations:
             self.spreading_activation(seed_activations)
        # Update context mit aktuellen Aktivierungsleveln (könnte für manche Stile nützlich sein)
        context["current_belief_activations"] = {name: b.activation for name, b in self.beliefs.items()}


        # --- 3. Überzeugungen durch kognitive Architektur verarbeiten ---
        # Nimm die *aktuellen* Stärken als Input für die Verarbeitung
        belief_inputs = {name: belief.strength for name, belief in self.beliefs.items()}
        # Verarbeite diese Inputs basierend auf Stil, Biases etc.
        processed_beliefs = self.cognitive_architecture.process_information(belief_inputs, context)
        context["processed_beliefs"] = processed_beliefs # Füge zum Kontext hinzu

        # --- 4. Optionen bewerten ---
        option_scores: Dict[str, Dict[str, Any]] = {}
        ingroup_bias = self.cognitive_architecture.cognitive_biases.get("ingroup_bias", 0.5)

        for option_name, option_impacts in scenario.options.items():
            raw_score = 0.0
            justifications = {}

            # a) Moralischer Beitrag
            moral_contribution = 0.0
            for foundation, strength in self.moral_foundations.items():
                # Hat diese Option Implikationen für diese moralische Grundlage?
                impact = scenario.moral_implications.get(option_name, {}).get(foundation, 0.0)
                if impact != 0:
                    contribution = strength * impact
                    moral_contribution += contribution
                    justifications[f"moral_{foundation}"] = contribution
            raw_score += moral_contribution * DECISION_MORAL_WEIGHT
            justifications["total_moral_score"] = moral_contribution * DECISION_MORAL_WEIGHT

            # b) Überzeugungsbeitrag (basierend auf *verarbeiteten* Beliefs)
            belief_contribution = 0.0
            for belief_name, impact in option_impacts.items():
                if belief_name in processed_beliefs:
                    # Nutze den *verarbeiteten* Wert der Überzeugung
                    processed_strength = processed_beliefs[belief_name]
                    contribution = processed_strength * impact
                    belief_contribution += contribution
                    justifications[belief_name] = contribution # Begründung: Einfluss durch Belief X
            raw_score += belief_contribution # Gewichtung implizit durch Anzahl/Stärke der Beliefs
            justifications["total_belief_score"] = belief_contribution

            # c) Risikoabschätzung (basierend auf Persönlichkeit)
            risk_score = 0.0
            option_risk = scenario.option_attributes.get(option_name, {}).get("risks", 0.0)
            if option_risk > 0:
                 # Risikoaversion skaliert mit Offenheit (weniger offen -> averser)
                 risk_aversion = DECISION_RISK_AVERSION_BASE - DECISION_RISK_OPENNESS_MOD * self.personality_traits["openness"]
                 risk_penalty = -option_risk * clip(risk_aversion) # Negativer Score für Risiko
                 risk_score += risk_penalty
                 justifications["risk_consideration"] = risk_penalty
            raw_score += risk_score

            # d) Gruppennormen (beeinflusst durch Ingroup Bias)
            group_norm_score = 0.0
            option_group_norms = scenario.option_attributes.get(option_name, {}).get("group_norms", {})
            if option_group_norms:
                 total_norm_alignment = 0.0
                 num_relevant_groups = 0
                 for group, norm_alignment in option_group_norms.items():
                      if group in self.group_identities:
                           identification = self.group_identities[group]
                           # Verstärkung durch Ingroup-Bias
                           biased_alignment = norm_alignment * (1.0 + ingroup_bias * identification)
                           total_norm_alignment += identification * biased_alignment
                           num_relevant_groups += 1

                 # Durchschnittlicher Einfluss gewichtet mit Identifikation
                 avg_norm_influence = total_norm_alignment / num_relevant_groups if num_relevant_groups > 0 else 0.0
                 group_norm_score = avg_norm_influence * DECISION_GROUP_NORM_WEIGHT
                 justifications["group_norms"] = group_norm_score
            raw_score += group_norm_score

            # Speichere den Roh-Score und die Begründungen
            option_scores[option_name] = {
                "raw_score": raw_score,
                "justifications": justifications
            }

        # --- 5. Finale Entscheidung treffen ---
        final_decision = self._finalize_decision(scenario, option_scores, context)

        # --- 6. Entscheidung protokollieren ---
        self.decision_history.append(final_decision)

        # Episodisches Gedächtnis aktualisieren (nur wichtige Entscheidungen)
        if final_decision["cognitive_dissonance"] > DECISION_SIGNIFICANT_DISSONANCE_THRESHOLD or \
           abs(final_decision["confidence"]) > DECISION_HIGH_CONFIDENCE_THRESHOLD:
            self._add_to_episodic_memory(
                event_type="significant_decision",
                details={
                    "scenario": scenario.scenario_id,
                    "decision": final_decision["chosen_option"],
                    "confidence": final_decision["confidence"],
                    "dissonance": final_decision["cognitive_dissonance"],
                    "option_scores": final_decision["final_scores"] # Speichere die finalen Scores
                }
            )

        return final_decision


    def _finalize_decision(self, scenario: 'EthicalScenario',
                         option_scores: Dict[str, Dict],
                         context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Finalisiert die Entscheidung basierend auf Scores, kognitiver Architektur und Rauschen.

        Args:
            scenario: Das aktuelle Szenario.
            option_scores: Dictionary mit Roh-Scores und Begründungen pro Option.
            context: Der für die Entscheidung relevante Kontext.

        Returns:
            Dictionary mit der finalen Entscheidung und Metadaten.
        """
        options = list(option_scores.keys())
        raw_scores = np.array([option_scores[opt]["raw_score"] for opt in options])
        adjusted_scores = raw_scores.copy() # Start mit Roh-Scores

        # --- Modifikatoren basierend auf kognitivem Stil / Biases ---
        # Diese repräsentieren eher prozedurale Einflüsse als direkte Bewertungsänderungen
        randomness_factor = DECISION_NOISE_STD_DEV_DEFAULT
        primary_style = self.cognitive_architecture.primary_processing
        personality = self.personality_traits

        if primary_style == NeuralProcessingType.SYSTEMATIC:
            randomness_factor = DECISION_NOISE_STD_DEV_SYSTEMATIC
            # Konsistenz-Boost: Bevorzuge Optionen, die kürzlich in ähnlichen Szenarien gewählt wurden
            consistency_boost = self._calculate_consistency_boost(scenario, options)
            adjusted_scores += consistency_boost
            option_scores = self._add_justification(option_scores, options, "consistency_boost", consistency_boost)

        elif primary_style == NeuralProcessingType.INTUITIVE:
            randomness_factor = DECISION_NOISE_STD_DEV_INTUITIVE
            # Verfügbarkeitsheuristik / Recency-Boost: Bevorzuge Optionen, die generell kürzlich gewählt wurden
            recency_boost = self._calculate_recency_boost(options)
            availability_bias = self.cognitive_architecture.cognitive_biases.get("availability_bias", 0.5)
            adjusted_scores += recency_boost * availability_bias # Moduliert durch Bias
            option_scores = self._add_justification(option_scores, options, "recency_boost", recency_boost * availability_bias)

        elif primary_style == NeuralProcessingType.EMOTIONAL:
            # Emotionaler Boost: Bevorzuge Optionen, die positive emotionale Reaktionen hervorrufen
            emotion_boost = self._calculate_emotion_boost(scenario, options, context)
            emotional_reactivity = self.cognitive_architecture.emotional_parameters.get("emotional_reactivity", 0.5)
            adjusted_scores += emotion_boost * emotional_reactivity # Moduliert durch Reaktivität
            option_scores = self._add_justification(option_scores, options, "emotion_boost", emotion_boost * emotional_reactivity)

        # TODO: Einflüsse für andere Stile hinzufügen (z.B. Narrative: Kohärenz-Boost?)

        # --- Zufallskomponente hinzufügen ---
        # Simuliert Unsicherheit, nicht modellierte Faktoren, freier Wille?
        decision_noise = np.random.normal(0, randomness_factor, len(adjusted_scores))
        final_scores = adjusted_scores + decision_noise

        # --- Option wählen ---
        chosen_index = np.argmax(final_scores)
        chosen_option = options[chosen_index]

        # --- Konfidenz berechnen ---
        # Basierend auf dem Abstand zum nächstbesten Score (nach Hinzufügen von Rauschen)
        confidence = 0.5
        if len(final_scores) > 1:
            sorted_scores = np.sort(final_scores)[::-1] # Absteigend sortieren
            score_diff = sorted_scores[0] - sorted_scores[1]
            # Tanh-Skalierung für Bereich (-1, 1), sensibler bei kleinen Unterschieden
            confidence = np.tanh(score_diff * 2.0) # Faktor 2 verstärkt Sensitivität

        # --- Belief-Update während Deliberation (Optional, aktuell DEAKTIVIERT) ---
        belief_updates_during_decision = {}
        if DECISION_UPDATE_DURING_DELIBERATION:
            pass # Logik wurde entfernt, um Klarheit zu verbessern (Updates primär nach Erfahrung)
                 # Hier könnte man simulieren, dass das Nachdenken selbst Beliefs leicht anpasst


        # --- Ergebnis zusammenstellen ---
        return {
            "agent_id": self.agent_id,
            "scenario_id": scenario.scenario_id,
            "chosen_option": chosen_option,
            "confidence": float(confidence),
            "cognitive_dissonance": self.calculate_cognitive_dissonance(),
            "option_scores_raw": {opt: data["raw_score"] for opt, data in option_scores.items()},
            "option_scores_adjusted": dict(zip(options, adjusted_scores.tolist())),
            "final_scores": dict(zip(options, final_scores.tolist())),
            "justifications": {opt: data["justifications"] for opt, data in option_scores.items()},
            "belief_updates_during_decision": belief_updates_during_decision,
            "context_at_decision": context, # Speichere den Kontext für spätere Analyse
            "timestamp": self.current_time
        }

    def _add_justification(self, scores_dict, options, key, values):
         """Hilfsfunktion zum Hinzufügen von Boosts zu den Justifications."""
         for i, option in enumerate(options):
             if key not in scores_dict[option]["justifications"]:
                  scores_dict[option]["justifications"][key] = 0.0
             scores_dict[option]["justifications"][key] += values[i]
         return scores_dict

    def _calculate_consistency_boost(self, scenario: 'EthicalScenario', options: List[str]) -> np.ndarray:
        """Belohnt Optionen, die in *ähnlichen/gleichen* Szenarien kürzlich gewählt wurden."""
        boost = np.zeros(len(options))
        # Suche in den letzten N Entscheidungen
        lookback = 10
        for decision in self.decision_history[-lookback:]:
            # Nur gleiches Szenario berücksichtigen
            if decision.get("scenario_id") == scenario.scenario_id:
                past_option = decision.get("chosen_option")
                if past_option in options:
                    option_index = options.index(past_option)
                    # Zeitlicher Abfall: Neuere Entscheidungen haben mehr Gewicht
                    recency = 1.0 - (self.current_time - decision["timestamp"]) / (lookback * 2.0) # Linearer Abfall über doppelte Lookback-Periode
                    boost[option_index] += DECISION_CONSISTENCY_BOOST_FACTOR * max(0.1, recency)
        return boost

    def _calculate_recency_boost(self, options: List[str]) -> np.ndarray:
        """Belohnt Optionen, die *generell* kürzlich gewählt wurden (Verfügbarkeit)."""
        boost = np.zeros(len(options))
        lookback = 5
        for decision in self.decision_history[-lookback:]:
            past_option = decision.get("chosen_option")
            if past_option in options:
                option_index = options.index(past_option)
                recency = 1.0 - (self.current_time - decision["timestamp"]) / (lookback * 2.0)
                boost[option_index] += DECISION_RECENCY_BOOST_FACTOR * max(0.1, recency)
        return boost

    def _calculate_emotion_boost(self, scenario: 'EthicalScenario', options: List[str], context: Dict[str, Any]) -> np.ndarray:
        """Beeinflusst Scores basierend auf der emotionalen Valenz der Konsequenzen für Beliefs."""
        boost = np.zeros(len(options))
        processed_beliefs = context.get("processed_beliefs", {}) # Verwende die verarbeiteten Beliefs

        for i, option in enumerate(options):
            option_emotion_valence = 0.0
            num_relevant_beliefs = 0
            # Gehe durch die Auswirkungen der Option auf Beliefs
            for belief_name, impact in scenario.options[option].items():
                if belief_name in self.beliefs:
                    belief = self.beliefs[belief_name]
                    # Der "emotionale Wert" dieser Konsequenz:
                    # Positiver Impact auf pos. Belief -> gut (+)
                    # Negativer Impact auf pos. Belief -> schlecht (-)
                    # Positiver Impact auf neg. Belief -> schlecht (-)
                    # Negativer Impact auf neg. Belief -> gut (+)
                    # -> Multiplikation von Impact und Valenz
                    valence_contribution = impact * belief.emotional_valence
                    option_emotion_valence += valence_contribution
                    num_relevant_beliefs += 1

            # Durchschnittliche Valenz der Option
            avg_option_valence = option_emotion_valence / num_relevant_beliefs if num_relevant_beliefs > 0 else 0.0

            # Negativitätsverzerrung anwenden
            negativity_bias = self.cognitive_architecture.emotional_parameters.get("negativity_bias", 0.6)
            if avg_option_valence < 0:
                # Verstärkt negative Valenz
                final_valence = avg_option_valence * (1.0 + (negativity_bias - 0.5) * 2) # Skaliert -1..+1 -> Faktor 0..2
            else:
                # Dämpft positive Valenz (wenn Negativity Bias > 0.5)
                final_valence = avg_option_valence * (1.0 - (negativity_bias - 0.5) * 2)

            boost[i] = final_valence * DECISION_EMOTION_BOOST_FACTOR

        return boost

    # --- Lernen und Anpassung ---
    def update_beliefs_from_experience(self, scenario: 'EthicalScenario', chosen_option: str) -> Dict[str, float]:
        """
        Aktualisiert Überzeugungen basierend auf dem (simulierten) Ergebnis der Entscheidung.
        Nutzt das zentrale _update_belief_internal.

        Args:
            scenario: Das Szenario, in dem die Entscheidung getroffen wurde.
            chosen_option: Die vom Agenten gewählte Option.

        Returns:
            Dictionary mit den resultierenden Überzeugungsänderungen (belief_name -> stärke_änderung).
        """
        if chosen_option not in scenario.outcome_feedback:
            return {} # Kein Feedback für diese Option definiert

        belief_changes = {}
        feedback_data = scenario.outcome_feedback[chosen_option]

        # Lernrate basierend auf kognitiver Architektur und Persönlichkeit
        base_learning_rate = EXPERIENCE_LEARNING_RATE_BASE
        style_mod = 1.0
        if self.cognitive_architecture.primary_processing == NeuralProcessingType.SYSTEMATIC:
             style_mod = EXPERIENCE_LEARNING_RATE_SYS_MOD
        elif self.cognitive_architecture.primary_processing == NeuralProcessingType.ASSOCIATIVE:
             style_mod = EXPERIENCE_LEARNING_RATE_ASSOC_MOD
        # Offenheit beeinflusst Lernrate (0..1 -> 0.8..1.2)
        openness_mod = scale_value(self.personality_traits["openness"], 0, 1, *EXPERIENCE_LEARNING_OPENNESS_MOD_RANGE)
        learning_rate = base_learning_rate * style_mod * openness_mod

        for belief_name, feedback_value in feedback_data.items():
            if belief_name in self.beliefs:
                # Feedback-Wert als Evidenz interpretieren
                evidence_strength = abs(feedback_value) * learning_rate # Skalierte Stärke
                evidence_direction = np.sign(feedback_value)
                # Annahme: Direkte Erfahrung hat hohe Gewissheit
                evidence_certainty = 0.9

                # Kontext für das Update
                update_context = {
                    "scenario_id": scenario.scenario_id,
                    "chosen_option": chosen_option,
                    "feedback_value": feedback_value,
                    "update_valence": abs(feedback_value) > EXPERIENCE_VALENCE_CHANGE_THRESHOLD, # Valenz nur bei starkem Feedback ändern
                    "valence_impact": np.sign(feedback_value) * 0.5 # Zielvalenz bei starkem Feedback
                }

                # Zentrales Update durchführen
                strength_change, _, _ = self._update_belief_internal(
                    belief_name=belief_name,
                    evidence_strength=evidence_strength,
                    evidence_direction=int(evidence_direction),
                    evidence_certainty=evidence_certainty,
                    source="experience",
                    context=update_context
                )
                belief_changes[belief_name] = strength_change

        # Änderungen propagieren (optional, aber konsistent mit Original)
        propagated = self._propagate_belief_changes(belief_changes)
        belief_changes.update(propagated)

        return belief_changes


    def _propagate_belief_changes(self, initial_changes: Dict[str, float]) -> Dict[str, float]:
        """
        Verbreitet initiale Überzeugungsänderungen durch das Netzwerk.
        Nutzt *lineare* Propagation, moduliert durch kognitiven Stil (KEIN Bayes hier).
        Dies simuliert eine strukturelle Anpassung des Netzwerks nach einer Änderung.
        """
        propagated_changes = {}
        if not initial_changes: return propagated_changes

        # Propagationsstärke basierend auf kognitiver Architektur
        propagation_strength = PROPAGATION_STRENGTH_DEFAULT
        primary_style = self.cognitive_architecture.primary_processing
        if primary_style == NeuralProcessingType.ASSOCIATIVE:
            propagation_strength = PROPAGATION_STRENGTH_ASSOC_MOD
        elif primary_style == NeuralProcessingType.SYSTEMATIC:
            propagation_strength = PROPAGATION_STRENGTH_SYS_MOD

        # Änderungen durch Verbindungen weitergeben
        changes_to_propagate = initial_changes.copy()
        processed_in_round = set(initial_changes.keys()) # Verhindert Endlosschleifen in einem Schritt

        # Nur eine Runde Propagation hier (vereinfacht)
        for belief_name, change in changes_to_propagate.items():
             if belief_name in self.beliefs:
                 belief = self.beliefs[belief_name]
                 for conn_name, (influence, polarity) in belief.connections.items():
                     if conn_name in self.beliefs and conn_name not in processed_in_round:
                         # Berechne die weiterzugebende Änderung
                         propagated_change = change * influence * polarity * propagation_strength

                         # Assoziative Verstärkung (falls Stil aktiv)
                         if (primary_style == NeuralProcessingType.ASSOCIATIVE or
                             self.cognitive_architecture.secondary_processing == NeuralProcessingType.ASSOCIATIVE):
                             # Prüfe auf gemeinsame assoziierte Konzepte
                             common_concepts = set(belief.associated_concepts.keys()) & \
                                               set(self.beliefs[conn_name].associated_concepts.keys())
                             if common_concepts:
                                 # Verstärkung basierend auf max. gemeinsamer Assoziationsstärke
                                 max_assoc_boost = 0
                                 for concept in common_concepts:
                                     boost = (belief.associated_concepts[concept] *
                                              self.beliefs[conn_name].associated_concepts[concept])
                                     max_assoc_boost = max(max_assoc_boost, boost)
                                 propagated_change *= (1.0 + PROPAGATION_ASSOC_CONCEPT_BOOST * max_assoc_boost)

                         # Wende die lineare Änderung an (KEIN Bayes)
                         target_belief = self.beliefs[conn_name]
                         old_strength = target_belief.strength
                         new_strength = clip(old_strength + propagated_change)

                         if abs(new_strength - old_strength) > 0.001: # Nur signifikante Änderungen speichern
                             target_belief.strength = new_strength
                             # Gewissheit wird durch Propagation nicht direkt geändert (Annahme)
                             # Historie aktualisieren
                             self._add_to_history(conn_name, new_strength, target_belief.certainty)
                             # Änderung für Rückgabe speichern
                             propagated_changes[conn_name] = propagated_changes.get(conn_name, 0.0) + (new_strength - old_strength)
                             # Markieren, um Doppelverarbeitung in dieser Runde zu vermeiden
                             # (Einfache Lösung, komplexere Graphenalgorithmen wären besser)
                             # processed_in_round.add(conn_name) # Aktivieren für komplexere Propagation

        return propagated_changes


    def update_from_social_influence(self, other_agent: 'NeuralEthicalAgent',
                                     influenced_beliefs: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Aktualisiert Überzeugungen basierend auf sozialem Einfluss eines anderen Agenten.
        Nutzt das zentrale _update_belief_internal.

        Args:
            other_agent: Der Agent, der Einfluss ausübt.
            influenced_beliefs: Optionale Liste von Beliefs, die beeinflusst werden sollen.
                                Wenn None, werden alle gemeinsamen Beliefs betrachtet.

        Returns:
            Dictionary mit Überzeugungsänderungen (belief_name -> stärke_änderung).
        """
        if other_agent.agent_id not in self.social_connections:
            return {} # Kein Einfluss ohne Verbindung

        connection_strength = self.social_connections[other_agent.agent_id]
        belief_changes = {}

        # --- Soziale Lernrate berechnen ---
        base_social_learning_rate = SOCIAL_LEARNING_RATE_BASE
        # Stil-Einfluss
        style_factor = 1.0
        if self.cognitive_architecture.primary_processing == NeuralProcessingType.NARRATIVE:
             style_factor = SOCIAL_LEARNING_NARRATIVE_FACTOR
        elif self.cognitive_architecture.primary_processing == NeuralProcessingType.SYSTEMATIC:
             style_factor = SOCIAL_LEARNING_SYSTEMATIC_FACTOR
        # Persönlichkeits-Einfluss (Verträglichkeit)
        agreeableness_mod = scale_value(self.personality_traits["agreeableness"], 0, 1, *SOCIAL_LEARNING_AGREEABLENESS_FACTOR_RANGE)
        # Autoritäts-Bias
        authority_bias = self.cognitive_architecture.cognitive_biases.get("authority_bias", 0.5)
        perceived_authority = 0.0 # TODO: Mechanismus zur Bestimmung von Autorität fehlt noch
        authority_mod = 1.0 + authority_bias * perceived_authority * SOCIAL_LEARNING_AUTHORITY_BIAS_MOD

        social_learning_rate = base_social_learning_rate * style_factor * agreeableness_mod * connection_strength * authority_mod

        # --- Zu beeinflussende Überzeugungen identifizieren ---
        beliefs_to_consider = set(self.beliefs.keys()) & set(other_agent.beliefs.keys())
        if influenced_beliefs is not None:
            beliefs_to_consider &= set(influenced_beliefs)

        # --- Überzeugungen aktualisieren ---
        ingroup_bias = self.cognitive_architecture.cognitive_biases.get("ingroup_bias", 0.5)
        dogmatism = self.cognitive_architecture.cognitive_biases.get("dogmatism", 0.5) # Neu hinzugefügt

        for belief_name in beliefs_to_consider:
            my_belief = self.beliefs[belief_name]
            other_belief = other_agent.beliefs[belief_name]

            # Differenz als Basis für Evidenz
            strength_diff = other_belief.strength - my_belief.strength
            if abs(strength_diff) < SOCIAL_LEARNING_MIN_CHANGE_THRESHOLD:
                 continue # Ignoriere minimale Unterschiede

            # Evidenzstärke basiert auf Differenz und Gewissheit des anderen
            evidence_strength_base = abs(strength_diff)
            other_certainty_factor = scale_value(other_belief.certainty, 0, 1, SOCIAL_LEARNING_CERTAINTY_FACTOR_OTHER_MOD, 1.0) # 0.5 bis 1.0

            # Eigene Gewissheit reduziert Anfälligkeit
            self_certainty_factor = 1.0 - scale_value(my_belief.certainty, 0, 1, 0.0, SOCIAL_LEARNING_CERTAINTY_FACTOR_SELF_MOD) # 1.0 bis 0.5

            # Gruppenidentität (Ingroup Bias)
            group_weight = 1.0
            common_groups = set(self.group_identities.keys()) & set(other_agent.group_identities.keys())
            if common_groups:
                max_shared_identity = 0.0
                for group in common_groups:
                    my_id = self.group_identities[group]
                    other_id = other_agent.group_identities[group]
                    # Stärkster gemeinsamer Nenner zählt
                    max_shared_identity = max(max_shared_identity, min(my_id, other_id))
                group_weight = 1.0 + ingroup_bias * max_shared_identity * SOCIAL_LEARNING_INGROUP_BIAS_MOD

            # Dogmatismus reduziert Einfluss bei großen Differenzen
            dogmatism_factor = 1.0
            if abs(strength_diff) > 0.5: # Großer Unterschied
                dogmatism_factor = 1.0 - dogmatism * SOCIAL_LEARNING_DOGMATISM_FACTOR

            # Finale Evidenzstärke für das Update
            final_evidence_strength = (evidence_strength_base *
                                       other_certainty_factor *
                                       self_certainty_factor *
                                       group_weight *
                                       dogmatism_factor *
                                       social_learning_rate) # Skaliert mit Lernrate

            # Richtung der Evidenz
            evidence_direction = np.sign(strength_diff)

            # Gewissheit der "sozialen Evidenz" (basiert auf Verbindung und Gewissheit des Senders)
            evidence_certainty = clip(connection_strength * other_belief.certainty)

            # Kontext für Update
            update_context = {
                "source_agent": other_agent.agent_id,
                "connection_strength": connection_strength,
                "strength_diff": strength_diff,
                "update_valence": abs(strength_diff) > SOCIAL_LEARNING_SIGNIFICANT_VALENCE_CHANGE_THRESHOLD, # Valenz nur bei signifikanter Änderung anpassen
                 "valence_impact": other_belief.emotional_valence # Zielvalenz ist die des anderen Agenten
            }

            # Zentrales Update durchführen
            strength_change, _, _ = self._update_belief_internal(
                belief_name=belief_name,
                evidence_strength=final_evidence_strength,
                evidence_direction=int(evidence_direction),
                evidence_certainty=evidence_certainty,
                source="social",
                context=update_context
            )
            if abs(strength_change) > SOCIAL_LEARNING_MIN_CHANGE_THRESHOLD:
                 belief_changes[belief_name] = strength_change

        # Episodisches Gedächtnis bei signifikantem Einfluss
        significant_changes_dict = {k: v for k, v in belief_changes.items() if abs(v) >= SOCIAL_LEARNING_SIGNIFICANT_INFLUENCE_THRESHOLD}
        if significant_changes_dict:
            self._add_to_episodic_memory(
                event_type="social_influence",
                details={
                    "from_agent": other_agent.agent_id,
                    "significant_changes": significant_changes_dict,
                    "all_changes": belief_changes # Optional: Alle Änderungen speichern
                }
            )

        return belief_changes


    def reflect_on_experiences(self) -> Dict[str, float]:
        """
        Reflektiert über vergangene Erfahrungen (aus episodischem Gedächtnis)
        und konsolidiert Überzeugungen. Nutzt das zentrale _update_belief_internal.
        Primär für Agenten mit bestimmten kognitiven Stilen.

        Returns:
            Dictionary mit Überzeugungsänderungen durch Reflexion.
        """
        belief_changes = {}
        primary_style = self.cognitive_architecture.primary_processing
        secondary_style = self.cognitive_architecture.secondary_processing

        # Prüfen, ob der Agent überhaupt reflektiert
        can_reflect = (primary_style in REFLECTION_ENABLED_STYLES or
                      (secondary_style in REFLECTION_ENABLED_STYLES and self.cognitive_architecture.processing_balance < 0.8)) # Sekundär nur wenn nicht völlig dominiert

        if not self.episodic_memory or not can_reflect:
            return {}

        # --- Reflexionsstärke bestimmen ---
        reflection_strength = REFLECTION_STRENGTH_BASE
        if primary_style == NeuralProcessingType.SYSTEMATIC: # Systematische reflektieren stärker
             reflection_strength = REFLECTION_STRENGTH_SYS_MOD
        # Offenheit beeinflusst Reflexionsstärke
        openness_mod = scale_value(self.personality_traits["openness"], 0, 1, *REFLECTION_OPENNESS_MOD_RANGE)
        reflection_strength *= openness_mod

        # --- Relevante Erinnerungen auswählen ---
        num_memories_to_consider = max(REFLECTION_MIN_MEMORY_ITEMS, int(len(self.episodic_memory) * REFLECTION_MEMORY_FRACTION))
        recent_memories = self.episodic_memory[-num_memories_to_consider:]

        # --- Analyse der Erinnerungen ---
        belief_related_memories = {} # belief_name -> List[change_value]
        for memory in recent_memories:
            if memory["type"] == "belief_change":
                belief = memory.get("belief")
                change = memory.get("change")
                if belief and change is not None:
                    if belief not in belief_related_memories: belief_related_memories[belief] = []
                    belief_related_memories[belief].append(change)
            elif memory["type"] == "social_influence":
                changes = memory.get("significant_changes", {})
                for belief, change in changes.items():
                     if belief not in belief_related_memories: belief_related_memories[belief] = []
                     belief_related_memories[belief].append(change)
            # TODO: Andere Gedächtnistypen könnten auch analysiert werden

        # --- Überzeugungen basierend auf Konsistenz konsolidieren ---
        for belief_name, changes in belief_related_memories.items():
             if belief_name in self.beliefs and len(changes) >= 2: # Mindestens zwei Datenpunkte nötig
                 # Konsistenz der Änderungen prüfen
                 mean_change = np.mean(changes)
                 abs_changes = [abs(c) for c in changes]
                 mean_abs_change = np.mean(abs_changes) if abs_changes else 0

                 # Konsistenz = |Mittelwert der Änderungen| / Mittelwert der |Änderungen|
                 consistency = abs(mean_change) / mean_abs_change if mean_abs_change > 0.001 else 1.0

                 # Nur bei hoher Konsistenz konsolidieren
                 if consistency > REFLECTION_CONSISTENCY_THRESHOLD:
                     # Evidenzstärke basiert auf Konsistenz und mittlerer Änderungsgröße
                     evidence_strength = consistency * abs(mean_change) * reflection_strength

                     # Richtung der "Reflexionsevidenz"
                     evidence_direction = np.sign(mean_change)

                     # Annahme: Reflexion liefert relativ sichere Evidenz über den Trend
                     evidence_certainty = 0.7 * reflection_strength

                     # Kontext für Update
                     update_context = {
                         "reflection_consistency": consistency,
                         "mean_change": mean_change,
                         "num_memories": len(changes),
                         "update_valence": False # Reflexion ändert Valenz nicht direkt
                     }

                     # Zentrales Update durchführen
                     strength_change, _, _ = self._update_belief_internal(
                         belief_name=belief_name,
                         evidence_strength=evidence_strength,
                         evidence_direction=int(evidence_direction),
                         evidence_certainty=evidence_certainty,
                         source="reflection",
                         context=update_context
                     )
                     if abs(strength_change) > 0.001:
                          belief_changes[belief_name] = belief_changes.get(belief_name, 0.0) + strength_change


        # Optional: Episodisches Gedächtnis nach Reflexion bereinigen oder markieren?

        return belief_changes


    def __str__(self):
        """String-Repräsentation des Agenten."""
        return (f"Agent(ID: {self.agent_id}, Beliefs: {len(self.beliefs)}, "
                f"Arch: {self.cognitive_architecture}, "
                f"Connections: {len(self.social_connections)}, Groups: {len(self.group_identities)})")


# Block 4: EthicalScenario und NeuralEthicalSociety Setup
# Purpose: Definition von Szenarien und der Gesellschaftsstruktur.
# Änderungen: Konstanten verwendet, kleine Verfeinerungen in Generierungslogik.

import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.distance import cosine
from scipy.stats import entropy
import random
from typing import List, Dict, Tuple, Optional, Set, Union, Any
import pickle
import os
import logging

# Importiere Konstanten und Klassen aus vorherigen Blöcken
# Annahme: NeuralEthicalBelief, CognitiveArchitecture, NeuralEthicalAgent sind verfügbar
# from your_module import NeuralEthicalBelief, CognitiveArchitecture, NeuralEthicalAgent

class EthicalScenario:
    """Repräsentiert ein ethisches Szenario oder Dilemma."""
    def __init__(self, scenario_id: str, description: str,
                 relevant_beliefs: Dict[str, float], # belief_name -> relevance (0-1)
                 options: Dict[str, Dict[str, float]], # option_name -> {belief_name: impact (-1 to 1)}
                 option_attributes: Optional[Dict[str, Dict[str, Any]]] = None, # option_name -> {attr_name: value}
                 outcome_feedback: Optional[Dict[str, Dict[str, float]]] = None, # option_name -> {belief_name: feedback (-1 to 1)}
                 moral_implications: Optional[Dict[str, Dict[str, float]]] = None, # option_name -> {foundation: impact (-1 to 1)}
                 emotional_valence: float = 0.0, # Grundstimmung des Szenarios (-1 bis +1)
                 narrative_elements: Optional[Dict[str, Any]] = None): # Für narrative Verarbeitung
        """
        Initialisiert ein ethisches Szenario.

        Args:
            scenario_id: Eindeutige ID des Szenarios.
            description: Textuelle Beschreibung.
            relevant_beliefs: Welche Überzeugungen sind für dieses Szenario besonders relevant und mit welcher Stärke.
            options: Die Handlungsmöglichkeiten und ihre direkten Auswirkungen auf Überzeugungsstärken.
            option_attributes: Zusätzliche Attribute der Optionen (z.B. {'risks': 0.8, 'group_norms': {'GroupA': 0.5}}).
            outcome_feedback: Wie sich die Wahl einer Option langfristig auf Überzeugungen auswirkt (für Lernprozesse).
            moral_implications: Auswirkungen der Optionen auf die moralischen Grundlagen.
            emotional_valence: Die allgemeine emotionale Tönung des Szenarios.
            narrative_elements: Elemente für narrative Verarbeitung (z.B. {'characters': [], 'conflict': '', 'coherence': 0.7}).
        """
        self.scenario_id = scenario_id
        self.description = description
        self.relevant_beliefs = relevant_beliefs
        self.options = options
        self.option_attributes = option_attributes if option_attributes is not None else {}
        self.outcome_feedback = outcome_feedback if outcome_feedback is not None else {}
        self.moral_implications = moral_implications if moral_implications is not None else {}
        self.emotional_valence = clip(emotional_valence, -1.0, 1.0)
        self.narrative_elements = narrative_elements if narrative_elements is not None else self._default_narrative()

        # Validierung der Narrative Elements
        if "coherence" not in self.narrative_elements:
             self.narrative_elements["coherence"] = 0.5 # Default coherence

    def _default_narrative(self) -> Dict[str, Any]:
        """Erzeugt Standard-Narrativelemente."""
        return {
            "characters": [],
            "conflict": "Unspecified ethical conflict.",
            "context": "General societal context.",
            "coherence": 0.5 # Neutrale Kohärenz
        }

    def __str__(self):
        return f"Scenario(ID: {self.scenario_id}, Options: {len(self.options)}, Beliefs: {len(self.relevant_beliefs)})"


class NeuralEthicalSociety:
    """
    Repräsentiert eine Gesellschaft von ethischen Agenten mit neuronalen Verarbeitungsmodellen.
    Verwaltet Agenten, Szenarien, das soziale Netzwerk und führt Simulationen durch.
    """
    def __init__(self):
        """Initialisiert eine leere Gesellschaft."""
        self.agents: Dict[str, NeuralEthicalAgent] = {}
        self.scenarios: Dict[str, EthicalScenario] = {}
        self.belief_templates: Dict[str, Dict[str, Any]] = {} # name -> {category, connections, concepts, valence}
        self.social_network = nx.Graph()
        self.groups: Dict[str, Set[str]] = {} # group_name -> set of agent_ids

        # --- Simulation Controls ---
        self.current_step: int = 0
        # Robustheitseinstellungen könnten aus einer Konfigurationsdatei geladen werden
        self.robustness_settings = {
            "validation_enabled": True,
            "error_checking": True,
            "ensemble_size": SIMULATION_DEFAULT_ENSEMBLE_SIZE,
            "boundary_checks": True, # Werden jetzt großteils durch clip() sichergestellt
            "sensitivity_analysis": False, # Muss explizit aktiviert werden
            "resilience_to_outliers": True # Fehlerhafte Agenten/Schritte überspringen
        }

        # Validierungsmetriken (werden während der Simulation gefüllt)
        self.validation_log: List[Dict[str, Any]] = []

    # --- Setup Methoden ---
    def add_agent(self, agent: NeuralEthicalAgent):
        """Fügt einen Agenten zur Gesellschaft hinzu und initialisiert seinen Zeitstempel."""
        if agent.agent_id in self.agents:
            logging.warning(f"Agent {agent.agent_id} wird überschrieben.")
        agent.current_time = self.current_step # Synchronisiere Agentenzeit
        self.agents[agent.agent_id] = agent
        self.social_network.add_node(agent.agent_id, agent_instance=agent) # Speichere Referenz im Knoten

    def add_scenario(self, scenario: EthicalScenario):
        """Fügt ein Szenario zur Gesellschaft hinzu."""
        self.scenarios[scenario.scenario_id] = scenario

    def add_belief_template(self, name: str, category: str,
                            connections: Optional[Dict[str, Tuple[float, int]]] = None,
                            associated_concepts: Optional[Dict[str, float]] = None,
                            emotional_valence: float = 0.0):
        """Fügt eine Template-Überzeugung hinzu, die zur Generierung von Agenten genutzt wird."""
        self.belief_templates[name] = {
            "category": category,
            "connections": connections or {},
            "associated_concepts": associated_concepts or {},
            "emotional_valence": clip(emotional_valence, -1.0, 1.0)
        }

    def add_social_connection(self, agent1_id: str, agent2_id: str, strength: float):
        """Fügt eine soziale Verbindung (Kante im Netzwerk) hinzu."""
        if agent1_id in self.agents and agent2_id in self.agents:
            strength = clip(strength)
            self.agents[agent1_id].add_social_connection(agent2_id, strength)
            self.agents[agent2_id].add_social_connection(agent1_id, strength)
            self.social_network.add_edge(agent1_id, agent2_id, weight=strength)
        else:
            logging.warning(f"Konnte Verbindung nicht hinzufügen: Agent(en) {agent1_id} oder {agent2_id} nicht gefunden.")

    def add_group(self, group_name: str, agent_ids: List[str],
                  min_identification: float = 0.5, max_identification: float = 1.0):
        """Definiert eine Gruppe und weist Agenten Identifikationsstärken zu."""
        self.groups[group_name] = set()
        for agent_id in agent_ids:
            if agent_id in self.agents:
                identification = clip(np.random.uniform(min_identification, max_identification))
                self.agents[agent_id].add_group_identity(group_name, identification)
                self.groups[group_name].add(agent_id)
            else:
                logging.warning(f"Agent {agent_id} für Gruppe {group_name} nicht gefunden.")

    # --- Agenten Generierung ---
    def generate_random_agent(self, agent_id: str) -> NeuralEthicalAgent:
        """
        Generiert einen zufälligen Agenten basierend auf Templates und Konstanten.
        """
        agent = NeuralEthicalAgent(agent_id) # Verwendet Defaults für Persönlichkeit etc.

        # Zufällige Anzahl an Überzeugungen auswählen
        num_beliefs = np.random.randint(AGENT_GENERATION_MIN_BELIEFS, AGENT_GENERATION_MAX_BELIEFS + 1)
        available_templates = list(self.belief_templates.keys())
        if not available_templates:
             logging.warning("Keine Belief-Templates definiert, Agent bleibt ohne Beliefs.")
             return agent
        if num_beliefs > len(available_templates):
             num_beliefs = len(available_templates)

        selected_template_names = np.random.choice(available_templates, size=num_beliefs, replace=False)

        # Überzeugungen aus Templates erstellen und hinzufügen
        for belief_name in selected_template_names:
            template = self.belief_templates[belief_name]

            # Anfangsstärke basierend auf globaler Einstellung
            if AGENT_GENERATION_BELIEF_DIST == "normal":
                init_strength = clip(np.random.normal(0.5, 0.15))
            elif AGENT_GENERATION_BELIEF_DIST == "uniform":
                init_strength = clip(np.random.uniform(0.1, 0.9))
            else: # Default: beta
                init_strength = clip(np_random_beta(2, 2)) # Beta(2,2) ist breit um 0.5

            init_certainty = clip(np_random_beta(4, 4)) # Mittlere Gewissheit
            init_valence = clip(template["emotional_valence"] + np.random.normal(0, 0.2), -1.0, 1.0) # Valenz mit Rauschen

            belief = NeuralEthicalBelief(belief_name, template["category"], init_strength, init_certainty, init_valence)

            # Verbindungen (nur zu anderen *ausgewählten* Beliefs dieses Agenten)
            for conn_name, (conn_strength, polarity) in template["connections"].items():
                if conn_name in selected_template_names:
                    actual_strength = clip(conn_strength * np.random.uniform(0.8, 1.2))
                    belief.add_connection(conn_name, actual_strength, polarity)

            # Assoziierte Konzepte
            for concept_name, assoc_strength in template["associated_concepts"].items():
                actual_strength = clip(assoc_strength * np.random.uniform(0.8, 1.2))
                belief.add_associated_concept(concept_name, actual_strength)

            agent.add_belief(belief)

        # Zusätzliche zufällige Verbindungen zwischen den erstellten Beliefs
        agent_belief_names = list(agent.beliefs.keys())
        for belief_name in agent_belief_names:
            belief = agent.beliefs[belief_name]
            for other_name in agent_belief_names:
                 # Wenn nicht selbst und noch keine Verbindung existiert und Wahrscheinlichkeit eintritt
                 if other_name != belief_name and \
                    other_name not in belief.connections and \
                    np.random.random() < AGENT_GENERATION_CONN_PROB:

                     rand_strength = np.random.uniform(*AGENT_GENERATION_CONN_STRENGTH_RANGE)
                     rand_polarity = np.random.choice([-1, 1])
                     belief.add_connection(other_name, rand_strength, rand_polarity)
                     # Symmetrische Verbindung hinzufügen (optional, macht Netzwerk ungerichtet bzgl. Struktur)
                     # agent.beliefs[other_name].add_connection(belief_name, rand_strength, rand_polarity)

        return agent


    def generate_similar_agent(self, base_agent: NeuralEthicalAgent, agent_id: str,
                              similarity: float = 0.8) -> NeuralEthicalAgent:
        """
        Generiert einen Agenten, der `base_agent` ähnlich ist.
        Je höher `similarity`, desto geringer die Abweichungen.
        """
        similarity = clip(similarity)
        dissimilarity_factor = (1.0 - similarity)

        # 1. Persönlichkeit variieren
        new_personality = {}
        for trait, value in base_agent.personality_traits.items():
             variation = np.random.normal(0, dissimilarity_factor * AGENT_GENERATION_SIMILARITY_VARIATION_FACTOR)
             new_personality[trait] = clip(value + variation)

        # Erstelle neuen Agenten mit ähnlicher Persönlichkeit (Architektur wird neu generiert!)
        new_agent = NeuralEthicalAgent(agent_id, personality_traits=new_personality)
        # Kognitive Architektur wird in __init__ basierend auf neuer Persönlichkeit generiert.
        # Optional: Man könnte auch die Architektur direkt kopieren und leicht variieren.

        # 2. Moralische Grundlagen variieren
        for foundation, value in base_agent.moral_foundations.items():
             variation = np.random.normal(0, dissimilarity_factor * AGENT_GENERATION_SIMILARITY_VARIATION_FACTOR)
             new_agent.moral_foundations[foundation] = clip(value + variation)

        # 3. Überzeugungen kopieren und variieren
        for belief_name, base_belief in base_agent.beliefs.items():
            # Wahrscheinlichkeit, eine Überzeugung zu übernehmen, sinkt mit Unähnlichkeit
            if np.random.random() < similarity + 0.1: # Etwas höhere Basiswahrscheinlichkeit
                 # Stärke, Gewissheit, Valenz variieren
                 strength_var = np.random.normal(0, dissimilarity_factor * AGENT_GENERATION_BELIEF_STRENGTH_VARIATION_FACTOR)
                 new_strength = clip(base_belief.strength + strength_var)

                 certainty_var = np.random.normal(0, dissimilarity_factor * AGENT_GENERATION_BELIEF_CERTAINTY_VARIATION_FACTOR)
                 new_certainty = clip(base_belief.certainty + certainty_var)

                 valence_var = np.random.normal(0, dissimilarity_factor * AGENT_GENERATION_BELIEF_VALENCE_VARIATION_FACTOR)
                 new_valence = clip(base_belief.emotional_valence + valence_var, -1.0, 1.0)

                 new_belief = NeuralEthicalBelief(belief_name, base_belief.category, new_strength, new_certainty, new_valence)

                 # Verbindungen kopieren/variieren (nur zu Beliefs, die der neue Agent auch hat!)
                 for conn_name, (conn_strength, polarity) in base_belief.connections.items():
                      if conn_name in base_agent.beliefs: # Nur existierende Verbindungen betrachten
                           # Evtl. Polarität flippen bei hoher Unähnlichkeit
                           if np.random.random() < dissimilarity_factor * AGENT_GENERATION_SIMILARITY_POLARITY_FLIP_PROB:
                                polarity *= -1
                           # Stärke variieren
                           conn_var = np.random.normal(0, dissimilarity_factor * AGENT_GENERATION_SIMILARITY_VARIATION_FACTOR)
                           new_conn_strength = clip(conn_strength + conn_var)
                           new_belief.add_connection(conn_name, new_conn_strength, polarity)

                 # Assoziierte Konzepte kopieren/variieren
                 for concept, assoc_strength in base_belief.associated_concepts.items():
                      assoc_var = np.random.normal(0, dissimilarity_factor * AGENT_GENERATION_SIMILARITY_VARIATION_FACTOR)
                      new_assoc_strength = clip(assoc_strength + assoc_var)
                      new_belief.add_associated_concept(concept, new_assoc_strength)

                 new_agent.add_belief(new_belief)

        # 4. Gruppenidentitäten kopieren/variieren
        for group, identification in base_agent.group_identities.items():
             id_var = np.random.normal(0, dissimilarity_factor * AGENT_GENERATION_GROUP_ID_VARIATION_FACTOR)
             new_identification = clip(identification + id_var)
             # Nur hinzufügen, wenn Identifikation > 0 (verhindert negative Werte)
             if new_identification > 0.01:
                  new_agent.add_group_identity(group, new_identification)

        return new_agent


    def generate_diverse_society(self, num_agents: int, num_archetypes: int = 4,
                                 similarity_range: Tuple[float, float] = (0.6, 0.9)):
        """
        Generiert eine diverse Gesellschaft mit `num_agents` Agenten.
        Erstellt zuerst `num_archetypes` und generiert dann ähnliche Agenten darum herum.
        """
        if num_agents < num_archetypes:
            num_archetypes = num_agents
            logging.warning(f"Weniger Agenten als Archetypen angefordert. Reduziere Archetypen auf {num_agents}.")

        # 1. Archetypen erstellen
        archetypes = []
        for i in range(num_archetypes):
            archetype_id = f"archetype_{i+1}"
            archetype = self.generate_random_agent(archetype_id)
            archetypes.append(archetype)
            self.add_agent(archetype)
            # Definiere Archetyp-Gruppe
            self.add_group(f"ArchetypeGroup_{i+1}", [archetype_id], min_identification=0.9)

        # 2. Weitere Agenten generieren, basierend auf zufälligen Archetypen
        agents_generated = num_archetypes
        while agents_generated < num_agents:
             base_archetype = random.choice(archetypes)
             agent_id = f"agent_{agents_generated + 1}"
             similarity = np.random.uniform(*similarity_range)
             new_agent = self.generate_similar_agent(base_archetype, agent_id, similarity)
             self.add_agent(new_agent)
             # Neuen Agenten zur Gruppe seines Archetyps hinzufügen
             archetype_index = archetypes.index(base_archetype)
             group_name = f"ArchetypeGroup_{archetype_index + 1}"
             # Identifikation skaliert mit Ähnlichkeit zum Archetyp
             identification = clip(similarity * np.random.uniform(0.7, 1.1))
             new_agent.add_group_identity(group_name, identification)
             # Update Gruppe in Society (falls nicht schon durch add_group passiert)
             if group_name in self.groups:
                 self.groups[group_name].add(agent_id)

             agents_generated += 1

        # 3. Realistisches soziales Netzwerk generieren
        logging.info("Generiere soziales Netzwerk...")
        self.generate_realistic_social_network()
        logging.info(f"Netzwerk generiert mit {self.social_network.number_of_edges()} Verbindungen.")


    def generate_realistic_social_network(self):
        """
        Generiert ein soziales Netzwerk basierend auf Ähnlichkeit (Beliefs, Kognition, Gruppe)
        und Persönlichkeit (Extroversion). Verwendet Konstanten für Gewichtungen.
        """
        agent_ids = list(self.agents.keys())
        num_agents = len(agent_ids)
        if num_agents < 2: return # Kein Netzwerk bei < 2 Agenten

        # --- Ähnlichkeitsberechnungen (nur einmal pro Paar) ---
        similarity_cache: Dict[Tuple[str, str], Dict[str, float]] = {}
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                agent1_id = agent_ids[i]
                agent2_id = agent_ids[j]
                agent1 = self.agents[agent1_id]
                agent2 = self.agents[agent2_id]
                pair_key = tuple(sorted((agent1_id, agent2_id)))

                belief_sim = self._calculate_belief_similarity(agent1, agent2)
                group_sim = self._calculate_group_similarity(agent1, agent2)
                cognitive_sim = self._calculate_cognitive_style_similarity(agent1, agent2)

                # Gewichtete Gesamtähnlichkeit
                total_sim = (NETWORK_BELIEF_SIM_FACTOR * belief_sim +
                             NETWORK_COGNITIVE_SIM_FACTOR * cognitive_sim +
                             (1.0 - NETWORK_BELIEF_SIM_FACTOR - NETWORK_COGNITIVE_SIM_FACTOR) * group_sim)

                similarity_cache[pair_key] = {
                    "belief": belief_sim,
                    "group": group_sim,
                    "cognitive": cognitive_sim,
                    "total": clip(total_sim) # Sicherstellen, dass Sim >= 0
                }

        # --- Verbindungen erstellen ---
        for i in range(num_agents):
             for j in range(i + 1, num_agents):
                 agent1_id = agent_ids[i]
                 agent2_id = agent_ids[j]
                 agent1 = self.agents[agent1_id]
                 agent2 = self.agents[agent2_id]
                 pair_key = tuple(sorted((agent1_id, agent2_id)))
                 similarities = similarity_cache[pair_key]

                 # Basis-Verbindungswahrscheinlichkeit
                 prob = NETWORK_DEFAULT_DENSITY

                 # Einfluss der Gesamtähnlichkeit
                 prob += similarities["total"] * NETWORK_SIMILARITY_CONN_PROB_MOD

                 # Einfluss gemeinsamer Gruppen (direkter Boost)
                 common_groups = set(agent1.group_identities.keys()) & set(agent2.group_identities.keys())
                 if common_groups:
                      max_shared_id = 0.0
                      for group in common_groups:
                           max_shared_id = max(max_shared_id, min(agent1.group_identities[group], agent2.group_identities[group]))
                      prob += NETWORK_GROUP_CONN_BOOST * max_shared_id

                 # Einfluss Persönlichkeit (Extroversion)
                 extroversion_factor = 0.5 * (agent1.personality_traits["extroversion"] + agent2.personality_traits["extroversion"])
                 prob += extroversion_factor * NETWORK_EXTROVERSION_CONN_PROB_MOD

                 # Einfluss Kognitiver Stil (sozialere Stile erhöhen Wahrscheinlichkeit leicht)
                 style_boost = 0.0
                 for agent in [agent1, agent2]:
                      style = agent.cognitive_architecture.primary_processing
                      if style == NeuralProcessingType.NARRATIVE: style_boost += NETWORK_NARRATIVE_CONN_PROB_BOOST
                      elif style == NeuralProcessingType.EMOTIONAL: style_boost += NETWORK_EMOTIONAL_CONN_PROB_BOOST
                 prob += style_boost / 2.0 # Durchschnittlicher Boost

                 # Verbindung erstellen?
                 if np.random.random() < clip(prob, 0.0, NETWORK_MAX_CONN_PROB):
                      # Stärke der Verbindung basiert auf Gesamtähnlichkeit
                      strength = scale_value(similarities["total"], 0, 1, *NETWORK_STRENGTH_FROM_SIMILARITY_RANGE)
                      self.add_social_connection(agent1_id, agent2_id, strength)


    # --- Ähnlichkeitsberechnungen (interne Helfer) ---
    def _calculate_belief_similarity(self, agent1: NeuralEthicalAgent, agent2: NeuralEthicalAgent) -> float:
        """Berechnet Ähnlichkeit basierend auf gemeinsamen Überzeugungen (Stärke, Gewissheit, Valenz)."""
        common_beliefs = set(agent1.beliefs.keys()) & set(agent2.beliefs.keys())
        if not common_beliefs: return 0.0

        total_similarity = 0.0
        for name in common_beliefs:
            b1 = agent1.beliefs[name]
            b2 = agent2.beliefs[name]
            strength_sim = 1.0 - abs(b1.strength - b2.strength)
            certainty_sim = 1.0 - abs(b1.certainty - b2.certainty)
            valence_sim = 1.0 - abs(b1.emotional_valence - b2.emotional_valence) / 2.0 # Max diff is 2
            # Gewichtung: Stärke 60%, Gewissheit 20%, Valenz 20%
            combined_sim = 0.6 * strength_sim + 0.2 * certainty_sim + 0.2 * valence_sim
            total_similarity += combined_sim

        return total_similarity / len(common_beliefs) if common_beliefs else 0.0

    def _calculate_group_similarity(self, agent1: NeuralEthicalAgent, agent2: NeuralEthicalAgent) -> float:
        """Berechnet Ähnlichkeit basierend auf geteilten Gruppenidentitäten."""
        all_groups = set(agent1.group_identities.keys()) | set(agent2.group_identities.keys())
        if not all_groups: return 0.0

        total_similarity = 0.0
        for group in all_groups:
            id1 = agent1.group_identities.get(group, 0.0)
            id2 = agent2.group_identities.get(group, 0.0)
            total_similarity += 1.0 - abs(id1 - id2) # Je näher die Identifikation, desto ähnlicher

        return total_similarity / len(all_groups) if all_groups else 0.0

    def _calculate_cognitive_style_similarity(self, agent1: NeuralEthicalAgent, agent2: NeuralEthicalAgent) -> float:
        """Berechnet Ähnlichkeit basierend auf kognitiver Architektur."""
        arch1 = agent1.cognitive_architecture
        arch2 = agent2.cognitive_architecture

        primary_sim = 1.0 if arch1.primary_processing == arch2.primary_processing else 0.0
        secondary_sim = 1.0 if arch1.secondary_processing == arch2.secondary_processing else 0.0
        balance_sim = 1.0 - abs(arch1.processing_balance - arch2.processing_balance)

        bias_sim = self._compare_parameter_dicts(arch1.cognitive_biases, arch2.cognitive_biases)
        emotion_sim = self._compare_parameter_dicts(arch1.emotional_parameters, arch2.emotional_parameters)
        bayesian_sim = self._compare_parameter_dicts(arch1.bayesian_parameters, arch2.bayesian_parameters)

        # Gewichtung: Primärstil 30%, Sekundärstil 10%, Balance 15%, Biases 15%, Emotionen 15%, Bayes 15%
        total_sim = (0.30 * primary_sim + 0.10 * secondary_sim + 0.15 * balance_sim +
                     0.15 * bias_sim + 0.15 * emotion_sim + 0.15 * bayesian_sim)
        return clip(total_sim)

    def _compare_parameter_dicts(self, dict1: Dict[str, float], dict2: Dict[str, float]) -> float:
        """Vergleicht zwei Dictionaries mit Parametern (0-1) und gibt Ähnlichkeit zurück."""
        common_keys = set(dict1.keys()) & set(dict2.keys())
        if not common_keys: return 0.0

        total_diff = sum(abs(dict1[key] - dict2[key]) for key in common_keys)
        avg_diff = total_diff / len(common_keys)
        return 1.0 - avg_diff # Ähnlichkeit = 1 - durchschnittliche Differenz

    # --- Simulation Execution (wird in nächstem Block implementiert) ---
    def run_robust_simulation(self, num_steps: int, scenario_probability: Optional[float] = None,
                               social_influence_probability: Optional[float] = None,
                               reflection_probability: Optional[float] = None) -> Dict[str, Any]:
         # Platzhalter - Implementierung folgt
         raise NotImplementedError("Simulation logic will be implemented in the next block.")

    # --- Persistenz ---
    def save_simulation(self, filename: str):
        """Speichert den Zustand der Gesellschaft (inkl. Agenten, Netzwerk etc.)"""
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            logging.info(f"Simulation erfolgreich gespeichert in {filename}")
        except Exception as e:
            logging.error(f"Fehler beim Speichern der Simulation nach {filename}: {e}")

    @classmethod
    def load_simulation(cls, filename: str) -> Optional['NeuralEthicalSociety']:
        """Lädt eine gespeicherte Gesellschaft."""
        try:
            with open(filename, 'rb') as f:
                society = pickle.load(f)
            if isinstance(society, cls):
                 logging.info(f"Simulation erfolgreich geladen aus {filename}")
                 # Optional: Netzwerkknoten-Referenzen wiederherstellen, falls nötig
                 for node_id, data in society.social_network.nodes(data=True):
                      if 'agent_instance' not in data and node_id in society.agents:
                           society.social_network.nodes[node_id]['agent_instance'] = society.agents[node_id]
                 return society
            else:
                 logging.error(f"Geladene Datei {filename} ist keine NeuralEthicalSociety Instanz.")
                 return None
        except FileNotFoundError:
            logging.error(f"Datei nicht gefunden: {filename}")
            return None
        except Exception as e:
            logging.error(f"Fehler beim Laden der Simulation aus {filename}: {e}")
            return None

    def __str__(self):
        return (f"NeuralEthicalSociety(Agents: {len(self.agents)}, Scenarios: {len(self.scenarios)}, "
                f"Groups: {len(self.groups)}, Edges: {self.social_network.number_of_edges()})")

    def _calculate_polarization(self, belief_strengths: List[float], metric: str = 'bimodality') -> float:
        """
        Berechnet eine Polarisierungsmetrik für die gegebenen Belief-Stärken.
        
        Args:
            belief_strengths: Liste der Belief-Stärken aller Agenten
            metric: Welche Metrik verwendet werden soll ('bimodality', 'variance', etc.)
            
        Returns:
            float: Polarisierungswert (0-1, höher = mehr polarisiert)
        """
        if not belief_strengths or len(belief_strengths) < 2:
            return 0.0
            
        if metric == 'bimodality':
            # Bimodalitätsindex basierend auf Pearson's Kurtosis und Varianz
            # Werte nahe 1 deuten auf Bimodalität hin
            mean = np.mean(belief_strengths)
            variance = np.var(belief_strengths)
            if variance == 0:
                return 0.0  # Keine Polarisierung wenn alle Werte gleich sind
                
            # Kurtosis (4. Moment)
            n = len(belief_strengths)
            kurtosis = (n * sum((x - mean) ** 4 for x in belief_strengths)) / ((sum((x - mean) ** 2 for x in belief_strengths)) ** 2)
            
            # Bimodalitätskoeffizient
            bimodality = (1 + kurtosis) / (3 + variance)
            # Normalisieren von 0 bis 1
            return min(1.0, max(0.0, (bimodality - 0.2) / 0.5))
            
        elif metric == 'variance':
            # Varianz als einfache Polarisierungsmetrik
            variance = np.var(belief_strengths)
            # Normalisieren (typische Varianz für Beliefs zwischen 0-1 liegt zwischen 0 und 0.25)
            return min(1.0, variance * 4)
            
        elif metric == 'extremity':
            # Wie viele Agenten haben extreme Werte (nahe 0 oder 1)?
            extremity_threshold = 0.2
            extremes = sum(1 for x in belief_strengths if x < extremity_threshold or x > (1 - extremity_threshold))
            return extremes / len(belief_strengths)
            
        else:
            logging.warning(f"Unbekannte Polarisierungsmetrik: {metric}")
            return 0.0


# Block 5: Simulation Execution in NeuralEthicalSociety
# Purpose: Implementierung der Simulationslogik mit Robustheit und Ensemble-Methoden.
# Änderungen: run_robust_simulation implementiert, verwendet Konstanten,
#              vervollständigt Ensemble-Kombination, integriert Validierung.

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import entropy
from tqdm import tqdm # Fortschrittsanzeige
import seaborn as sns
import pandas as pd
import networkx as nx
import numpy as np
from typing import List, Dict, Any, Optional

# Importiere ggf. Society und Agenten-Klassen, falls für Typ-Hinweise benötigt
# from your_module import NeuralEthicalSociety, NeuralEthicalAgent, NeuralProcessingType
# Importiere Konstanten


class SimulationAnalyzer:
    """Klasse zur Analyse der Ergebnisse einer NeuralEthicalSociety Simulation."""

    def __init__(self, society: 'NeuralEthicalSociety', results: Dict[str, Any]):
        """
        Initialisiert den Analyzer mit der Gesellschaft und den Simulationsergebnissen.

        Args:
            society: Die NeuralEthicalSociety Instanz (vor der Simulation).
            results: Das Ergebnis-Dictionary von run_robust_simulation.
        """
        self.society = society
        self.results = results
        self.num_steps = len(results.get("decisions", []))
        self.agent_ids = list(society.agents.keys())

        # Berechne einige Metriken direkt bei der Initialisierung
        self.final_agent_states = results.get("agent_states", [])[-1] if results.get("agent_states") else {}
        self.final_polarization = self._calculate_final_polarization()
        self.belief_evolution_df = self._create_belief_evolution_dataframe()

    def _create_belief_evolution_dataframe(self) -> Optional[pd.DataFrame]:
        """Erstellt ein DataFrame zur Verfolgung der Belief-Entwicklung."""
        records = []
        agent_states_history = self.results.get("agent_states", [])
        if not agent_states_history: return None

        for step, step_states in enumerate(agent_states_history):
            for agent_id, state in step_states.items():
                 if agent_id in self.society.agents: # Nur Agenten, die wir kennen
                     agent_arch = self.society.agents[agent_id].cognitive_architecture
                     for belief_name, belief_data in state.get("beliefs", {}).items():
                          records.append({
                              "step": step,
                              "agent_id": agent_id,
                              "belief": belief_name,
                              "strength": belief_data.get("strength"),
                              "certainty": belief_data.get("certainty"),
                              "valence": belief_data.get("valence"),
                              "activation": belief_data.get("activation"),
                              "primary_style": agent_arch.primary_processing,
                              "balance": agent_arch.processing_balance
                          })
        if not records: return None
        return pd.DataFrame(records)

    def get_belief_evolution(self, belief_name: str) -> Optional[pd.DataFrame]:
        """Gibt die Zeitreihenentwicklung für einen bestimmten Belief zurück."""
        if self.belief_evolution_df is None: return None
        return self.belief_evolution_df[self.belief_evolution_df['belief'] == belief_name]

    def get_decision_summary(self, scenario_id: Optional[str] = None) -> Dict[str, Dict[str, int]]:
         """Zählt die Entscheidungen pro Option für Szenarien."""
         decision_counts: Dict[str, Dict[str, int]] = {} # scenario_id -> option -> count
         for step_decisions in self.results.get("decisions", []):
             for agent_id, decision in step_decisions.items():
                 scen_id = decision.get("scenario_id")
                 option = decision.get("chosen_option")
                 if scen_id and option:
                      # Filter nach Szenario, falls angegeben
                      if scenario_id is not None and scen_id != scenario_id:
                           continue

                      if scen_id not in decision_counts: decision_counts[scen_id] = {}
                      decision_counts[scen_id][option] = decision_counts[scen_id].get(option, 0) + 1
         return decision_counts

    def get_cognitive_style_decision_summary(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        """Zählt Entscheidungen gruppiert nach primärem kognitiven Stil."""
        style_counts: Dict[str, Dict[str, Dict[str, int]]] = {style: {} for style in NeuralProcessingType.ALL_TYPES}

        for step_decisions in self.results.get("decisions", []):
             for agent_id, decision in step_decisions.items():
                 scen_id = decision.get("scenario_id")
                 option = decision.get("chosen_option")
                 if scen_id and option and agent_id in self.society.agents:
                      agent = self.society.agents[agent_id]
                      style = agent.cognitive_architecture.primary_processing
                      if style not in style_counts: style_counts[style] = {} # Sicherstellen
                      if scen_id not in style_counts[style]: style_counts[style][scen_id] = {}
                      style_counts[style][scen_id][option] = style_counts[style][scen_id].get(option, 0) + 1
        # Entferne leere Stile
        return {s: data for s, data in style_counts.items() if data}


    def _calculate_final_polarization(self) -> Optional[Dict[str, Dict[str, float]]]:
        """Berechnet Polarisierungsmetriken für den finalen Zustand."""
        if not self.final_agent_states: return None
        # Erstelle temporäre Agentenobjekte für die Berechnung
        temp_agents = {}
        initial_agents = self.society.agents # Zustand vor der Sim
        for agent_id, state in self.final_agent_states.items():
            if agent_id in initial_agents:
                temp_agent = copy.deepcopy(initial_agents[agent_id]) # Verwende Struktur des Originals
                # Update Beliefs
                for b_name, b_data in state.get("beliefs", {}).items():
                    if b_name in temp_agent.beliefs:
                        temp_agent.beliefs[b_name].strength = b_data.get("strength", 0.5)
                        temp_agent.beliefs[b_name].certainty = b_data.get("certainty", 0.5)
                        temp_agent.beliefs[b_name].emotional_valence = b_data.get("valence", 0.0)
                temp_agents[agent_id] = temp_agent
        # Rufe die Polarisierungsfunktion der Society auf
        return self.society._calculate_polarization(temp_agents)

    def get_polarization_trend(self, metric: str = 'bimodality') -> Optional[pd.DataFrame]:
        """Gibt die Entwicklung der Polarisierung über die Zeit zurück."""
        polarization_history = []
        agent_states_history = self.results.get("agent_states", [])
        if not agent_states_history: return None

        # Verwende die _calculate_polarization Methode für jeden Schritt
        temp_agents_step = copy.deepcopy(self.society.agents) # Startzustand

        for step, step_states in enumerate(agent_states_history):
            # Update temp_agents_step basierend auf step_states
            current_step_agents = {}
            for agent_id, state in step_states.items():
                 if agent_id in temp_agents_step:
                      agent = copy.deepcopy(temp_agents_step[agent_id]) # Kopieren für diesen Schritt
                      for b_name, b_data in state.get("beliefs", {}).items():
                           if b_name in agent.beliefs:
                                agent.beliefs[b_name].strength = b_data.get("strength", 0.5)
                                agent.beliefs[b_name].certainty = b_data.get("certainty", 0.5)
                      current_step_agents[agent_id] = agent

            step_polarization = self.society._calculate_polarization(current_step_agents)
            if step_polarization:
                for belief, metrics in step_polarization.items():
                     polarization_history.append({
                         "step": step,
                         "belief": belief,
                         "metric": metric,
                         "value": metrics.get(metric)
                     })

        if not polarization_history: return None
        df = pd.DataFrame(polarization_history)
        return df[df['value'].notna()] # Entferne Zeilen ohne Wert

    def get_ensemble_variance(self, result_type: str = "belief_strength_variance") -> Optional[pd.DataFrame]:
        """Extrahiert die Ensemble-Varianz als DataFrame."""
        variance_data = self.results.get("ensemble_statistics", {}).get(result_type, [])
        if not variance_data: return None

        records = []
        for step, step_variance in enumerate(variance_data):
             for agent_id, belief_variances in step_variance.items():
                 for belief, variance in belief_variances.items():
                      records.append({
                          "step": step,
                          "agent_id": agent_id,
                          "belief": belief,
                          "variance": variance
                      })
        if not records: return None
        return pd.DataFrame(records)

    def identify_belief_clusters(self, step: int = -1) -> List[Dict[str, Any]]:
         """Identifiziert Agenten-Cluster basierend auf Belief-Ähnlichkeit zu einem bestimmten Zeitpunkt."""
         if step == -1: # Letzter Schritt
              agents_to_cluster = self.final_agent_states
         elif step < len(self.results.get("agent_states", [])):
              agents_to_cluster = self.results["agent_states"][step]
         else:
              logging.error(f"Ungültiger Schritt {step} für Cluster-Analyse.")
              return []

         # Erstelle temporäre Agentenobjekte für die Ähnlichkeitsberechnung
         temp_agents = {}
         initial_agents = self.society.agents
         for agent_id, state in agents_to_cluster.items():
             if agent_id in initial_agents:
                 temp_agent = copy.deepcopy(initial_agents[agent_id])
                 for b_name, b_data in state.get("beliefs", {}).items():
                     if b_name in temp_agent.beliefs:
                         temp_agent.beliefs[b_name].strength = b_data.get("strength", 0.5)
                 temp_agents[agent_id] = temp_agent

         if len(temp_agents) < 2: return []

         agent_ids = list(temp_agents.keys())
         similarity_matrix = np.zeros((len(agent_ids), len(agent_ids)))
         for i, id1 in enumerate(agent_ids):
             for j, id2 in enumerate(agent_ids):
                  if i == j: similarity_matrix[i, j] = 1.0
                  elif i < j:
                       sim = self.society._calculate_belief_similarity(temp_agents[id1], temp_agents[id2])
                       similarity_matrix[i, j] = sim
                       similarity_matrix[j, i] = sim

         # Einfacher Schwellenwert-basierter Cluster-Algorithmus
         clusters = []
         processed_agents = set()
         threshold = ANALYSIS_BELIEF_CLUSTER_THRESHOLD

         for i, agent_id in enumerate(agent_ids):
             if agent_id in processed_agents: continue

             # Finde alle ähnlichen Agenten
             similar_indices = np.where(similarity_matrix[i, :] > threshold)[0]
             current_cluster_agents = [agent_ids[idx] for idx in similar_indices]

             if len(current_cluster_agents) >= ANALYSIS_CLUSTER_MIN_SIZE:
                 # Markiere Agenten als verarbeitet
                 processed_agents.update(current_cluster_agents)
                 # Berechne durchschnittliche Ähnlichkeit innerhalb des Clusters
                 cluster_similarities = []
                 for k_idx in similar_indices:
                      for l_idx in similar_indices:
                           if k_idx < l_idx: cluster_similarities.append(similarity_matrix[k_idx, l_idx])
                 avg_similarity = np.mean(cluster_similarities) if cluster_similarities else 1.0

                 # Finde definierende Überzeugungen (z.B. hohe/niedrige mittlere Stärke im Cluster)
                 defining_beliefs = self._find_defining_beliefs(current_cluster_agents, temp_agents)

                 clusters.append({
                     "agents": current_cluster_agents,
                     "size": len(current_cluster_agents),
                     "average_similarity": avg_similarity,
                     "defining_beliefs": defining_beliefs,
                     "step": step
                 })

         clusters.sort(key=lambda x: x["size"], reverse=True)
         return clusters


    def _find_defining_beliefs(self, cluster_agent_ids: List[str], agent_snapshots: Dict[str, NeuralEthicalAgent]) -> Dict[str, float]:
        """Identifiziert Beliefs, die für einen Cluster charakteristisch sind."""
        belief_strengths = {} # belief_name -> list of strengths in cluster
        for agent_id in cluster_agent_ids:
            if agent_id in agent_snapshots:
                 agent = agent_snapshots[agent_id]
                 for name, belief in agent.beliefs.items():
                     if name not in belief_strengths: belief_strengths[name] = []
                     belief_strengths[name].append(belief.strength)

        defining = {}
        for name, strengths in belief_strengths.items():
            if len(strengths) >= len(cluster_agent_ids) * 0.8: # Belief muss bei >80% der Cluster-Mitglieder vorkommen
                mean_strength = np.mean(strengths)
                # Definiere Beliefs als charakteristisch, wenn sie stark ausgeprägt sind (hoch oder niedrig)
                if mean_strength > ANALYSIS_STRONG_BELIEF_THRESHOLD or mean_strength < (1.0 - ANALYSIS_STRONG_BELIEF_THRESHOLD):
                     defining[name] = mean_strength
        # Sortiere nach extremer Ausprägung (Abstand von 0.5)
        return dict(sorted(defining.items(), key=lambda item: abs(item[1] - 0.5), reverse=True))


class SimulationVisualizer:
    """Klasse zur Visualisierung der Simulationsergebnisse."""

    def __init__(self, society: 'NeuralEthicalSociety', analyzer: SimulationAnalyzer):
        self.society = society
        self.analyzer = analyzer
        self.results = analyzer.results # Zugriff auf Rohdaten falls nötig

    def plot_belief_evolution(self, belief_name: str, agent_ids: Optional[List[str]] = None,
                              show_mean: bool = True, show_styles: bool = True):
        """Plottet die Entwicklung der Stärke eines Beliefs über die Zeit."""
        belief_df = self.analyzer.get_belief_evolution(belief_name)
        if belief_df is None or belief_df.empty:
            print(f"Keine Daten für Belief '{belief_name}' gefunden.")
            return

        plt.figure(figsize=(12, 7))
        title = f"Entwicklung von Belief '{belief_name}'"

        if agent_ids:
            belief_df = belief_df[belief_df['agent_id'].isin(agent_ids)]
            title += f" (Agenten: {', '.join(agent_ids)})"

        if show_styles:
            # Färbe Linien nach primärem kognitiven Stil
            styles = belief_df['primary_style'].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(styles)))
            style_colors = dict(zip(styles, colors))

            for agent_id in belief_df['agent_id'].unique():
                 agent_data = belief_df[belief_df['agent_id'] == agent_id]
                 style = agent_data['primary_style'].iloc[0]
                 plt.plot(agent_data['step'], agent_data['strength'], marker='.', linestyle='-',
                          linewidth=0.5, alpha=0.6, color=style_colors.get(style, 'gray'), label=f"{agent_id}" if not agent_ids else None) # Label nur wenn alle gezeigt werden
            # Legende für Stile
            handles = [plt.Line2D([0], [0], color=color, lw=2, label=style) for style, color in style_colors.items()]
            plt.legend(handles=handles, title="Cognitive Style", bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
             # Einfacher Plot pro Agent
             for agent_id in belief_df['agent_id'].unique():
                 agent_data = belief_df[belief_df['agent_id'] == agent_id]
                 plt.plot(agent_data['step'], agent_data['strength'], marker='.', linestyle='-', linewidth=0.5, alpha=0.7, label=agent_id if len(belief_df['agent_id'].unique()) < 10 else None)
             if len(belief_df['agent_id'].unique()) < 10 : plt.legend(title="Agent ID")


        if show_mean:
            mean_strength = belief_df.groupby('step')['strength'].mean()
            plt.plot(mean_strength.index, mean_strength.values, color='black', linewidth=2, linestyle='--', label='Mean Strength')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # Stelle sicher, dass Legende angezeigt wird

        plt.title(title)
        plt.xlabel("Simulationsschritt")
        plt.ylabel("Überzeugungsstärke")
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Platz für Legende schaffen
        plt.show()


    def plot_polarization_trend(self, belief_name: Optional[str] = None, metric: str = 'bimodality'):
        """Plottet die Entwicklung der Polarisierung (z.B. Bimodalität) über die Zeit."""
        pol_df = self.analyzer.get_polarization_trend(metric=metric)
        if pol_df is None or pol_df.empty:
            print(f"Keine Polarisierungsdaten für Metrik '{metric}' gefunden.")
            return

        plt.figure(figsize=(12, 7))
        title = f"Polarisierungstrend ({metric})"

        if belief_name:
             belief_data = pol_df[pol_df['belief'] == belief_name]
             if belief_data.empty:
                  print(f"Keine Daten für Belief '{belief_name}' und Metrik '{metric}'.")
                  return
             plt.plot(belief_data['step'], belief_data['value'], marker='o', linestyle='-', label=belief_name)
             title = f"Polarisierungstrend für '{belief_name}' ({metric})"
             plt.legend()
        else:
             # Plotte für alle Beliefs oder einen Durchschnitt? Durchschnitt ist einfacher.
             mean_pol = pol_df.groupby('step')['value'].mean()
             std_pol = pol_df.groupby('step')['value'].std()
             plt.plot(mean_pol.index, mean_pol.values, marker='o', linestyle='-', color='blue', label=f'Mean {metric}')
             plt.fill_between(mean_pol.index, mean_pol - std_pol, mean_pol + std_pol, color='blue', alpha=0.2, label=f'Std Dev {metric}')
             title = f"Durchschnittlicher Polarisierungstrend ({metric}) über alle Beliefs"
             plt.legend()


        plt.title(title)
        plt.xlabel("Simulationsschritt")
        plt.ylabel(f"{metric.capitalize()}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


    def plot_decision_distribution(self, scenario_id: str):
        """Zeigt die Verteilung der Entscheidungen für ein Szenario."""
        counts = self.analyzer.get_decision_summary(scenario_id=scenario_id)
        if not counts or scenario_id not in counts:
            print(f"Keine Entscheidungsdaten für Szenario '{scenario_id}' gefunden.")
            return
        scenario_counts = counts[scenario_id]
        options = list(scenario_counts.keys())
        values = list(scenario_counts.values())

        plt.figure(figsize=(8, 5))
        plt.bar(options, values, color=plt.cm.viridis(np.linspace(0, 1, len(options))))
        plt.title(f"Entscheidungsverteilung für Szenario '{scenario_id}'")
        plt.xlabel("Gewählte Option")
        plt.ylabel("Anzahl Entscheidungen")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def plot_cognitive_style_decision_comparison(self, scenario_id: str):
        """Vergleicht Entscheidungsmuster verschiedener kognitiver Stile für ein Szenario."""
        style_counts = self.analyzer.get_cognitive_style_decision_summary()
        if not style_counts:
            print("Keine Daten für kognitive Stil-Entscheidungen gefunden.")
            return

        # Filtere Daten für das spezifische Szenario
        scenario_style_data = {style: counts.get(scenario_id, {})
                               for style, counts in style_counts.items() if scenario_id in counts}

        if not scenario_style_data:
            print(f"Keine Daten für Szenario '{scenario_id}' nach kognitivem Stil gefunden.")
            return

        # Sammle alle Optionen, die in diesem Szenario von irgendeinem Stil gewählt wurden
        all_options = set()
        for options_dict in scenario_style_data.values():
            all_options.update(options_dict.keys())
        sorted_options = sorted(list(all_options))

        # Erstelle DataFrame für einfaches Plotten
        plot_data = []
        for style, options_dict in scenario_style_data.items():
             total_decisions_by_style = sum(options_dict.values())
             for option in sorted_options:
                  count = options_dict.get(option, 0)
                  # Berechne Anteil pro Stil
                  proportion = count / total_decisions_by_style if total_decisions_by_style > 0 else 0
                  plot_data.append({"style": style, "option": option, "proportion": proportion})

        df = pd.DataFrame(plot_data)

        # Plot als gruppiertes Balkendiagramm
        plt.figure(figsize=(12, 7))
        sns.barplot(x='option', y='proportion', hue='style', data=df, palette='tab10')
        plt.title(f"Entscheidungsanteile nach Kognitivem Stil für Szenario '{scenario_id}'")
        plt.xlabel("Gewählte Option")
        plt.ylabel("Anteil der Entscheidungen pro Stil")
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Cognitive Style', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.show()


    def plot_ensemble_variance(self, belief_name: str):
        """Plottet die Varianz der Belief-Stärke über die Ensemble-Läufe."""
        var_df = self.analyzer.get_ensemble_variance("belief_strength_variance")
        if var_df is None or var_df.empty:
            print("Keine Ensemble-Varianzdaten gefunden.")
            return

        belief_var_df = var_df[var_df['belief'] == belief_name]
        if belief_var_df.empty:
            print(f"Keine Varianzdaten für Belief '{belief_name}'.")
            return

        plt.figure(figsize=(12, 7))
        # Plotte Varianz für jeden Agenten einzeln (kann unübersichtlich werden)
        # Alternative: Durchschnittliche Varianz über Agenten
        mean_variance = belief_var_df.groupby('step')['variance'].mean()
        plt.plot(mean_variance.index, mean_variance.values, marker='.', linestyle='-', label=f'Mean Variance for {belief_name}')

        # Optional: Zeige Agenten mit höchster Varianz
        # avg_var_per_agent = belief_var_df.groupby('agent_id')['variance'].mean().nlargest(3)
        # for agent_id in avg_var_per_agent.index:
        #    agent_data = belief_var_df[belief_var_df['agent_id'] == agent_id]
        #    plt.plot(agent_data['step'], agent_data['variance'], linestyle='--', alpha=0.7, label=f'Agent {agent_id}')

        plt.title(f"Ensemble-Varianz für Belief '{belief_name}'")
        plt.xlabel("Simulationsschritt")
        plt.ylabel("Varianz der Stärke über Ensemble-Läufe")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def visualize_social_network(self, color_by: str = 'cognitive_style', show_clusters: bool = False):
        """Visualisiert das soziale Netzwerk, optional mit Cluster-Markierung."""
        G = self.society.social_network
        if G.number_of_nodes() == 0:
            print("Soziales Netzwerk ist leer.")
            return

        plt.figure(figsize=(14, 12))
        pos = nx.spring_layout(G, seed=42, k=0.3) # k für Abstand

        node_colors = []
        legend_elements = []

        # --- Farbgebung der Knoten ---
        if color_by == 'cognitive_style':
            style_colors_map = plt.cm.tab10 # Verwende tab10 für distinkte Farben
            styles = sorted(list(NeuralProcessingType.ALL_TYPES))
            style_to_color = {style: style_colors_map(i / len(styles)) for i, style in enumerate(styles)}
            agent_styles = nx.get_node_attributes(G, 'agent_instance') # Hole Agenten-Instanzen
            node_colors = [style_to_color.get(agent.cognitive_architecture.primary_processing, 'grey')
                           for agent_id, agent in agent_styles.items() if agent] # Prüfe ob Agent existiert
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=style, markersize=10, markerfacecolor=color)
                               for style, color in style_to_color.items()]
        elif color_by == 'group':
            # Finde die dominanteste Gruppe für jeden Agenten
            group_colors_map = plt.cm.Accent
            all_groups = sorted(list(self.society.groups.keys()))
            group_to_color = {group: group_colors_map(i / len(all_groups)) for i, group in enumerate(all_groups)}
            node_colors = []
            agent_instances = nx.get_node_attributes(G, 'agent_instance')
            for node_id in G.nodes():
                 agent = agent_instances.get(node_id)
                 if agent and agent.group_identities:
                      dominant_group = max(agent.group_identities, key=agent.group_identities.get)
                      node_colors.append(group_to_color.get(dominant_group, 'lightgrey'))
                 else:
                      node_colors.append('lightgrey')
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=group, markersize=10, markerfacecolor=color)
                               for group, color in group_to_color.items()]
        # TODO: Add color_by 'belief' option
        else: # Default grey
            node_colors = 'grey'

        # --- Cluster-Visualisierung ---
        cluster_shapes = {} # agent_id -> marker shape
        if show_clusters:
            clusters = self.analyzer.identify_belief_clusters(step=-1) # Letzter Schritt
            markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h'] # Verschiedene Marker für Cluster
            for i, cluster in enumerate(clusters):
                marker = markers[i % len(markers)]
                for agent_id in cluster['agents']:
                    cluster_shapes[agent_id] = marker
            # Default für nicht-geclusterte Agenten
            default_marker = 'o'
        else:
            default_marker = 'o'

        # --- Knoten zeichnen (individuell für Marker) ---
        drawn_nodes = set()
        for shape in set(cluster_shapes.values()) | {default_marker}:
             node_list = [node for node in G.nodes() if cluster_shapes.get(node, default_marker) == shape]
             current_colors = [color for node, color in zip(G.nodes(), node_colors) if node in node_list] # Korrekte Farben zuordnen
             nx.draw_networkx_nodes(G, pos, nodelist=node_list, node_size=VIS_SOCIAL_NODE_DEGREE_SCALING,
                                    node_color=current_colors, alpha=0.8, node_shape=shape)
             drawn_nodes.update(node_list)
        # Fallback für Knoten ohne Agenten-Instanz (falls vorhanden)
        remaining_nodes = list(set(G.nodes()) - drawn_nodes)
        if remaining_nodes: nx.draw_networkx_nodes(G, pos, nodelist=remaining_nodes, node_size=100, node_color='black', alpha=0.5, node_shape=default_marker)


        # --- Kanten zeichnen ---
        edge_widths = [VIS_SOCIAL_EDGE_WEIGHT_SCALING * G[u][v]['weight'] for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.3, edge_color='grey')

        # --- Labels ---
        nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')

        plt.title(f"Soziales Netzwerk (gefärbt nach: {color_by}{', Cluster markiert' if show_clusters else ''})")
        plt.axis('off')
        if legend_elements:
            plt.legend(handles=legend_elements, title=color_by.replace('_', ' ').title(), bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.show()

# --- Beispielhafter Aufruf (ersetzt run_demo) ---
def run_full_simulation_and_analysis():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # 1. Gesellschaft erstellen (nutzt create_example_neural_society aus dem Original)
    logging.info("Erstelle Beispielgesellschaft...")
    try:
        # Korrigierter Aufruf ohne Import
        society = create_example_neural_society()
        logging.info(f"Gesellschaft erstellt: {society}")
    except Exception as e:
         logging.error(f"Fehler beim Erstellen der Gesellschaft: {e}")
         return

    # 2. Simulation durchführen
    logging.info("Starte Simulation...")
    try:
        results = society.run_robust_simulation(
            num_steps=20, # Mehr Schritte für aussagekräftigere Trends
            scenario_probability=0.3,
            social_influence_probability=0.4,
            reflection_probability=0.15
        )
        logging.info("Simulation abgeschlossen.")
        # Optional: Ergebnisse speichern
        # with open("simulation_results.pkl", "wb") as f: pickle.dump(results, f)
    except Exception as e:
        logging.error(f"Fehler während der Simulation: {e}")
        return

    # 3. Analyse durchführen
    logging.info("Analysiere Ergebnisse...")
    analyzer = SimulationAnalyzer(society, results)

    # 4. Visualisieren
    logging.info("Visualisiere Ergebnisse...")
    visualizer = SimulationVisualizer(society, analyzer)

    # Beispiel-Visualisierungen:
    common_beliefs = list(society.belief_templates.keys())
    if common_beliefs:
        visualizer.plot_belief_evolution(common_beliefs[0], show_mean=True, show_styles=True) # Ersten Belief plotten
        visualizer.plot_polarization_trend(belief_name=common_beliefs[0], metric='variance')
        # visualizer.plot_polarization_trend(metric='bimodality') # Durchschnittliche Bimodalität

    common_scenarios = list(society.scenarios.keys())
    if common_scenarios:
        visualizer.plot_decision_distribution(common_scenarios[0])
        visualizer.plot_cognitive_style_decision_comparison(common_scenarios[0])

    # Ensemble Varianz für einen Belief
    if common_beliefs:
         visualizer.plot_ensemble_variance(common_beliefs[0])

    # Soziales Netzwerk
    visualizer.visualize_social_network(color_by='cognitive_style', show_clusters=True)
    visualizer.visualize_social_network(color_by='group')

    logging.info("Analyse und Visualisierung abgeschlossen.")

def create_example_neural_society() -> NeuralEthicalSociety:
    """
    Erstellt eine einfache Beispielgesellschaft mit einigen Agenten und einem Szenario.
    Diese Funktion dient als Beispiel und für Testzwecke.
    
    Returns:
        Eine initialisierte NeuralEthicalSociety mit Agenten und einem Szenario
    """
    society = NeuralEthicalSociety()
    
    # Einige Belief-Templates hinzufügen
    society.add_belief_template(
        name="Autonomie",
        category="Individualrechte",
        connections={"Solidarität": (0.3, -1), "Verantwortung": (0.4, 1)},
        associated_concepts={"Freiheit": 0.8, "Selbstbestimmung": 0.9}
    )
    
    society.add_belief_template(
        name="Solidarität",
        category="Gemeinschaftswerte",
        connections={"Autonomie": (0.3, -1), "Verantwortung": (0.5, 1)},
        associated_concepts={"Gemeinschaft": 0.7, "Zusammenhalt": 0.8}
    )
    
    society.add_belief_template(
        name="Verantwortung",
        category="Pflichten",
        connections={"Autonomie": (0.4, 1), "Solidarität": (0.5, 1)},
        associated_concepts={"Pflicht": 0.6, "Führsorge": 0.5}
    )
    
    # Einige zufällige Agenten erstellen
    society.generate_diverse_society(num_agents=10, num_archetypes=3)
    
    # Ein einfaches Szenario hinzufügen
    szenario = EthicalScenario(
        scenario_id="covid_dilemma",
        description="Eine Pandemie erfordert Einschränkungen persönlicher Freiheiten zum Schutz der Gemeinschaft.",
        relevant_beliefs={"Autonomie": 0.8, "Solidarität": 0.9, "Verantwortung": 0.7},
        options={
            "strenge_regeln": {
                "Autonomie": -0.7,
                "Solidarität": 0.8,
                "Verantwortung": 0.5
            },
            "freiwillige_maßnahmen": {
                "Autonomie": 0.6,
                "Solidarität": -0.2,
                "Verantwortung": 0.2
            },
            "keine_maßnahmen": {
                "Autonomie": 0.9,
                "Solidarität": -0.8,
                "Verantwortung": -0.5
            }
        },
        option_attributes={
            "strenge_regeln": {"risks": 0.4, "group_norms": {"Kollektivisten": 0.8, "Individualisten": -0.5}},
            "freiwillige_maßnahmen": {"risks": 0.6, "group_norms": {"Kollektivisten": 0.2, "Individualisten": 0.5}},
            "keine_maßnahmen": {"risks": 0.9, "group_norms": {"Kollektivisten": -0.7, "Individualisten": 0.9}}
        },
        outcome_feedback={
            "strenge_regeln": {"Autonomie": -0.2, "Solidarität": 0.3, "Verantwortung": 0.2},
            "freiwillige_maßnahmen": {"Autonomie": 0.1, "Solidarität": 0.1, "Verantwortung": 0.1},
            "keine_maßnahmen": {"Autonomie": 0.2, "Solidarität": -0.4, "Verantwortung": -0.3}
        },
        moral_implications={
            "strenge_regeln": {"care": 0.7, "fairness": 0.5, "loyalty": 0.8, "authority": 0.8, "purity": 0.3, "liberty": -0.6},
            "freiwillige_maßnahmen": {"care": 0.4, "fairness": 0.6, "loyalty": 0.3, "authority": 0.2, "purity": 0.1, "liberty": 0.5},
            "keine_maßnahmen": {"care": -0.5, "fairness": -0.2, "loyalty": -0.4, "authority": -0.6, "purity": -0.2, "liberty": 0.9}
        },
        emotional_valence=-0.3,
        narrative_elements={
            "characters": ["Gesundheitsexperten", "Bürger", "Regierung"],
            "conflict": "Kollektive Gesundheit vs. persönliche Freiheit",
            "context": "Globale Gesundheitskrise",
            "coherence": 0.8
        }
    )
    
    society.add_scenario(szenario)
    
    # Ein paar Gruppen hinzufügen
    agent_ids = list(society.agents.keys())
    half = len(agent_ids) // 2
    
    society.add_group("Kollektivisten", agent_ids[:half])
    society.add_group("Individualisten", agent_ids[half:])
    
    # Soziales Netzwerk generieren
    society.generate_realistic_social_network()
    
    return society

# Main-Block wird ganz zum Schluss platziert
if __name__ == "__main__":
    # Stelle sicher, dass die benötigten Klassen definiert sind
    # (Normalerweise durch Importe am Anfang des Skripts)
    # Führe die Demo aus
    run_full_simulation_and_analysis()
