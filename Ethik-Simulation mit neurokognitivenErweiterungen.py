
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import cosine
from sklearn.manifold import TSNE
from scipy.stats import entropy
import seaborn as sns
from tqdm import tqdm
import random
from typing import List, Dict, Tuple, Optional, Set, Union, Callable, Any # Added Any
import pickle
import os
from ethik.visualization import ( # Added
    visualize_neural_processing,
    visualize_belief_network,
    visualize_cognitive_style_comparison,
    visualize_social_network
)

class NeuralProcessingType:
    """Repräsentiert unterschiedliche neuronale Verarbeitungsstile."""
    
    SYSTEMATIC = "systematic"       # Stark analytisch, schrittweise
    INTUITIVE = "intuitive"         # Schnell, ganzheitlich, gefühlsbasiert
    ASSOCIATIVE = "associative"     # Netzwerkartig, assoziativ 
    ANALOGICAL = "analogical"       # Basierend auf Analogien und Metaphern
    EMOTIONAL = "emotional"         # Stark emotionsgesteuert
    NARRATIVE = "narrative"         # Informationsverarbeitung durch Geschichten
    
    @staticmethod
    def get_random():
        """Liefert einen zufälligen Verarbeitungstyp."""
        types = [NeuralProcessingType.SYSTEMATIC, NeuralProcessingType.INTUITIVE, 
                NeuralProcessingType.ASSOCIATIVE, NeuralProcessingType.ANALOGICAL,
                NeuralProcessingType.EMOTIONAL, NeuralProcessingType.NARRATIVE]
        return random.choice(types)


class CognitiveArchitecture:
    """Modelliert die kognitive Architektur eines Agenten."""
    
    def __init__(self, 
                primary_processing: str = NeuralProcessingType.SYSTEMATIC,
                secondary_processing: str = NeuralProcessingType.INTUITIVE,
                processing_balance: float = 0.5):
        """
        Initialisiert die kognitive Architektur.
        
        Args:
            primary_processing: Primärer Verarbeitungstyp
            secondary_processing: Sekundärer Verarbeitungstyp
            processing_balance: Balance zwischen primärer und sekundärer Verarbeitung (0-1)
        """
        self.primary_processing = primary_processing
        self.secondary_processing = secondary_processing
        self.processing_balance = np.clip(processing_balance, 0.0, 1.0)
        
        # Kognitive Verzerrungen (Biases)
        self.cognitive_biases = {
            "confirmation_bias": np.random.beta(3, 7),  # Bestätigungsfehler
            "availability_bias": np.random.beta(5, 5),  # Verfügbarkeitsheuristik
            "anchoring_bias": np.random.beta(5, 5),     # Ankereffekt
            "authority_bias": np.random.beta(5, 5),     # Autoritätsverzerrung
            "ingroup_bias": np.random.beta(5, 5)        # Eigengruppenfavorisierung
        }
        
        # Emotionale Parameter
        self.emotional_parameters = {
            "emotional_reactivity": np.random.beta(5, 5),     # Emotionale Reaktivität
            "emotional_regulation": np.random.beta(5, 5),     # Emotionsregulation
            "empathy": np.random.beta(5, 5),                  # Empathie
            "negativity_bias": np.random.beta(6, 4)           # Negativitätsverzerrung 
        }
        
        # Bayes'sche Verarbeitungsparameter
        self.bayesian_parameters = {
            "prior_strength": np.random.beta(5, 5),          # Stärke der Vorannahmen
            "evidence_threshold": np.random.beta(5, 5),      # Schwellenwert für Beweisannahme
            "update_rate": np.random.beta(5, 5)              # Geschwindigkeit des Belief-Updates
        }
        
        # Neuronale Aktivierungsfunktionen für verschiedene Verarbeitungstypen
        self.activation_functions = {
            NeuralProcessingType.SYSTEMATIC: self._systematic_activation,
            NeuralProcessingType.INTUITIVE: self._intuitive_activation,
            NeuralProcessingType.ASSOCIATIVE: self._associative_activation,
            NeuralProcessingType.ANALOGICAL: self._analogical_activation,
            NeuralProcessingType.EMOTIONAL: self._emotional_activation,
            NeuralProcessingType.NARRATIVE: self._narrative_activation
        }
        
    def _systematic_activation(self, inputs: Dict[str, float], 
                              context: Dict[str, float] = None) -> Dict[str, float]:
        """
        Systematische, schrittweise Aktivierungsfunktion.
        Betont logische Konsistenz und sequentielle Verarbeitung.
        """
        results = {}
        for key, value in inputs.items():
            # Systematisches Denken reduziert den Einfluss von Verzerrungen
            bias_reduction = 0.7  # Reduktion der Verzerrungen
            
            # Angepasster Wert mit reduziertem Bias-Einfluss
            results[key] = value * (1.0 - self.cognitive_biases.get("confirmation_bias", 0) * 
                                   (1.0 - bias_reduction))
        return results
        
    def _intuitive_activation(self, inputs: Dict[str, float],
                             context: Dict[str, float] = None) -> Dict[str, float]:
        """
        Intuitive, schnelle Aktivierungsfunktion.
        Betont Gesamteindrücke und unmittelbare Reaktionen.
        """
        results = {}
        for key, value in inputs.items():
            # Intuitive Reaktionen sind stärker durch Verfügbarkeitsheuristiken beeinflusst
            availability_effect = self.cognitive_biases.get("availability_bias", 0) * 0.5
            
            # Angepasster Wert mit verstärktem Einfluss verfügbarer Informationen
            if context and key in context:
                context_influence = context[key] * availability_effect
                results[key] = value * (1.0 + context_influence)
            else:
                results[key] = value
        return results
        
    def _associative_activation(self, inputs: Dict[str, float],
                               context: Dict[str, float] = None) -> Dict[str, float]:
        """
        Assoziative Aktivierungsfunktion mit netzwerkartiger Aktivierung.
        Aktiviert verbundene Konzepte basierend auf semantischer Nähe.
        """
        results = {}
        
        # Schwellenwert für Aktivierungsübertragung
        activation_threshold = 0.3
        
        for key, value in inputs.items():
            # Basisaktivierung
            results[key] = value
            
            # Aktivierung verbundener Konzepte (falls kontext-informationen verfügbar)
            if context:
                for other_key, other_value in context.items():
                    if other_key != key and other_value > activation_threshold:
                        # Einfaches assoziatives Spreading
                        association_strength = other_value * 0.4  # Dämpfungsfaktor
                        results[key] = max(results[key], value * (1.0 + association_strength))
        
        return results
        
    def _analogical_activation(self, inputs: Dict[str, float],
                              context: Dict[str, float] = None) -> Dict[str, float]:
        """
        Aktivierungsfunktion basierend auf Analogien und Ähnlichkeiten.
        Betont strukturelle Ähnlichkeiten zwischen Konzepten.
        """
        results = {}
        
        # Parameter für die Analogiestärke
        analogy_strength = 0.5
        
        for key, value in inputs.items():
            # Basisaktivierung
            results[key] = value
            
            # Analogien verstärken ähnliche Konzepte (Simulation)
            if context:
                analogical_boost = sum(0.1 * v for k, v in context.items() if k != key) * analogy_strength
                results[key] *= (1.0 + analogical_boost)
        
        return results
        
    def _emotional_activation(self, inputs: Dict[str, float],
                             context: Dict[str, float] = None) -> Dict[str, float]:
        """
        Emotionsgesteuerte Aktivierungsfunktion.
        Betont emotionale Reaktionen und somatische Marker.
        """
        results = {}
        
        # Emotionale Reaktivität und Regulation beeinflussen die Aktivierung
        reactivity = self.emotional_parameters.get("emotional_reactivity", 0.5)
        regulation = self.emotional_parameters.get("emotional_regulation", 0.5)
        negativity_bias = self.emotional_parameters.get("negativity_bias", 0.6)
        
        for key, value in inputs.items():
            # Basisaktivierung
            base_activation = value
            
            # Emotionale Modulation (Simulation)
            if context and "emotional_valence" in context:
                valence = context["emotional_valence"]  # Positive oder negative Valenz (-1 bis +1)
                
                # Negativitätsverzerrung verstärkt negative Eindrücke
                if valence < 0:
                    emotional_effect = abs(valence) * reactivity * negativity_bias
                else:
                    emotional_effect = valence * reactivity * (1 - negativity_bias)
                
                # Regulation dämpft emotionale Effekte
                regulated_effect = emotional_effect * (1.0 - regulation * 0.5)
                
                # Angepasster Wert mit emotionalem Einfluss
                results[key] = base_activation * (1.0 + regulated_effect)
            else:
                results[key] = base_activation
        
        return results
        
    def _narrative_activation(self, inputs: Dict[str, float],
                             context: Dict[str, float] = None) -> Dict[str, float]:
        """
        Narrativbasierte Aktivierungsfunktion.
        Betont Kohärenz und Stimmigkeit in einer Geschichte.
        """
        results = {}
        
        # Kohärenz-Parameter simulieren
        coherence_bias = 0.3
        
        for key, value in inputs.items():
            # Basisaktivierung
            results[key] = value
            
            # Kohärenzeffekte (Simulation)
            if context and "narrative_coherence" in context:
                coherence = context["narrative_coherence"]  # 0 bis 1
                
                # Verstärkung für kohärente Narrative
                narrative_effect = coherence * coherence_bias
                results[key] *= (1.0 + narrative_effect)
        
        return results
    
    def process_information(self, inputs: Dict[str, float], 
                          context: Dict[str, float] = None) -> Dict[str, float]:
        """
        Verarbeitet Informationen gemäß der kognitiven Architektur.
        
        Kombiniert primäre und sekundäre Verarbeitungstypen.
        
        Args:
            inputs: Eingabewerte für verschiedene Konzepte
            context: Kontextinformationen zur Beeinflussung der Verarbeitung
            
        Returns:
            Verarbeitete Werte
        """
        # Primäre Verarbeitung
        primary_results = self.activation_functions[self.primary_processing](inputs, context)
        
        # Sekundäre Verarbeitung (falls vorhanden)
        if self.secondary_processing:
            secondary_results = self.activation_functions[self.secondary_processing](inputs, context)
            
            # Gewichtete Kombination der Ergebnisse
            results = {}
            for key in inputs.keys():
                primary_value = primary_results.get(key, 0)
                secondary_value = secondary_results.get(key, 0)
                
                # Gewichtete Kombination basierend auf processing_balance
                results[key] = (self.processing_balance * primary_value + 
                              (1.0 - self.processing_balance) * secondary_value)
        else:
            results = primary_results
        
        return results
    
    def apply_bayesian_update(self, prior_belief: float, evidence_strength: float, 
                            evidence_direction: int) -> float:
        """
        Wendet ein Bayes'sches Update auf eine Überzeugung an.
        
        Args:
            prior_belief: Vorherige Stärke der Überzeugung (0-1)
            evidence_strength: Stärke des Beweises (0-1)
            evidence_direction: Richtung des Beweises (+1 bestätigend, -1 widersprechend)
            
        Returns:
            Aktualisierte Überzeugungsstärke
        """
        # Bayes'sche Parameter abrufen
        prior_strength = self.bayesian_parameters.get("prior_strength", 0.5)
        update_rate = self.bayesian_parameters.get("update_rate", 0.5)
        
        # Skalierung der Beweisgewichtung basierend auf Verarbeitungstyp
        if self.primary_processing == NeuralProcessingType.SYSTEMATIC:
            evidence_weight = 0.7  # Systematisches Denken gewichtet Beweise stärker
        elif self.primary_processing == NeuralProcessingType.EMOTIONAL:
            evidence_weight = 0.3  # Emotionales Denken gewichtet Beweise weniger
        else:
            evidence_weight = 0.5  # Neutrale Gewichtung
        
        # Angepasste Stärke der Vorannahme
        adjusted_prior_strength = prior_strength * (1.0 - evidence_weight)
        
        # Angepasste Stärke des Beweises
        adjusted_evidence_strength = evidence_strength * evidence_weight
        
        # Einfaches Bayes'sches Update (vereinfacht)
        if evidence_direction > 0:  # Bestätigender Beweis
            belief_change = adjusted_evidence_strength * (1.0 - prior_belief) * update_rate
        else:  # Widersprechender Beweis
            belief_change = -adjusted_evidence_strength * prior_belief * update_rate
            
        # Prior-Anker-Effekt
        anchoring_effect = adjusted_prior_strength * (prior_belief - 0.5) * 0.2
        
        # Neue Überzeugungsstärke
        new_belief = prior_belief + belief_change + anchoring_effect
        
        return np.clip(new_belief, 0.0, 1.0)
    
    def __str__(self):
        """String-Repräsentation der kognitiven Architektur."""
        return f"Kognitive Architektur: {self.primary_processing} ({self.processing_balance:.2f}) / {self.secondary_processing} ({1-self.processing_balance:.2f})"


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


class NeuralEthicalAgent:
    """Repräsentiert einen ethischen Agenten mit neuronalen Verarbeitungsmodellen."""
    
    def __init__(self, agent_id: str, personality_traits: Dict[str, float] = None):
        """
        Initialisiert einen neuronalen ethischen Agenten.
        
        Args:
            agent_id: Eindeutige ID des Agenten
            personality_traits: Persönlichkeitsmerkmale des Agenten
        """
        self.agent_id = agent_id
        self.beliefs = {}  # name -> NeuralEthicalBelief
        
        # Persönlichkeitsmerkmale (Big Five)
        self.personality_traits = personality_traits or {
            "openness": np.random.beta(5, 5),          # Offenheit für neue Ideen
            "conscientiousness": np.random.beta(5, 5),  # Gewissenhaftigkeit
            "extroversion": np.random.beta(5, 5),      # Extroversion
            "agreeableness": np.random.beta(5, 5),     # Verträglichkeit
            "neuroticism": np.random.beta(5, 5)        # Neurotizismus
        }
        
        # Kognitive Architektur
        self.cognitive_architecture = self._generate_cognitive_architecture()
        
        # Historische Entscheidungen und Überzeugungsstärken
        self.decision_history = []
        self.belief_strength_history = {}
        self.belief_certainty_history = {}
        
        # Soziales Netzwerk - IDs anderer Agenten und Stärke der Verbindung
        self.social_connections = {}  # agent_id -> connection_strength
        
        # Gruppenidentitäten
        self.group_identities = {}  # group_name -> identification_strength
        
        # Moralische Grundlagen (nach Moral Foundations Theory)
        self.moral_foundations = {
            "care": np.random.beta(5, 5),             # Fürsorge/Schutz vor Schaden
            "fairness": np.random.beta(5, 5),         # Fairness/Gerechtigkeit
            "loyalty": np.random.beta(5, 5),          # Loyalität zur Gruppe
            "authority": np.random.beta(5, 5),        # Respekt vor Autorität
            "purity": np.random.beta(5, 5),           # Reinheit/Heiligkeit
            "liberty": np.random.beta(5, 5)           # Freiheit (zusätzliche Dimension)
        }
        
        # Arbeitsspeicher (für kognitives System)
        self.working_memory = {
            "capacity": 5 + int(2 * self.personality_traits["conscientiousness"]),  # 5-7 Elemente
            "contents": [],  # Aktuelle Inhalte
            "retention": 0.7 + 0.3 * self.personality_traits["conscientiousness"]  # Beibehaltungsrate
        }
        
        # Episodisches Gedächtnis (für wichtige Erfahrungen)
        self.episodic_memory = []
        
        # Aktivierungsniveau des gesamten Überzeugungsnetzwerks
        self.current_time = 0  # Simulationszeit für Aktivierungsdynamik
        
    def _generate_cognitive_architecture(self) -> CognitiveArchitecture:
        """Generiert eine zur Persönlichkeit passende kognitive Architektur."""
        # Primärer Prozesstyp basierend auf Persönlichkeit auswählen
        personality = self.personality_traits
        
        # Offene, reflektierte Menschen neigen zu systematischem Denken
        if personality["openness"] > 0.7 and personality["conscientiousness"] > 0.6:
            primary = NeuralProcessingType.SYSTEMATIC
        # Empathische, extrovertierte Menschen neigen zu emotionalem/narrativem Denken
        elif personality["agreeableness"] > 0.7 and personality["extroversion"] > 0.6:
            primary = random.choice([NeuralProcessingType.EMOTIONAL, NeuralProcessingType.NARRATIVE])
        # Kreative, offene Menschen neigen zu analogischem/assoziativem Denken
        elif personality["openness"] > 0.7:
            primary = random.choice([NeuralProcessingType.ANALOGICAL, NeuralProcessingType.ASSOCIATIVE])
        # Neurotische Menschen neigen zu intuitivem/emotionalem Denken
        elif personality["neuroticism"] > 0.7:
            primary = random.choice([NeuralProcessingType.INTUITIVE, NeuralProcessingType.EMOTIONAL])
        else:
            primary = NeuralProcessingType.get_random()
            
        # Sekundären Prozesstyp auswählen (komplementär zum primären)
        if primary == NeuralProcessingType.SYSTEMATIC:
            secondary = random.choice([NeuralProcessingType.INTUITIVE, NeuralProcessingType.EMOTIONAL])
        elif primary in [NeuralProcessingType.EMOTIONAL, NeuralProcessingType.INTUITIVE]:
            secondary = random.choice([NeuralProcessingType.SYSTEMATIC, NeuralProcessingType.ASSOCIATIVE])
        else:
            all_types = [NeuralProcessingType.SYSTEMATIC, NeuralProcessingType.INTUITIVE, 
                        NeuralProcessingType.ASSOCIATIVE, NeuralProcessingType.ANALOGICAL,
                        NeuralProcessingType.EMOTIONAL, NeuralProcessingType.NARRATIVE]
            secondary_options = [t for t in all_types if t != primary]
            secondary = random.choice(secondary_options)
            
        # Balance basierend auf Persönlichkeit
        if primary == NeuralProcessingType.SYSTEMATIC:
            # Gewissenhaftere Menschen verlassen sich mehr auf systematisches Denken
            balance = 0.5 + 0.3 * personality["conscientiousness"]
        elif primary in [NeuralProcessingType.EMOTIONAL, NeuralProcessingType.INTUITIVE]:
            # Neurotischere Menschen verlassen sich mehr auf emotionales/intuitives Denken
            balance = 0.5 + 0.3 * personality["neuroticism"]
        else:
            balance = np.random.uniform(0.6, 0.8)  # Primärer Stil dominiert normalerweise
            
        return CognitiveArchitecture(primary, secondary, balance)
        
    def add_belief(self, belief: NeuralEthicalBelief):
        """Fügt eine ethische Überzeugung hinzu."""
        self.beliefs[belief.name] = belief
        self.belief_strength_history[belief.name] = [belief.strength]
        self.belief_certainty_history[belief.name] = [belief.certainty]
        
    def update_belief(self, belief_name: str, new_strength: float, 
                     new_certainty: Optional[float] = None,
                     new_valence: Optional[float] = None):
        """Aktualisiert die Parameter einer Überzeugung."""
        if belief_name in self.beliefs:
            belief = self.beliefs[belief_name]
            old_strength = belief.strength
            
            # Stärke aktualisieren
            belief.update_strength(new_strength)
            self.belief_strength_history[belief_name].append(new_strength)
            
            # Gewissheit aktualisieren (falls angegeben)
            if new_certainty is not None:
                belief.update_certainty(new_certainty)
                if belief_name in self.belief_certainty_history:
                    self.belief_certainty_history[belief_name].append(new_certainty)
                else:
                    self.belief_certainty_history[belief_name] = [new_certainty]
                    
            # Emotionale Valenz aktualisieren (falls angegeben)
            if new_valence is not None:
                belief.update_emotional_valence(new_valence)
                
            return new_strength - old_strength
        return 0.0
    
    def add_social_connection(self, agent_id: str, connection_strength: float):
        """Fügt eine soziale Verbindung zu einem anderen Agenten hinzu."""
        self.social_connections[agent_id] = np.clip(connection_strength, 0.0, 1.0)
    
    def add_group_identity(self, group_name: str, identification_strength: float):
        """Fügt eine Gruppenidentität hinzu."""
        self.group_identities[group_name] = np.clip(identification_strength, 0.0, 1.0)
    
    def get_belief_vector(self) -> np.ndarray:
        """Gibt einen Vektor mit allen Überzeugungsstärken zurück."""
        return np.array([belief.strength for belief in self.beliefs.values()])
    
    def get_belief_names(self) -> List[str]:
        """Gibt die Namen aller Überzeugungen zurück."""
        return list(self.beliefs.keys())
    
    def calculate_cognitive_dissonance(self) -> float:
        """Berechnet die kognitive Dissonanz basierend auf widersprüchlichen Überzeugungen."""
        dissonance = 0.0
        processed_pairs = set()
        
        for belief_name, belief in self.beliefs.items():
            for other_name, (influence, polarity) in belief.connections.items():
                if other_name in self.beliefs and (belief_name, other_name) not in processed_pairs:
                    # Dissonanz entsteht, wenn starke Überzeugungen gegensätzlich verbunden sind
                    if polarity < 0:
                        # Gewichtung mit Gewissheit (höhere Gewissheit = mehr Dissonanz)
                        certainty_weight = belief.certainty * self.beliefs[other_name].certainty
                        dissonance += (belief.strength * self.beliefs[other_name].strength * 
                                     influence * abs(polarity) * certainty_weight)
                    processed_pairs.add((belief_name, other_name))
                    processed_pairs.add((other_name, belief_name))
                    
        # Anpassung basierend auf kognitiver Architektur
        if self.cognitive_architecture.primary_processing == NeuralProcessingType.SYSTEMATIC:
            # Systematische Denker spüren Dissonanz stärker
            dissonance *= 1.2
        elif self.cognitive_architecture.primary_processing == NeuralProcessingType.INTUITIVE:
            # Intuitive Denker spüren weniger Dissonanz
            dissonance *= 0.8
            
        return dissonance
        
    def spreading_activation(self, seed_beliefs: List[str], activation_levels: List[float]):
        """
        Führt spreading activation im Überzeugungsnetzwerk durch.
        
        Args:
            seed_beliefs: Liste der Initial-Überzeugungen für die Aktivierung
            activation_levels: Aktivierungsniveaus für die Seed-Überzeugungen
        """
        self.current_time += 1
        
        # Initial-Aktivierung
        for belief_name, activation_level in zip(seed_beliefs, activation_levels):
            if belief_name in self.beliefs:
                self.beliefs[belief_name].activate(activation_level, self.current_time)
        
        # Spreading activation (2 Durchgänge)
        for _ in range(2):
            # Aktivierungswerte für diesen Durchgang speichern
            activations = {name: belief.activation for name, belief in self.beliefs.items()}
            
            # Aktivierung verbreiten
            for belief_name, belief in self.beliefs.items():
                if belief.activation > 0.1:  # Mindestschwelle für Spreading
                    for conn_name, (strength, polarity) in belief.connections.items():
                        if conn_name in self.beliefs:
                            # Aktivierung weitergeben
                            spread_activation = belief.activation * strength * 0.5
                            
                            # Polarität berücksichtigen (negative Verbindungen hemmen)
                            if polarity < 0:
                                # Hemmung statt Aktivierung
                                self.beliefs[conn_name].activation *= (1.0 - spread_activation * 0.3)
                            else:
                                # Aktivierung
                                self.beliefs[conn_name].activate(spread_activation, self.current_time)
            
            # Assoziative Aktivierung (falls relevant)
            if self.cognitive_architecture.primary_processing == NeuralProcessingType.ASSOCIATIVE:
                for belief_name, belief in self.beliefs.items():
                    if belief.activation > 0.1:
                        for concept, strength in belief.associated_concepts.items():
                            # Assoziative Aktivierung zu verbundenen Konzepten
                            for other_name, other_belief in self.beliefs.items():
                                if concept in other_belief.associated_concepts:
                                    assoc_strength = strength * other_belief.associated_concepts[concept]
                                    assoc_activation = belief.activation * assoc_strength * 0.3
                                    other_belief.activate(assoc_activation, self.current_time)
        
        # Decay nach Aktivierung
        for belief in self.beliefs.values():
            belief.activation *= 0.8  # 20% Decay pro Runde
    
    def make_decision(self, scenario: 'EthicalScenario') -> Dict[str, Union[str, float, Dict]]:
        """
        Trifft eine Entscheidung in einem ethischen Szenario.
        
        Returns:
            Dict mit der Entscheidung und Begründungen
        """
        # Relevante Überzeugungen aktivieren
        seed_beliefs = list(scenario.relevant_beliefs.keys())
        activation_levels = [scenario.relevant_beliefs[b] for b in seed_beliefs 
                           if b in self.beliefs]
        seed_beliefs = [b for b in seed_beliefs if b in self.beliefs]
        
        # Spreading Activation durchführen
        self.spreading_activation(seed_beliefs, activation_levels)
        
        # Überzeugungswerte mit kognitiver Architektur verarbeiten
        belief_inputs = {name: belief.strength for name, belief in self.beliefs.items()}
        
        # Kontext für die Verarbeitung bereitstellen
        context = {
            "scenario": scenario.scenario_id,
            "emotional_valence": np.mean([self.beliefs[b].emotional_valence for b in seed_beliefs 
                                       if b in self.beliefs]) if seed_beliefs else 0,
            "narrative_coherence": 0.7  # Beispielwert
        }
        
        # Verarbeitung durch kognitive Architektur
        processed_beliefs = self.cognitive_architecture.process_information(belief_inputs, context)
        
        # Optionen bewerten
        option_scores = {}
        for option_name, option_impacts in scenario.options.items():
            score = 0
            justifications = {}
            
            # Moralische Grundlagen einbeziehen
            moral_contribution = 0
            for foundation, strength in self.moral_foundations.items():
                if foundation in scenario.moral_implications.get(option_name, {}):
                    moral_impact = scenario.moral_implications[option_name][foundation]
                    moral_contribution += strength * moral_impact
                    justifications[f"moral_{foundation}"] = strength * moral_impact
            
            # Gewichtung der moralischen Grundlagen
            moral_weight = 0.3
            score += moral_contribution * moral_weight
            
            # Überzeugungsbeitrag
            for belief_name, impact in option_impacts.items():
                if belief_name in processed_beliefs:
                    belief_score = processed_beliefs[belief_name] * impact
                    score += belief_score
                    justifications[belief_name] = belief_score
            
            # Persönlichkeitseinflüsse
            if "risks" in scenario.option_attributes.get(option_name, {}):
                risk_aversion = 0.7 - 0.4 * self.personality_traits["openness"]
                risk_adjustment = -scenario.option_attributes[option_name]["risks"] * risk_aversion
                score += risk_adjustment
                justifications["risk_consideration"] = risk_adjustment
            
            # Gruppennormen berücksichtigen
            group_influence = 0
            if "group_norms" in scenario.option_attributes.get(option_name, {}):
                for group, norm_alignment in scenario.option_attributes[option_name]["group_norms"].items():
                    if group in self.group_identities:
                        identification = self.group_identities[group]
                        group_influence += identification * norm_alignment
                
                # Ingroup-Bias verstärkt Gruppeneinfluss
                ingroup_bias = self.cognitive_architecture.cognitive_biases.get("ingroup_bias", 0.5)
                group_influence *= (1.0 + ingroup_bias)
                justifications["group_norms"] = group_influence
                
                score += group_influence * 0.2  # 20% Gewichtung für Gruppennormen
                
            option_scores[option_name] = {
                "score": score,
                "justifications": justifications
            }
        
        # Entscheidungsfindung basierend auf kognitiver Architektur
        decision = self._finalize_decision(scenario, option_scores)
        
        # Entscheidung zur Historie hinzufügen
        self.decision_history.append(decision)
        
        # Episodisches Gedächtnis aktualisieren (wichtige Entscheidungen speichern)
        if decision["cognitive_dissonance"] > 0.3 or abs(decision["confidence"]) > 0.7:
            self.episodic_memory.append({
                "time": self.current_time,
                "type": "significant_decision",
                "scenario": scenario.scenario_id,
                "decision": decision["chosen_option"],
                "dissonance": decision["cognitive_dissonance"],
                "confidence": decision["confidence"]
            })
        
        return decision
    
    def _finalize_decision(self, scenario: 'EthicalScenario', 
                         option_scores: Dict[str, Dict]) -> Dict:
        """
        Finalisiert die Entscheidung basierend auf der kognitiven Architektur.
        """
        options = list(option_scores.keys())
        scores = [option_scores[opt]["score"] for opt in options]
        
        # Unterschiedliche Entscheidungsstrategien je nach kognitiver Architektur
        if self.cognitive_architecture.primary_processing == NeuralProcessingType.SYSTEMATIC:
            # Systematische Denker: rationaler, konsistenter
            randomness_factor = 0.1
            # Vergleich mit früheren ähnlichen Entscheidungen für Konsistenz
            consistency_boost = self._calculate_consistency_boost(scenario, options, scores)
            adjusted_scores = np.array(scores) + consistency_boost
        
        elif self.cognitive_architecture.primary_processing == NeuralProcessingType.INTUITIVE:
            # Intuitive Denker: schneller, rauschempfindlicher
            randomness_factor = 0.3
            # Verfügbarkeitsheuristik einbeziehen
            availability_bias = self.cognitive_architecture.cognitive_biases.get("availability_bias", 0.5)
            recency_boost = self._calculate_recency_boost(options) * availability_bias
            adjusted_scores = np.array(scores) + recency_boost
            
        elif self.cognitive_architecture.primary_processing == NeuralProcessingType.EMOTIONAL:
            # Emotionale Denker: emotionsgesteuert
            randomness_factor = 0.2
            # Emotionale Reaktionen einbeziehen
            emotional_reactivity = self.cognitive_architecture.emotional_parameters.get("emotional_reactivity", 0.5)
            emotion_boost = self._calculate_emotion_boost(scenario, options) * emotional_reactivity
            adjusted_scores = np.array(scores) + emotion_boost
            
        else:
            # Andere Denkstile
            randomness_factor = 0.2
            adjusted_scores = np.array(scores)
        
        # Zufallskomponente hinzufügen (menschliche Unberechenbarkeit)
        decision_noise = np.random.normal(0, randomness_factor, len(adjusted_scores))
        final_scores = adjusted_scores + decision_noise
        
        # Option mit höchstem Score wählen
        chosen_index = np.argmax(final_scores)
        chosen_option = options[chosen_index]
        
        # Konfidenz berechnen (Abstand zum nächsthöchsten Score)
        confidence = 0.5
        if len(final_scores) > 1:
            sorted_scores = np.sort(final_scores)
            score_diff = sorted_scores[-1] - sorted_scores[-2]  # Differenz zwischen höchstem und zweithöchstem
            confidence = np.tanh(score_diff * 2)  # Tanh für Skalierung auf ~(-1,1)
        
        # Bayes'sche Aktualisierung von Überzeugungen während der Entscheidung
        # (Simulation von Überzeugungsänderungen während des Nachdenkens)
        belief_updates = {}
        for belief_name in scenario.relevant_beliefs:
            if belief_name in self.beliefs:
                # Finde die Auswirkung der gewählten Option auf diese Überzeugung
                if belief_name in scenario.options[chosen_option]:
                    impact = scenario.options[chosen_option][belief_name]
                    
                    # Bayes'sches Update basierend auf "Nachdenken"
                    old_strength = self.beliefs[belief_name].strength
                    evidence_strength = abs(impact) * 0.1  # Schwacher Effekt
                    evidence_direction = 1 if impact > 0 else -1
                    
                    new_strength = self.cognitive_architecture.apply_bayesian_update(
                        old_strength, evidence_strength, evidence_direction)
                    
                    # Überzeugung leicht anpassen
                    if abs(new_strength - old_strength) > 0.01:
                        self.update_belief(belief_name, new_strength)
                        belief_updates[belief_name] = new_strength - old_strength
        
        return {
            "scenario_id": scenario.scenario_id,
            "chosen_option": chosen_option,
            "option_scores": option_scores,
            "cognitive_dissonance": self.calculate_cognitive_dissonance(),
            "confidence": confidence,
            "belief_updates": belief_updates,
            "timestamp": self.current_time
        }
    
    def _calculate_consistency_boost(self, scenario: 'EthicalScenario', 
                                   options: List[str], 
                                   scores: List[float]) -> np.ndarray:
        """Berechnet einen Konsistenz-Boost basierend auf früheren Entscheidungen."""
        consistency_boost = np.zeros_like(scores)
        
        # Ähnliche frühere Szenarien finden
        similar_decisions = []
        for past_decision in self.decision_history[-10:]:  # Letzte 10 Entscheidungen
            if past_decision["scenario_id"] == scenario.scenario_id:
                similar_decisions.append(past_decision)
        
        if similar_decisions:
            # Konsistenz mit früheren Entscheidungen belohnen
            for i, option in enumerate(options):
                for past_decision in similar_decisions:
                    if past_decision["chosen_option"] == option:
                        # Stärkeren Boost für kürzlich getroffene Entscheidungen
                        recency = 1.0 - (self.current_time - past_decision["timestamp"]) / 20.0
                        recency = max(0.1, recency)
                        consistency_boost[i] += 0.2 * recency
        
        return consistency_boost
    
    def _calculate_recency_boost(self, options: List[str]) -> np.ndarray:
        """Berechnet einen Recency-Boost basierend auf kürzlich gewählten Optionen."""
        recency_boost = np.zeros(len(options))
        
        if not self.decision_history:
            return recency_boost
            
        # Letzte 5 Entscheidungen betrachten
        recent_decisions = self.decision_history[-5:]
        
        for i, option in enumerate(options):
            for decision in recent_decisions:
                if decision["chosen_option"] == option:
                    # Stärkerer Boost für kürzliche Entscheidungen
                    recency = 1.0 - (self.current_time - decision["timestamp"]) / 10.0
                    recency = max(0.1, recency)
                    recency_boost[i] += 0.15 * recency
        
        return recency_boost
    
    def _calculate_emotion_boost(self, scenario: 'EthicalScenario', options: List[str]) -> np.ndarray:
        """Berechnet einen emotionalen Boost basierend auf emotionalen Reaktionen."""
        emotion_boost = np.zeros(len(options))
        
        # Emotionale Valenz relevanter Überzeugungen sammeln
        emotional_reactions = {}
        for belief_name in scenario.relevant_beliefs:
            if belief_name in self.beliefs:
                emotional_reactions[belief_name] = self.beliefs[belief_name].emotional_valence
        
        if not emotional_reactions:
            return emotion_boost
            
        # Für jede Option emotionale Reaktion berechnen
        for i, option in enumerate(options):
            option_emotion = 0
            
            for belief_name, impact in scenario.options[option].items():
                if belief_name in emotional_reactions:
                    # Positive Auswirkung auf positiv bewertete Überzeugung = positiv
                    # Negative Auswirkung auf positiv bewertete Überzeugung = negativ
                    option_emotion += impact * emotional_reactions[belief_name]
            
            # Negativitätsverzerrung anwenden
            negativity_bias = self.cognitive_architecture.emotional_parameters.get("negativity_bias", 0.6)
            if option_emotion < 0:
                option_emotion *= (1.0 + negativity_bias)
                
            emotion_boost[i] = option_emotion * 0.3  # Skalierungsfaktor
        
        return emotion_boost
        
    def update_beliefs_from_experience(self, scenario: 'EthicalScenario', 
                                     chosen_option: str) -> Dict[str, float]:
        """
        Aktualisiert Überzeugungen basierend auf Erfahrungen aus einer Entscheidung.
        
        Returns:
            Dictionary mit Überzeugungsänderungen
        """
        belief_changes = {}
        
        # Feedback aus dem Szenario für die gewählte Option
        if chosen_option in scenario.outcome_feedback:
            for belief_name, feedback in scenario.outcome_feedback[chosen_option].items():
                if belief_name in self.beliefs:
                    belief = self.beliefs[belief_name]
                    old_strength = belief.strength
                    old_certainty = belief.certainty
                    
                    # Lernrate basierend auf kognitiver Architektur
                    base_learning_rate = 0.05
                    
                    if self.cognitive_architecture.primary_processing == NeuralProcessingType.SYSTEMATIC:
                        learning_rate = base_learning_rate * 1.2  # Systematische Denker lernen mehr aus Erfahrung
                    elif self.cognitive_architecture.primary_processing == NeuralProcessingType.ASSOCIATIVE:
                        learning_rate = base_learning_rate * 1.1  # Assoziative Denker lernen gut
                    else:
                        learning_rate = base_learning_rate
                    
                    # Offenheit verstärkt Lernbereitschaft
                    learning_rate *= (0.8 + 0.4 * self.personality_traits["openness"])
                    
                    # Stärke des Feedbacks berücksichtigen
                    feedback_strength = abs(feedback)
                    
                    # Bayes'sches Update für Überzeugungsstärke
                    new_strength = self.cognitive_architecture.apply_bayesian_update(
                        old_strength, feedback_strength, np.sign(feedback))
                    
                    # Gewissheit aktualisieren
                    certainty_change = 0.05  # Grundlegende Änderung
                    
                    # Konsistentes Feedback erhöht Gewissheit
                    if (feedback > 0 and old_strength > 0.5) or (feedback < 0 and old_strength < 0.5):
                        certainty_change = 0.05 * feedback_strength
                    # Inkonsistentes Feedback verringert Gewissheit
                    else:
                        certainty_change = -0.1 * feedback_strength
                        
                    new_certainty = np.clip(old_certainty + certainty_change, 0.1, 1.0)
                    
                    # Emotionale Valenz anpassen (falls Feedback stark)
                    if abs(feedback) > 0.3:
                        old_valence = belief.emotional_valence
                        valence_change = np.sign(feedback) * 0.1 * feedback_strength
                        new_valence = np.clip(old_valence + valence_change, -1.0, 1.0)
                    else:
                        new_valence = None  # Keine Änderung
                    
                    # Überzeugung aktualisieren
                    self.update_belief(belief_name, new_strength, new_certainty, new_valence)
                    belief_changes[belief_name] = new_strength - old_strength
        
        # Propagation der Änderungen durch das Netzwerk von Überzeugungen
        propagated_changes = self._propagate_belief_changes(belief_changes)
        belief_changes.update(propagated_changes)
        
        # Episodisches Gedächtnis aktualisieren (überraschende Änderungen speichern)
        for belief_name, change in belief_changes.items():
            if abs(change) > 0.1:  # Signifikante Änderung
                self.episodic_memory.append({
                    "time": self.current_time,
                    "type": "belief_change",
                    "belief": belief_name,
                    "change": change,
                    "scenario": scenario.scenario_id,
                    "option": chosen_option
                })
        
        return belief_changes
    
    def _propagate_belief_changes(self, initial_changes: Dict[str, float]) -> Dict[str, float]:
        """
        Verbreitet Änderungen in Überzeugungen durch das Netzwerk von Überzeugungen.
        
        Args:
            initial_changes: Dictionary mit initialen Änderungen (belief_name -> change)
            
        Returns:
            Dictionary mit zusätzlichen Änderungen
        """
        propagated_changes = {}
        
        # Propagationsstärke basierend auf kognitiver Architektur
        if self.cognitive_architecture.primary_processing == NeuralProcessingType.ASSOCIATIVE:
            propagation_strength = 0.6  # Stärkere Propagation bei assoziativem Denken
        elif self.cognitive_architecture.primary_processing == NeuralProcessingType.SYSTEMATIC:
            propagation_strength = 0.4  # Schwächere, aber gezieltere Propagation
        else:
            propagation_strength = 0.5
        
        # Ausbreitungslogik basierend auf Verarbeitungstyp
        for belief_name, change in initial_changes.items():
            if belief_name in self.beliefs:
                belief = self.beliefs[belief_name]
                
                # Änderungen an verbundene Überzeugungen weitergeben
                for connected_belief, (influence, polarity) in belief.connections.items():
                    if connected_belief in self.beliefs:
                        # Stärke der Änderung basierend auf Verbindungsstärke und Polarität
                        connected_change = change * influence * polarity * propagation_strength
                        
                        # Bei assoziativem Denken zusätzliche Ausbreitung über assoziierte Konzepte
                        if (self.cognitive_architecture.primary_processing == NeuralProcessingType.ASSOCIATIVE or
                            self.cognitive_architecture.secondary_processing == NeuralProcessingType.ASSOCIATIVE):
                            
                            # Assoziative Verstärkung
                            for concept in belief.associated_concepts:
                                if concept in self.beliefs[connected_belief].associated_concepts:
                                    assoc_strength = (belief.associated_concepts[concept] * 
                                                    self.beliefs[connected_belief].associated_concepts[concept])
                                    connected_change *= (1.0 + 0.3 * assoc_strength)
                        
                        # Aktualisieren der verbundenen Überzeugung
                        old_strength = self.beliefs[connected_belief].strength
                        new_strength = old_strength + connected_change
                        self.update_belief(connected_belief, new_strength)
                        
                        if connected_belief not in initial_changes:
                            propagated_changes[connected_belief] = new_strength - old_strength
        
        return propagated_changes
    
    def update_from_social_influence(self, other_agent: 'NeuralEthicalAgent', 
                                   influenced_beliefs: List[str] = None) -> Dict[str, float]:
        """
        Aktualisiert Überzeugungen basierend auf dem sozialen Einfluss eines anderen Agenten.
        
        Args:
            other_agent: Agent, der Einfluss ausübt
            influenced_beliefs: Liste spezifischer Überzeugungen, die beeinflusst werden sollen
                              (None für alle gemeinsamen Überzeugungen)
        
        Returns:
            Dictionary mit Überzeugungsänderungen
        """
        if other_agent.agent_id not in self.social_connections:
            return {}
        
        connection_strength = self.social_connections[other_agent.agent_id]
        belief_changes = {}
        
        # Soziale Lernrate basierend auf kognitiver Architektur
        base_social_learning_rate = 0.02
        
        if self.cognitive_architecture.primary_processing == NeuralProcessingType.NARRATIVE:
            # Narrative Denker sind empfänglicher für sozialen Einfluss
            social_factor = 1.3
        elif self.cognitive_architecture.primary_processing == NeuralProcessingType.SYSTEMATIC:
            # Systematische Denker sind weniger empfänglich
            social_factor = 0.7
        else:
            social_factor = 1.0
            
        # Persönlichkeitsfaktoren einbeziehen
        agreeableness_factor = 0.7 + 0.6 * self.personality_traits["agreeableness"]
        
        # Gesamte soziale Lernrate
        social_learning_rate = base_social_learning_rate * social_factor * agreeableness_factor * connection_strength
        
        # Autorität erhöht den Einfluss (wenn vorhanden)
        authority_bias = self.cognitive_architecture.cognitive_biases.get("authority_bias", 0.5)
        # Prüfen, ob der andere Agent als Autorität wahrgenommen wird
        perceived_authority = 0.0
        for group, strength in other_agent.group_identities.items():
            if group in ["Experts", "Leaders", "Teachers"] and strength > 0.6:
                perceived_authority = max(perceived_authority, strength)
        
        if perceived_authority > 0:
            social_learning_rate *= (1.0 + authority_bias * perceived_authority)
        
        # Überzeugungen identifizieren, die beeinflusst werden sollen
        if influenced_beliefs is None:
            common_beliefs = set(self.beliefs.keys()).intersection(set(other_agent.beliefs.keys()))
        else:
            common_beliefs = set(influenced_beliefs).intersection(
                set(self.beliefs.keys())).intersection(set(other_agent.beliefs.keys()))
        
        # Überzeugungen vergleichen und aktualisieren
        for belief_name in common_beliefs:
            # Stärke des Einflusses hängt von der Differenz der Überzeugungsstärken ab
            my_belief = self.beliefs[belief_name]
            other_belief = other_agent.beliefs[belief_name]
            
            my_strength = my_belief.strength
            other_strength = other_belief.strength
            strength_diff = other_strength - my_strength
            
            # Gewissheit beeinflusst die Überzeugungsänderung
            my_certainty = my_belief.certainty
            other_certainty = other_belief.certainty
            
            # Bei höherer eigener Gewissheit weniger beeinflussbar
            certainty_factor = 1.0 - 0.5 * my_certainty
            # Bei höherer Gewissheit des anderen mehr Einfluss
            other_certainty_factor = 0.5 + 0.5 * other_certainty
            
            # Gewichtung des Einflusses basierend auf Gruppenidentität
            group_weight = 1.0
            ingroup_bias = self.cognitive_architecture.cognitive_biases.get("ingroup_bias", 0.5)
            
            for group, my_identity in self.group_identities.items():
                if group in other_agent.group_identities:
                    other_identity = other_agent.group_identities[group]
                    # Stärkerer Einfluss bei gemeinsamer Gruppenidentität
                    if my_identity > 0.5 and other_identity > 0.5:
                        shared_identity = min(my_identity, other_identity)
                        group_weight *= (1.0 + ingroup_bias * shared_identity)
            
            # Aktualisierung basierend auf sozialem Einfluss
            change = (strength_diff * social_learning_rate * certainty_factor * 
                     other_certainty_factor * group_weight)
            
            # Kognitive Faktoren berücksichtigen
            if abs(strength_diff) > 0.3:  # Große Meinungsunterschiede
                # Dogmatischere Menschen ändern ihre Meinung weniger
                dogmatism = self.cognitive_architecture.cognitive_biases.get("dogmatism", 0.5)
                change *= (1.0 - 0.7 * dogmatism)
            
            # Nur signifikante Änderungen anwenden
            if abs(change) > 0.01:
                # Aktualisieren der Überzeugung
                new_strength = my_strength + change
                
                # Auch emotionale Valenz beeinflussen, wenn Änderung signifikant
                if abs(change) > 0.05:
                    my_valence = my_belief.emotional_valence
                    other_valence = other_belief.emotional_valence
                    valence_diff = other_valence - my_valence
                    valence_change = valence_diff * social_learning_rate * 0.5
                    new_valence = my_valence + valence_change
                else:
                    new_valence = None
                
                # Gewissheit leicht anpassen (Annäherung an den anderen)
                certainty_diff = other_certainty - my_certainty
                certainty_change = certainty_diff * social_learning_rate * 0.3
                new_certainty = my_certainty + certainty_change
                
                self.update_belief(belief_name, new_strength, new_certainty, new_valence)
                belief_changes[belief_name] = change
            
        # Episodisches Gedächtnis aktualisieren (signifikante soziale Einflüsse)
        if any(abs(change) > 0.1 for change in belief_changes.values()):
            self.episodic_memory.append({
                "time": self.current_time,
                "type": "social_influence",
                "from_agent": other_agent.agent_id,
                "significant_changes": {k: v for k, v in belief_changes.items() if abs(v) > 0.05}
            })
            
        return belief_changes
    
    def reflect_on_experiences(self) -> Dict[str, float]:
        """
        Reflektiert über Erfahrungen und konsolidiert Überzeugungen.
        Simuliert Nachdenken/Verarbeiten von Erfahrungen.
        
        Returns:
            Dictionary mit Überzeugungsänderungen
        """
        if not self.episodic_memory:
            return {}
        
        belief_changes = {}
        
        # Nur durch systematisches/analytisches Denken möglich
        if (self.cognitive_architecture.primary_processing != NeuralProcessingType.SYSTEMATIC and
            self.cognitive_architecture.secondary_processing != NeuralProcessingType.SYSTEMATIC):
            return {}
            
        # Stärke der Reflexion basierend auf kognitiven Parametern
        reflection_strength = 0.3
        if self.cognitive_architecture.primary_processing == NeuralProcessingType.SYSTEMATIC:
            reflection_strength = 0.5
            
        # Erhöhung durch Persönlichkeit
        reflection_strength *= (0.7 + 0.6 * self.personality_traits["openness"])
        
        # Nur neuere Erinnerungen betrachten (letzte 30% oder mindestens 5)
        memory_count = max(5, int(len(self.episodic_memory) * 0.3))
        recent_memories = sorted(self.episodic_memory, key=lambda x: x["time"], reverse=True)[:memory_count]
        
        # Überzeugungen sammeln, die häufig in Erinnerungen vorkommen
        belief_mentions = {}
        for memory in recent_memories:
            if memory["type"] == "belief_change":
                belief_name = memory["belief"]
                if belief_name not in belief_mentions:
                    belief_mentions[belief_name] = 0
                belief_mentions[belief_name] += 1
            elif memory["type"] == "social_influence" and "significant_changes" in memory:
                for belief_name in memory["significant_changes"]:
                    if belief_name not in belief_mentions:
                        belief_mentions[belief_name] = 0
                    belief_mentions[belief_name] += 1
        
        # Die häufigsten Überzeugungen konsolidieren
        most_common_beliefs = sorted(belief_mentions.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for belief_name, count in most_common_beliefs:
            if belief_name in self.beliefs:
                belief = self.beliefs[belief_name]
                
                # Betrachte Veränderungen dieser Überzeugung
                changes = []
                for memory in recent_memories:
                    if memory["type"] == "belief_change" and memory["belief"] == belief_name:
                        changes.append(memory["change"])
                    elif memory["type"] == "social_influence" and belief_name in memory.get("significant_changes", {}):
                        changes.append(memory["significant_changes"][belief_name])
                
                if not changes:
                    continue
                    
                # Konsistenz in der Veränderungsrichtung berechnen
                avg_change = sum(changes) / len(changes)
                consistency = abs(avg_change) / (sum(abs(c) for c in changes) / len(changes)) if changes else 0
                
                # Konsolidierung basierend auf Konsistenz
                if consistency > 0.6:  # Relativ konsistente Richtung
                    old_strength = belief.strength
                    
                    # Stärkere Konsolidierung bei höherer Konsistenz
                    consolidation = avg_change * reflection_strength * consistency * 0.3
                    
                    # Überzeugung anpassen
                    new_strength = np.clip(old_strength + consolidation, 0.0, 1.0)
                    
                    # Gewissheit erhöhen bei konsistenten Veränderungen
                    certainty_boost = 0.05 * consistency * reflection_strength
                    new_certainty = np.clip(belief.certainty + certainty_boost, 0.0, 1.0)
                    
                    self.update_belief(belief_name, new_strength, new_certainty)
                    belief_changes[belief_name] = new_strength - old_strength
        
        return belief_changes
        
    def __str__(self):
        """String-Repräsentation des Agenten."""
        return f"Agent {self.agent_id} mit {len(self.beliefs)} Überzeugungen, {self.cognitive_architecture}"


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


class NeuralEthicalSociety:
    """Repräsentiert eine Gesellschaft von ethischen Agenten mit neuronalen Verarbeitungsmodellen."""
    
    def __init__(self):
        """Initialisiert eine Gesellschaft von ethischen Agenten."""
        self.agents = {}  # agent_id -> NeuralEthicalAgent
        self.scenarios = {}  # scenario_id -> EthicalScenario
        self.belief_templates = {}  # Template-Überzeugungen für die Agentenerzeugung
        self.social_network = nx.Graph()  # Soziales Netzwerk
        self.groups = {}  # group_name -> set of agent_ids
        
        # Simulation Controls
        self.current_step = 0
        self.robustness_settings = {
            "validation_enabled": True,
            "error_checking": True,
            "ensemble_size": 3,  # Anzahl der Ensemble-Durchläufe
            "boundary_checks": True,
            "sensitivity_analysis": False,
            "resilience_to_outliers": True
        }
        
        # Validierungsmessungen
        self.validation_metrics = {
            "belief_distributions": {},
            "decision_patterns": {},
            "polarization_history": [],
            "validation_errors": []
        }
        
    def add_agent(self, agent: NeuralEthicalAgent):
        """Fügt einen Agenten zur Gesellschaft hinzu."""
        self.agents[agent.agent_id] = agent
        self.social_network.add_node(agent.agent_id)
        
    def add_scenario(self, scenario: EthicalScenario):
        """Fügt ein Szenario zur Gesellschaft hinzu."""
        self.scenarios[scenario.scenario_id] = scenario
        
    def add_belief_template(self, name: str, category: str, 
                           connections: Dict[str, Tuple[float, int]] = None,
                           associated_concepts: Dict[str, float] = None,
                           emotional_valence: float = 0.0):
        """
        Fügt eine Template-Überzeugung hinzu.
        
        Args:
            name: Name der Überzeugung
            category: Kategorie der Überzeugung
            connections: Verbindungen zu anderen Überzeugungen (name -> (strength, polarity))
            associated_concepts: Assoziierte Konzepte (name -> strength)
            emotional_valence: Emotionale Ladung (-1 bis +1)
        """
        self.belief_templates[name] = (category, connections or {}, 
                                     associated_concepts or {}, emotional_valence)
        
    def add_social_connection(self, agent1_id: str, agent2_id: str, strength: float):
        """Fügt eine soziale Verbindung zwischen zwei Agenten hinzu."""
        if agent1_id in self.agents and agent2_id in self.agents:
            self.agents[agent1_id].add_social_connection(agent2_id, strength)
            self.agents[agent2_id].add_social_connection(agent1_id, strength)
            self.social_network.add_edge(agent1_id, agent2_id, weight=strength)
            
    def add_group(self, group_name: str, agent_ids: List[str], 
                 min_identification: float = 0.5, max_identification: float = 1.0):
        """
        Fügt eine Gruppe von Agenten hinzu.
        
        Args:
            group_name: Name der Gruppe
            agent_ids: Liste von Agenten-IDs in der Gruppe
            min_identification: Minimale Identifikationsstärke mit der Gruppe
            max_identification: Maximale Identifikationsstärke mit der Gruppe
        """
        self.groups[group_name] = set(agent_ids)
        
        for agent_id in agent_ids:
            if agent_id in self.agents:
                identification = np.random.uniform(min_identification, max_identification)
                self.agents[agent_id].add_group_identity(group_name, identification)
                
    def generate_random_agent(self, agent_id: str, belief_distribution: str = "normal",
                             min_beliefs: int = 5, max_beliefs: int = 15,
                             connection_probability: float = 0.3) -> NeuralEthicalAgent:
        """
        Generiert einen zufälligen Agenten mit Überzeugungen aus den Templates.
        
        Args:
            agent_id: ID für den neuen Agenten
            belief_distribution: Art der Verteilung für Überzeugungsstärken
            min_beliefs: Minimale Anzahl an Überzeugungen
            max_beliefs: Maximale Anzahl an Überzeugungen
            connection_probability: Wahrscheinlichkeit für Verbindungen zwischen Überzeugungen
            
        Returns:
            Neuer zufälliger Agent
        """
        agent = NeuralEthicalAgent(agent_id)
        
        # Zufällige Anzahl an Überzeugungen auswählen
        num_beliefs = np.random.randint(min_beliefs, max_beliefs + 1)
        belief_names = list(self.belief_templates.keys())
        
        if num_beliefs > len(belief_names):
            num_beliefs = len(belief_names)
            
        selected_belief_names = np.random.choice(belief_names, 
                                               size=num_beliefs, 
                                               replace=False)
        
        # Überzeugungen hinzufügen
        for belief_name in selected_belief_names:
            category, connections, associated_concepts, emotional_valence = self.belief_templates[belief_name]
            
            # Anfangsstärke basierend auf der gewählten Verteilung
            if belief_distribution == "normal":
                initial_strength = np.clip(np.random.normal(0.5, 0.15), 0.1, 0.9)
            elif belief_distribution == "uniform":
                initial_strength = np.random.uniform(0.1, 0.9)
            elif belief_distribution == "beta":
                initial_strength = np.random.beta(2, 2)
            else:
                initial_strength = 0.5
                
            # Anfängliche Gewissheit
            initial_certainty = np.random.beta(4, 4)  # Mittlere Gewissheit mit Variation
            
            # Emotionale Valenz mit Variation
            valence_variation = np.random.normal(0, 0.2)
            adjusted_valence = np.clip(emotional_valence + valence_variation, -1.0, 1.0)
            
            # Überzeugung erstellen
            belief = NeuralEthicalBelief(belief_name, category, initial_strength, 
                                      initial_certainty, adjusted_valence)
            
            # Verbindungen hinzufügen (nur zu anderen ausgewählten Überzeugungen)
            for conn_name, (conn_strength, polarity) in connections.items():
                if conn_name in selected_belief_names:
                    # Variabilität in Verbindungsstärken
                    actual_strength = conn_strength * np.random.uniform(0.8, 1.2)
                    belief.add_connection(conn_name, actual_strength, polarity)
            
            # Assoziierte Konzepte hinzufügen
            for concept_name, association_strength in associated_concepts.items():
                # Variabilität in Assoziationsstärken
                actual_strength = association_strength * np.random.uniform(0.8, 1.2)
                belief.add_associated_concept(concept_name, actual_strength)
            
            # Zusätzliche zufällige Verbindungen
            for other_name in selected_belief_names:
                if (other_name != belief_name and 
                    other_name not in belief.connections and 
                    np.random.random() < connection_probability):
                    # Zufällige Verbindungsstärke und Polarität
                    rand_strength = np.random.uniform(0.1, 0.5)
                    rand_polarity = np.random.choice([-1, 1])
                    belief.add_connection(other_name, rand_strength, rand_polarity)
            
            agent.add_belief(belief)
            
        return agent
    
    def generate_similar_agent(self, base_agent: NeuralEthicalAgent, agent_id: str, 
                              similarity: float = 0.8) -> NeuralEthicalAgent:
        """
        Generiert einen Agenten, der dem Basis-Agenten ähnlich ist.
        
        Args:
            base_agent: Basis-Agent für die Ähnlichkeit
            agent_id: ID für den neuen Agenten
            similarity: Grad der Ähnlichkeit (0-1)
            
        Returns:
            Neuer ähnlicher Agent
        """
        # Neuen Agenten erstellen mit ähnlichen Persönlichkeitsmerkmalen
        new_agent = NeuralEthicalAgent(agent_id)
        
        # Persönlichkeitsmerkmale anpassen
        for trait, value in base_agent.personality_traits.items():
            # Wert in der Nähe des Basis-Agenten, aber mit etwas Variation
            variation = np.random.normal(0, (1 - similarity) * 0.2)
            new_value = np.clip(value + variation, 0.0, 1.0)
            new_agent.personality_traits[trait] = new_value
        
        # Kognitive Architektur mit Ähnlichkeit generieren
        if np.random.random() < similarity:
            # Ähnliche kognitive Architektur
            primary = base_agent.cognitive_architecture.primary_processing
            secondary = base_agent.cognitive_architecture.secondary_processing
            balance = base_agent.cognitive_architecture.processing_balance
            
            # Leichte Variation im Balance
            balance_variation = np.random.normal(0, (1 - similarity) * 0.2)
            new_balance = np.clip(balance + balance_variation, 0.0, 1.0)
            
            new_agent.cognitive_architecture = CognitiveArchitecture(primary, secondary, new_balance)
            
            # Kognitive Biases ähnlich anpassen
            for bias, value in base_agent.cognitive_architecture.cognitive_biases.items():
                variation = np.random.normal(0, (1 - similarity) * 0.2)
                new_value = np.clip(value + variation, 0.0, 1.0)
                new_agent.cognitive_architecture.cognitive_biases[bias] = new_value
                
            # Emotionale Parameter ähnlich anpassen
            for param, value in base_agent.cognitive_architecture.emotional_parameters.items():
                variation = np.random.normal(0, (1 - similarity) * 0.2)
                new_value = np.clip(value + variation, 0.0, 1.0)
                new_agent.cognitive_architecture.emotional_parameters[param] = new_value
        else:
            # Gänzlich neue kognitive Architektur
            new_agent.cognitive_architecture = new_agent._generate_cognitive_architecture()
            
        # Moralische Grundlagen anpassen
        for foundation, value in base_agent.moral_foundations.items():
            variation = np.random.normal(0, (1 - similarity) * 0.3)
            new_value = np.clip(value + variation, 0.0, 1.0)
            new_agent.moral_foundations[foundation] = new_value
            
        # Überzeugungen kopieren und variieren
        for belief_name, belief in base_agent.beliefs.items():
            # Mit zunehmender Unähnlichkeit steigt die Wahrscheinlichkeit, eine Überzeugung auszulassen
            if np.random.random() < similarity:
                # Variieren der Überzeugungsstärke
                strength_variation = np.random.normal(0, (1 - similarity) * 0.3)
                new_strength = np.clip(belief.strength + strength_variation, 0.0, 1.0)
                
                # Variieren der Gewissheit
                certainty_variation = np.random.normal(0, (1 - similarity) * 0.3)
                new_certainty = np.clip(belief.certainty + certainty_variation, 0.0, 1.0)
                
                # Variieren der emotionalen Valenz
                valence_variation = np.random.normal(0, (1 - similarity) * 0.3)
                new_valence = np.clip(belief.emotional_valence + valence_variation, -1.0, 1.0)
                
                # Neue Überzeugung erstellen
                new_belief = NeuralEthicalBelief(belief.name, belief.category, 
                                              new_strength, new_certainty, new_valence)
                
                # Verbindungen kopieren und variieren
                for conn_name, (conn_strength, polarity) in belief.connections.items():
                    # Mit geringer Wahrscheinlichkeit Polarität umkehren
                    if np.random.random() < 0.1 * (1 - similarity):
                        polarity *= -1
                        
                    # Verbindungsstärke variieren
                    conn_variation = np.random.normal(0, (1 - similarity) * 0.2)
                    new_conn_strength = np.clip(conn_strength + conn_variation, 0.0, 1.0)
                    
                    new_belief.add_connection(conn_name, new_conn_strength, polarity)
                
                # Assoziierte Konzepte kopieren und variieren
                for concept, assoc_strength in belief.associated_concepts.items():
                    assoc_variation = np.random.normal(0, (1 - similarity) * 0.2)
                    new_assoc_strength = np.clip(assoc_strength + assoc_variation, 0.0, 1.0)
                    new_belief.add_associated_concept(concept, new_assoc_strength)
                
                new_agent.add_belief(new_belief)
                
        # Gruppenzugehörigkeiten mit Variation kopieren
        for group, identification in base_agent.group_identities.items():
            # Variieren der Identifikationsstärke
            id_variation = np.random.normal(0, (1 - similarity) * 0.3)
            new_identification = np.clip(identification + id_variation, 0.0, 1.0)
            new_agent.add_group_identity(group, new_identification)
            
        return new_agent
        
    def generate_diverse_society(self, num_archetypes: int = 4, 
                               agents_per_archetype: int = 5,
                               similarity_range: Tuple[float, float] = (0.5, 0.9),
                               randomize_cognitive_styles: bool = True) -> List[NeuralEthicalAgent]:
        """
        Generiert eine diverse Gesellschaft mit verschiedenen Denkstilen.
        
        Args:
            num_archetypes: Anzahl unterschiedlicher Archetypen
            agents_per_archetype: Anzahl Agenten pro Archetyp
            similarity_range: Bereich der Ähnlichkeit zu Archetypen
            randomize_cognitive_styles: Ob Denkstile zufällig verteilt werden sollen
            
        Returns:
            Liste der generierten Agenten
        """
        # Archetypen generieren
        archetypes = []
        
        # Falls kognitive Stile gezielt verteilt werden sollen
        cognitive_styles = [
            (NeuralProcessingType.SYSTEMATIC, NeuralProcessingType.ANALOGICAL),
            (NeuralProcessingType.INTUITIVE, NeuralProcessingType.ASSOCIATIVE),
            (NeuralProcessingType.EMOTIONAL, NeuralProcessingType.SYSTEMATIC),
            (NeuralProcessingType.NARRATIVE, NeuralProcessingType.INTUITIVE),
            (NeuralProcessingType.ASSOCIATIVE, NeuralProcessingType.EMOTIONAL),
            (NeuralProcessingType.ANALOGICAL, NeuralProcessingType.NARRATIVE)
        ]
        
        # Archetypen erstellen
        for i in range(num_archetypes):
            archetype_id = f"archetype_{i+1}"
            archetype = self.generate_random_agent(archetype_id)
            
            # Optional: Gezielt unterschiedliche kognitive Stile zuweisen
            if not randomize_cognitive_styles and i < len(cognitive_styles):
                primary, secondary = cognitive_styles[i]
                balance = np.random.uniform(0.6, 0.8)
                archetype.cognitive_architecture = CognitiveArchitecture(primary, secondary, balance)
            
            archetypes.append(archetype)
            self.add_agent(archetype)
            
            # Archetyp als eigene Gruppe definieren
            self.add_group(f"Group_{i+1}", [archetype_id])
        
        # Weitere Agenten basierend auf Archetypen generieren
        new_agents = []
        agent_counter = num_archetypes
        
        for archetype in archetypes:
            for j in range(agents_per_archetype):
                agent_id = f"agent_{agent_counter + j + 1}"
                
                # Zufällige Ähnlichkeit im angegebenen Bereich
                similarity = np.random.uniform(similarity_range[0], similarity_range[1])
                
                # Neuen Agenten erstellen
                new_agent = self.generate_similar_agent(archetype, agent_id, similarity)
                new_agents.append(new_agent)
                self.add_agent(new_agent)
                
                # Zur Gruppe des Archetyps hinzufügen
                group_id = f"Group_{archetypes.index(archetype) + 1}"
                self.add_group(group_id, [agent_id])
                
            agent_counter += agents_per_archetype
            
        # Soziales Netzwerk generieren
        self.generate_realistic_social_network(
            connection_density=0.1,
            group_connection_boost=0.4,
            belief_similarity_factor=0.6,
            cognitive_style_influence=0.3
        )
        
        return archetypes + new_agents
    
    def generate_realistic_social_network(self, connection_density: float = 0.1,
                                        group_connection_boost: float = 0.3,
                                        belief_similarity_factor: float = 0.5,
                                        cognitive_style_influence: float = 0.2):
        """
        Generiert ein realistisches soziales Netzwerk basierend auf Gruppen, Überzeugungsähnlichkeit
        und kognitiven Stilen.
        
        Args:
            connection_density: Grundlegende Dichte von Verbindungen
            group_connection_boost: Erhöhte Verbindungswahrscheinlichkeit innerhalb von Gruppen
            belief_similarity_factor: Einfluss der Überzeugungsähnlichkeit auf Verbindungen
            cognitive_style_influence: Einfluss des kognitiven Stils auf Verbindungen
        """
        agent_ids = list(self.agents.keys())
        
        # Berechnen von Ähnlichkeiten zwischen Agenten
        similarity_matrix = {}
        for i, agent1_id in enumerate(agent_ids):
            for agent2_id in agent_ids[i+1:]:
                agent1 = self.agents[agent1_id]
                agent2 = self.agents[agent2_id]
                
                # Überzeugungsähnlichkeit berechnen
                belief_sim = self._calculate_belief_similarity(agent1, agent2)
                
                # Gruppenzugehörigkeitsähnlichkeit berechnen
                group_sim = self._calculate_group_similarity(agent1, agent2)
                
                # Kognitive Stil-Ähnlichkeit berechnen
                cognitive_sim = self._calculate_cognitive_style_similarity(agent1, agent2)
                
                # Gewichtete Gesamtähnlichkeit
                total_sim = (belief_similarity_factor * belief_sim + 
                           (1 - belief_similarity_factor - cognitive_style_influence) * group_sim +
                           cognitive_style_influence * cognitive_sim)
                           
                similarity_matrix[(agent1_id, agent2_id)] = total_sim
        
        # Verbindungen basierend auf Ähnlichkeit und Zufall erstellen
        for (agent1_id, agent2_id), similarity in similarity_matrix.items():
            # Grundwahrscheinlichkeit für eine Verbindung
            base_prob = connection_density
            
            # Wahrscheinlichkeit basierend auf Ähnlichkeit erhöhen
            prob = base_prob + similarity * 0.5
            
            # Prüfen, ob Agenten gemeinsame Gruppen haben
            agent1 = self.agents[agent1_id]
            agent2 = self.agents[agent2_id]
            common_groups = set(agent1.group_identities.keys()) & set(agent2.group_identities.keys())
            
            # Wahrscheinlichkeit für Agenten in gleichen Gruppen erhöhen
            for group in common_groups:
                id1 = agent1.group_identities[group]
                id2 = agent2.group_identities[group]
                if id1 > 0.5 and id2 > 0.5:
                    prob += group_connection_boost * min(id1, id2)
            
            # Persönlichkeitsfaktoren berücksichtigen
            # Extrovertierte Menschen haben mehr Verbindungen
            extroversion_factor = 0.5 * (agent1.personality_traits["extroversion"] + 
                                       agent2.personality_traits["extroversion"])
            prob += extroversion_factor * 0.2
            
            # Kognitive Stile berücksichtigen
            # Narrative und emotionale Denker bevorzugen mehr soziale Verbindungen
            # Kognitive Stile berücksichtigen
            # Narrative und emotionale Denker bevorzugen mehr soziale Verbindungen
            narrative_factor = 0
            for agent in [agent1, agent2]:
                if agent.cognitive_architecture.primary_processing == NeuralProcessingType.NARRATIVE:
                    narrative_factor += 0.2
                elif agent.cognitive_architecture.primary_processing == NeuralProcessingType.EMOTIONAL:
                    narrative_factor += 0.15
                
            # Erhöhung der Verbindungswahrscheinlichkeit für sozial orientierte Denkstile
            prob += narrative_factor
            
            # Verbindung mit berechneter Wahrscheinlichkeit erstellen
            if np.random.random() < min(prob, 0.95):  # Max 95% Wahrscheinlichkeit
                # Verbindungsstärke basierend auf Ähnlichkeit
                strength = 0.3 + 0.7 * similarity
                self.add_social_connection(agent1_id, agent2_id, strength)



    def _calculate_cognitive_style_similarity(self, agent1: NeuralEthicalAgent, 
                                           agent2: NeuralEthicalAgent) -> float:
        """Berechnet die Ähnlichkeit der kognitiven Stile zwischen zwei Agenten."""
        # Gleicher primärer Verarbeitungsstil = hohe Ähnlichkeit
        if agent1.cognitive_architecture.primary_processing == agent2.cognitive_architecture.primary_processing:
            primary_similarity = 1.0
        else:
            primary_similarity = 0.2
        
        # Gleicher sekundärer Verarbeitungsstil = moderate Ähnlichkeit
        if agent1.cognitive_architecture.secondary_processing == agent2.cognitive_architecture.secondary_processing:
            secondary_similarity = 0.7
        else:
            secondary_similarity = 0.1
            
        # Ähnlichkeit in der Balance zwischen primär und sekundär
        balance_diff = abs(agent1.cognitive_architecture.processing_balance - 
                          agent2.cognitive_architecture.processing_balance)
        balance_similarity = 1.0 - balance_diff
        
        # Kognitive Biases vergleichen
        bias_similarity = self._compare_parameter_dicts(
            agent1.cognitive_architecture.cognitive_biases,
            agent2.cognitive_architecture.cognitive_biases)
        
        # Emotionale Parameter vergleichen
        emotional_similarity = self._compare_parameter_dicts(
            agent1.cognitive_architecture.emotional_parameters,
            agent2.cognitive_architecture.emotional_parameters)
        
        # Gewichtete Kombination
        return (0.3 * primary_similarity + 
               0.2 * secondary_similarity + 
               0.2 * balance_similarity + 
               0.15 * bias_similarity + 
               0.15 * emotional_similarity)
    
    def _compare_parameter_dicts(self, dict1: Dict[str, float], dict2: Dict[str, float]) -> float:
        """Vergleicht zwei Parameterdictionaries und gibt eine Ähnlichkeit zurück."""
        common_keys = set(dict1.keys()) & set(dict2.keys())
        
        if not common_keys:
            return 0.0
        
        total_diff = sum(abs(dict1[key] - dict2[key]) for key in common_keys)
        avg_diff = total_diff / len(common_keys)
        
        return 1.0 - avg_diff
    
    def _calculate_belief_similarity(self, agent1: NeuralEthicalAgent, 
                                   agent2: NeuralEthicalAgent) -> float:
        """Berechnet die Ähnlichkeit der Überzeugungen zwischen zwei Agenten."""
        # Gemeinsame Überzeugungen finden
        common_beliefs = set(agent1.beliefs.keys()) & set(agent2.beliefs.keys())
        
        if not common_beliefs:
            return 0.0
        
        # Ähnlichkeit basierend auf Überzeugungsstärken und Gewissheit
        similarity = 0.0
        for belief_name in common_beliefs:
            belief1 = agent1.beliefs[belief_name]
            belief2 = agent2.beliefs[belief_name]
            
            # Stärkeähnlichkeit (0-1)
            strength_sim = 1.0 - abs(belief1.strength - belief2.strength)
            
            # Gewissheitsähnlichkeit (0-1)
            certainty_sim = 1.0 - abs(belief1.certainty - belief2.certainty)
            
            # Valenzähnlichkeit (0-1)
            valence_sim = 1.0 - abs(belief1.emotional_valence - belief2.emotional_valence) / 2.0
            
            # Gewichtete Kombination der Ähnlichkeiten
            combined_sim = 0.6 * strength_sim + 0.2 * certainty_sim + 0.2 * valence_sim
            similarity += combined_sim
            
        return similarity / len(common_beliefs)
    
    def _calculate_group_similarity(self, agent1: NeuralEthicalAgent, 
                                  agent2: NeuralEthicalAgent) -> float:
        """Berechnet die Ähnlichkeit der Gruppenzugehörigkeit zwischen zwei Agenten."""
        # Alle Gruppen
        all_groups = set(agent1.group_identities.keys()) | set(agent2.group_identities.keys())
        
        if not all_groups:
            return 0.0
        
        # Ähnlichkeit basierend auf Gruppenzugehörigkeit
        similarity = 0.0
        for group in all_groups:
            id1 = agent1.group_identities.get(group, 0.0)
            id2 = agent2.group_identities.get(group, 0.0)
            # Je ähnlicher die Identifikation, desto ähnlicher
            similarity += 1.0 - abs(id1 - id2)
            
        return similarity / len(all_groups)
    def run_robust_simulation(self, num_steps: int, 
                            scenario_probability: float = 0.2,
                            social_influence_probability: float = 0.3,
                            reflection_probability: float = 0.1) -> Dict:
        """
        Führt eine robuste Simulation über mehrere Zeitschritte aus mit Fehlerprüfung
        und Validierung.
        
        Args:
            num_steps: Anzahl der Simulationsschritte
            scenario_probability: Wahrscheinlichkeit für ein Szenario pro Agent und Schritt
            social_influence_probability: Wahrscheinlichkeit für sozialen Einfluss
            reflection_probability: Wahrscheinlichkeit für Reflexion über Erfahrungen
            
        Returns:
            Dictionary mit Simulationsergebnissen
        """
        # Initialisierung der Ergebnisstruktur
        results = {
            "decisions": [],
            "belief_changes": [],
            "social_influences": [],
            "reflections": [],
            "validation": {
                "errors": [],
                "warnings": [],
                "agent_consistency": {},
                "simulation_stability": {}
            }
        }
        
        # Ensemble-Durchläufe initialisieren (für Robustheit)
        ensemble_results = []
        
        for ensemble_run in range(self.robustness_settings["ensemble_size"]):
            # Temporäres Ergebnis für diesen Ensemble-Durchlauf
            ensemble_result = {
                "decisions": [],
                "belief_changes": [],
                "social_influences": [],
                "reflections": []
            }
            
            # Für jeden Zeitschritt
            for step in range(num_steps):
                self.current_step = step
                
                step_results = {
                    "step": step,
                    "decisions": {},
                    "belief_changes": {},
                    "social_influences": {},
                    "reflections": {}
                }
                
                # Für jeden Agenten
                for agent_id, agent in self.agents.items():
                    try:
                        # 1. Reflexion über Erfahrungen (zufällig)
                        if np.random.random() < reflection_probability:
                            reflection_changes = agent.reflect_on_experiences()
                            if reflection_changes:
                                step_results["reflections"][agent_id] = reflection_changes
                        
                        # 2. Mögliches Szenario erleben
                        if np.random.random() < scenario_probability and self.scenarios:
                            # Zufälliges Szenario auswählen
                            scenario_id = np.random.choice(list(self.scenarios.keys()))
                            scenario = self.scenarios[scenario_id]
                            
                            # Fehlerbehandlung für robuste Ausführung
                            try:
                                # Entscheidung treffen
                                decision = agent.make_decision(scenario)
                                step_results["decisions"][agent_id] = decision
                                
                                # Überzeugungen basierend auf Erfahrung aktualisieren
                                belief_changes = agent.update_beliefs_from_experience(
                                    scenario, decision["chosen_option"])
                                
                                # Validierung der Belief-Änderungen
                                if self.robustness_settings["validation_enabled"]:
                                    self._validate_belief_changes(agent_id, belief_changes)
                                    
                                step_results["belief_changes"][agent_id] = belief_changes
                                
                            except Exception as e:
                                # Fehler protokollieren
                                error_msg = f"Fehler bei Agent {agent_id} in Szenario {scenario_id}: {str(e)}"
                                results["validation"]["errors"].append(error_msg)
                                
                                # Robuste Ausführung fortsetzen
                                if self.robustness_settings["resilience_to_outliers"]:
                                    # Standardentscheidung generieren (erste Option wählen)
                                    fallback_option = list(scenario.options.keys())[0]
                                    step_results["decisions"][agent_id] = {
                                        "scenario_id": scenario_id,
                                        "chosen_option": fallback_option,
                                        "error_recovery": True
                                    }
                        
                        # 3. Möglicher sozialer Einfluss
                        if np.random.random() < social_influence_probability and agent.social_connections:
                            # Zufälligen verbundenen Agenten auswählen
                            connected_id = np.random.choice(list(agent.social_connections.keys()))
                            connected_agent = self.agents[connected_id]
                            
                            # Fehlerbehandlung
                            try:
                                # Überzeugungen basierend auf sozialem Einfluss aktualisieren
                                social_changes = agent.update_from_social_influence(connected_agent)
                                
                                if social_changes:
                                    if agent_id not in step_results["social_influences"]:
                                        step_results["social_influences"][agent_id] = {}
                                    step_results["social_influences"][agent_id][connected_id] = social_changes
                                    
                            except Exception as e:
                                # Fehler protokollieren
                                error_msg = f"Fehler bei sozialem Einfluss von {connected_id} auf {agent_id}: {str(e)}"
                                results["validation"]["errors"].append(error_msg)
                    
                    except Exception as e:
                        # Allgemeine Fehlerbehandlung für den Agenten
                        error_msg = f"Allgemeiner Fehler bei Agent {agent_id}: {str(e)}"
                        results["validation"]["errors"].append(error_msg)
                
                # Ergebnisse für diesen Schritt speichern
                ensemble_result["decisions"].append(step_results["decisions"])
                ensemble_result["belief_changes"].append(step_results["belief_changes"])
                ensemble_result["social_influences"].append(step_results["social_influences"])
                ensemble_result["reflections"].append(step_results["reflections"])
                
                # Zwischenvalidierung nach bestimmten Intervallen
                if step % 10 == 0 and self.robustness_settings["validation_enabled"]:
                    self._validate_simulation_state()
            
            # Ensemble-Ergebnis speichern
            ensemble_results.append(ensemble_result)
        
        # Ensemble-Ergebnisse kombinieren (für Robustheit)
        if self.robustness_settings["ensemble_size"] > 1:
            results = self._combine_ensemble_results(ensemble_results)
            # Ensure validation key exists after combining results
            if "validation" not in results:
                results["validation"] = {
                    "errors": [],
                    "warnings": [],
                    "agent_consistency": {},
                    "simulation_stability": {}
                }
        else:
            # Bei nur einem Durchlauf direkt übernehmen
            results["decisions"] = ensemble_results[0]["decisions"]
            results["belief_changes"] = ensemble_results[0]["belief_changes"]
            results["social_influences"] = ensemble_results[0]["social_influences"]
            results["reflections"] = ensemble_results[0]["reflections"]
        
        # Abschließende Validierung
        if self.robustness_settings["validation_enabled"]:
            self._final_validation(results)
            
        return results    

    
    def _validate_belief_changes(self, agent_id: str, belief_changes: Dict[str, float]):
        """Validiert Überzeugungsänderungen auf Plausibilität."""
        agent = self.agents[agent_id]
        
        for belief_name, change in belief_changes.items():
            # Überprüfen, ob Änderungen im plausiblen Bereich liegen
            if abs(change) > 0.3:
                warning = f"Große Überzeugungsänderung bei Agent {agent_id}, Belief '{belief_name}': {change}"
                self.validation_metrics["validation_errors"].append({"type": "warning", "message": warning})
                
            # Überprüfen der neuen Stärke
            if belief_name in agent.beliefs:
                strength = agent.beliefs[belief_name].strength
                if strength < 0.0 or strength > 1.0:
                    error = f"Ungültige Belief-Stärke bei Agent {agent_id}, Belief '{belief_name}': {strength}"
                    self.validation_metrics["validation_errors"].append({"type": "error", "message": error})
    
    def _validate_simulation_state(self):
        """Validiert den Gesamtzustand der Simulation."""
        # Überprüfen auf konsistente Zustände der Agenten
        for agent_id, agent in self.agents.items():
            # Überprüfen auf NaN-Werte in Überzeugungen
            for belief_name, belief in agent.beliefs.items():
                if np.isnan(belief.strength) or np.isnan(belief.certainty):
                    error = f"NaN-Wert in Überzeugung bei Agent {agent_id}, Belief '{belief_name}'"
                    self.validation_metrics["validation_errors"].append({"type": "error", "message": error})
                    
                    # Korrektur anwenden
                    if self.robustness_settings["error_checking"]:
                        if np.isnan(belief.strength):
                            belief.strength = 0.5  # Standardwert
                        if np.isnan(belief.certainty):
                            belief.certainty = 0.5  # Standardwert
            
            # Überprüfen der kognitiven Dissonanz auf Plausibilität
            dissonance = agent.calculate_cognitive_dissonance()
            if dissonance > 1.0:
                warning = f"Hohe kognitive Dissonanz bei Agent {agent_id}: {dissonance}"
                self.validation_metrics["validation_errors"].append({"type": "warning", "message": warning})
    
    def _combine_ensemble_results(self, ensemble_results: List[Dict]) -> Dict:
        """
        Kombiniert die Ergebnisse mehrerer Ensemble-Durchläufe für robustere Schätzungen.
        Verwendet Medianwerte statt Mittelwerte, um Ausreißer zu minimieren.
        """
        combined_results = {
            "decisions": [],
            "belief_changes": [],
            "social_influences": [],
            "reflections": [],
            "ensemble_statistics": {
                "variance": {},
                "confidence_intervals": {}
            }
        }
        
        # Anzahl der Zeitschritte ermitteln
        num_steps = len(ensemble_results[0]["decisions"])
        
        # Für jeden Zeitschritt
        for step in range(num_steps):
            # Initialisierung der kombinierten Ergebnisse für diesen Schritt
            step_decisions = {}
            step_belief_changes = {}
            step_social_influences = {}
            step_reflections = {}
            
            # Alle Agenten
            all_agent_ids = set()
            for ensemble in ensemble_results:
                all_agent_ids.update(ensemble["decisions"][step].keys())
                all_agent_ids.update(ensemble["belief_changes"][step].keys())
                all_agent_ids.update(ensemble["social_influences"][step].keys())
                all_agent_ids.update(ensemble["reflections"][step].keys())
            
            # Für jeden Agenten
            for agent_id in all_agent_ids:
                # Entscheidungen sammeln
                agent_decisions = [
                    ensemble["decisions"][step].get(agent_id, {"chosen_option": None})
                    for ensemble in ensemble_results
                ]
                
                # Robuste Entscheidungsauswahl (Mehrheitsentscheidung)
                if agent_decisions and any(d.get("chosen_option") for d in agent_decisions):
                    valid_decisions = [d for d in agent_decisions if d.get("chosen_option")]
                    
                    if valid_decisions:
                        # Häufigste Entscheidung auswählen
                        options = [d["chosen_option"] for d in valid_decisions]
                        option_counts = {}
                        for option in options:
                            if option not in option_counts:
                                option_counts[option] = 0
                            option_counts[option] += 1
                        
                        # Am häufigsten gewählte Option
                        chosen_option = max(option_counts.items(), key=lambda x: x[1])[0]
                        
                        # Repräsentative Entscheidung aus dem Ensemble auswählen
                        representative_decision = next(
                            (d for d in valid_decisions if d["chosen_option"] == chosen_option), 
                            valid_decisions[0]
                        )
                        
                        step_decisions[agent_id] = representative_decision
                
                # Belief-Änderungen kombinieren (Median pro Überzeugung)
                agent_belief_changes = {}
                
                # Alle relevanten Überzeugungen sammeln
                all_beliefs = set()
                for ensemble in ensemble_results:
                    if agent_id in ensemble["belief_changes"][step]:
                        all_beliefs.update(ensemble["belief_changes"][step][agent_id].keys())
                
                # Kombinierte Änderungen berechnen
                for belief_name in all_beliefs:
                    # Sammle alle Änderungswerte aus den Ensemble-Durchläufen
                    change_values = []
                    for ensemble in ensemble_results:
                        if (agent_id in ensemble["belief_changes"][step] and 
                            belief_name in ensemble["belief_changes"][step][agent_id]):
                            change_values.append(ensemble["belief_changes"][step][agent_id][belief_name])
                    
                    if change_values:
                        # Verwende den Median für robuste Schätzung
                        median_change = np.median(change_values)
                        agent_belief_changes[belief_name] = median_change
                        
                        # Varianz für Konfidenzschätzung
                        variance = np.var(change_values)
                        belief_key = f"{agent_id}_{belief_name}"
                        if belief_key not in combined_results["ensemble_statistics"]["variance"]:
                            combined_results["ensemble_statistics"]["variance"][belief_key] = []
                        combined_results["ensemble_statistics"]["variance"][belief_key].append(variance)
                
                if agent_belief_changes:
                    step_belief_changes[agent_id] = agent_belief_changes
                
                # Ähnlich für soziale Einflüsse und Reflexionen...
                # (Implementierungsdetails ausgelassen der Kürze halber)
            
            # Ergebnisse für diesen Schritt speichern
            combined_results["decisions"].append(step_decisions)
            combined_results["belief_changes"].append(step_belief_changes)
            combined_results["social_influences"].append(step_social_influences)
            combined_results["reflections"].append(step_reflections)
        
        return combined_results
    
    def _final_validation(self, results: Dict):
        """Führt eine abschließende Validierung der Simulationsergebnisse durch."""
        # Überprüfen der Gesamtkonsistenz
        
        # 1. Überprüfen, ob Überzeugungen im gültigen Bereich (0-1) liegen
        for agent_id, agent in self.agents.items():
            for belief_name, belief in agent.beliefs.items():
                if belief.strength < 0.0 or belief.strength > 1.0:
                    error = f"Ungültige finale Belief-Stärke bei Agent {agent_id}, Belief '{belief_name}': {belief.strength}"
                    results["validation"]["errors"].append(error)
                    
                    # Korrektur anwenden
                    belief.strength = np.clip(belief.strength, 0.0, 1.0)
        
        # 2. Überprüfen auf extreme Polarisierung
        polarization = self._calculate_final_polarization()
        for belief_name, metrics in polarization.items():
            if metrics["bimodality"] > 0.8:
                warning = f"Extreme Polarisierung bei Überzeugung '{belief_name}': {metrics['bimodality']:.2f}"
                results["validation"]["warnings"].append(warning)
        
        # 3. Überprüfen der Simulationsstabilität
        if len(results["validation"]["errors"]) > 0:
            results["validation"]["simulation_stability"] = "Unstable"
        elif len(results["validation"]["warnings"]) > 5:
            results["validation"]["simulation_stability"] = "Questionable"
        else:
            results["validation"]["simulation_stability"] = "Stable"
    
    def _calculate_final_polarization(self) -> Dict[str, Dict[str, float]]:
        """Berechnet die finale Polarisierung für alle Überzeugungen."""
        polarization = {}
        
        # Für jede Überzeugung
        all_beliefs = set()
        for agent in self.agents.values():
            all_beliefs.update(agent.beliefs.keys())
            
        for belief_name in all_beliefs:
            # Sammle alle Stärken dieser Überzeugung
            belief_strengths = []
            for agent in self.agents.values():
                if belief_name in agent.beliefs:
                    belief_strengths.append(agent.beliefs[belief_name].strength)
            
            if belief_strengths:
                # Bimodalität als Polarisierungsmaß
                hist, _ = np.histogram(belief_strengths, bins=10, range=(0, 1))
                bimodality = self._calculate_bimodality(hist)
                
                # Varianz als Maß für Meinungsvielfalt
                variance = np.var(belief_strengths)
                
                # Entropie als Maß für Ungleichverteilung
                hist_norm = hist / np.sum(hist) if np.sum(hist) > 0 else hist
                hist_norm = hist_norm[hist_norm > 0]  # Nur positive Werte für die Entropie
                entropy_value = entropy(hist_norm) if len(hist_norm) > 0 else 0
                
                polarization[belief_name] = {
                    "bimodality": bimodality,
                    "variance": variance,
                    "entropy": entropy_value
                }
                
        return polarization
    
    def _calculate_bimodality(self, histogram: np.ndarray) -> float:
        """Berechnet einen Bimodalitätsindex für ein Histogramm."""
        if np.sum(histogram) == 0:
            return 0.0
            
        # Normalisieren
        hist_norm = histogram / np.sum(histogram)
        
        # Mittelwert und Varianz berechnen
        mean = np.sum(np.arange(len(hist_norm)) * hist_norm)
        variance = np.sum((np.arange(len(hist_norm)) - mean) ** 2 * hist_norm)
        
        if variance == 0:
            return 0.0
        
        # Schiefe und Kurtosis berechnen
        skewness = np.sum((np.arange(len(hist_norm)) - mean) ** 3 * hist_norm) / (variance ** 1.5)
        kurtosis = np.sum((np.arange(len(hist_norm)) - mean) ** 4 * hist_norm) / (variance ** 2)
        
        # Bimodalitätskoeffizient nach SAS: (skewness^2 + 1) / kurtosis
        # Werte > 0.555 deuten auf Bimodalität hin
        if kurtosis > 0:
            bimodality = (skewness**2 + 1) / kurtosis
        else:
            bimodality = 1.0  # Maximalwert bei Nullvarianz
            
        return bimodality
    
    def run_sensitivity_analysis(self, parameter_ranges: Dict[str, Tuple[float, float]], 
                              num_samples: int = 10, metrics: List[str] = None) -> Dict:
        """
        Führt eine Sensitivitätsanalyse durch, um den Einfluss verschiedener Parameter 
        auf die Simulationsergebnisse zu quantifizieren.
        
        Args:
            parameter_ranges: Dictionary mit Parameternamen und ihren Wertebereichen
            num_samples: Anzahl der Stichproben pro Parameter
            metrics: Liste der zu trackenden Metriken
            
        Returns:
            Dictionary mit Sensitivitätsergebnissen
        """
        if not self.robustness_settings["sensitivity_analysis"]:
            return {"error": "Sensitivity analysis is disabled in robustness settings"}
        
        # Standardmetriken, falls keine angegeben
        if metrics is None:
            metrics = ["polarization", "decision_consensus", "belief_change_magnitude"]
        
        results = {
            "parameters": {},
            "metrics": {metric: {} for metric in metrics}
        }
        
        # Parameter separat variieren
        for param_name, (min_val, max_val) in parameter_ranges.items():
            param_values = np.linspace(min_val, max_val, num_samples)
            metric_values = {metric: [] for metric in metrics}
            
            # Original-Wert speichern
            original_value = None
            
            # Parameter-spezifische Logik
            if param_name == "scenario_probability":
                original_value = 0.2  # Standardwert
                
                for value in param_values:
                    # Kurze Simulation mit diesem Parameterwert
                    sim_results = self.run_robust_simulation(
                        num_steps=10,
                        scenario_probability=value,
                        social_influence_probability=0.3
                    )
                    
                    # Metriken berechnen
                    for metric in metrics:
                        metric_value = self._calculate_metric(metric, sim_results)
                        metric_values[metric].append(metric_value)
            
            elif param_name == "social_influence_probability":
                original_value = 0.3  # Standardwert
                
                for value in param_values:
                    # Kurze Simulation mit diesem Parameterwert
                    sim_results = self.run_robust_simulation(
                        num_steps=10,
                        scenario_probability=0.2,
                        social_influence_probability=value
                    )
                    
                    # Metriken berechnen
                    for metric in metrics:
                        metric_value = self._calculate_metric(metric, sim_results)
                        metric_values[metric].append(metric_value)
            
            # Weitere Parameter können hier hinzugefügt werden
            
            # Ergebnisse speichern
            results["parameters"][param_name] = {
                "values": param_values.tolist(),
                "original_value": original_value
            }
            
            for metric in metrics:
                results["metrics"][metric][param_name] = metric_values[metric]
        
        return results
    
    def _calculate_metric(self, metric_name: str, sim_results: Dict) -> float:
        """Berechnet eine spezifische Metrik aus den Simulationsergebnissen."""
        if metric_name == "polarization":
            # Durchschnittliche Bimodalität als Polarisierungsmaß
            polarization = self._calculate_final_polarization()
            if not polarization:
                return 0.0
            return np.mean([data["bimodality"] for data in polarization.values()])
            
        elif metric_name == "decision_consensus":
            # Konsens bei Entscheidungen (Anteil der Agenten, die die gleiche Option wählen)
            consensus = 0.0
            decision_counts = {}
            
            # Letzte 3 Entscheidungsrunden betrachten (falls verfügbar)
            num_steps = min(3, len(sim_results["decisions"]))
            
            for step_idx in range(-num_steps, 0):
                step_decisions = sim_results["decisions"][step_idx]
                
                if not step_decisions:
                    continue
                    
                # Zählen, wie oft jede Option gewählt wurde
                options_count = {}
                for agent_id, decision in step_decisions.items():
                    option = decision.get("chosen_option")
                    if option:
                        if option not in options_count:
                            options_count[option] = 0
                        options_count[option] += 1
                
                # Meistgewählte Option finden
                if options_count:
                    max_count = max(options_count.values())
                    total_decisions = sum(options_count.values())
                    
                    # Konsenslevel für diesen Schritt
                    step_consensus = max_count / total_decisions if total_decisions > 0 else 0
                    consensus += step_consensus
            
            # Durchschnittlicher Konsens über die betrachteten Schritte
            return consensus / num_steps if num_steps > 0 else 0.0
            
        elif metric_name == "belief_change_magnitude":
            # Durchschnittliche Magnitude von Überzeugungsänderungen
            total_magnitude = 0.0
            count = 0
            
            for step_changes in sim_results["belief_changes"]:
                for agent_id, belief_changes in step_changes.items():
                    for belief_name, change in belief_changes.items():
                        total_magnitude += abs(change)
                        count += 1
            
            return total_magnitude / count if count > 0 else 0.0
        
        # Weitere Metriken können hier hinzugefügt werden
        
        return 0.0
    def _calculate_polarization_at_step(self, step: int) -> Dict[str, Dict[str, float]]:
        """Berechnet die Polarisierungsmetriken für Überzeugungen zu einem bestimmten Simulationsschritt."""
        polarization = {}
        
        # Alle Überzeugungen über alle Agenten sammeln
        all_beliefs = set()
        for agent in self.agents.values():
            all_beliefs.update(agent.beliefs.keys())
        
        # Für jede Überzeugung die Stärken über Agenten hinweg sammeln
        for belief_name in all_beliefs:
            # Überzeugungsstärken für diese Überzeugung sammeln
            belief_strengths = []
            for agent in self.agents.values():
                if belief_name in agent.beliefs:
                    # Historische Daten verwenden, falls verfügbar
                    if (belief_name in agent.belief_strength_history and 
                        len(agent.belief_strength_history[belief_name]) > step):
                        strength = agent.belief_strength_history[belief_name][step]
                    else:
                        strength = agent.beliefs[belief_name].strength
                    belief_strengths.append(strength)
            
            if belief_strengths:
                # Polarisierungsmetriken berechnen
                hist, _ = np.histogram(belief_strengths, bins=10, range=(0, 1))
                bimodality = self._calculate_bimodality(hist)
                
                # Varianz berechnen
                variance = np.var(belief_strengths)
                
                # Entropie berechnen
                hist_norm = hist / np.sum(hist) if np.sum(hist) > 0 else hist
                hist_norm = hist_norm[hist_norm > 0]  # Nur positive Werte für Entropie
                entropy_value = entropy(hist_norm) if len(hist_norm) > 0 else 0
                
                polarization[belief_name] = {
                    "bimodality": bimodality,
                    "variance": variance,
                    "entropy": entropy_value
                }
        
        return polarization
    
    def _identify_belief_clusters(self) -> List[Dict]:
        """Identifiziert Cluster von Agenten mit ähnlichen Überzeugungsmustern."""
        # Hierarchisches Clustering basierend auf Überzeugungsähnlichkeit
        if len(self.agents) < 2:
            return []
            
        # Ähnlichkeitsmatrix zwischen allen Agenten erstellen
        agent_ids = list(self.agents.keys())
        similarity_matrix = np.zeros((len(agent_ids), len(agent_ids)))
        
        for i, agent1_id in enumerate(agent_ids):
            for j, agent2_id in enumerate(agent_ids):
                if i == j:
                    similarity_matrix[i][j] = 1.0  # Perfekte Ähnlichkeit mit sich selbst
                elif i < j:  # Nur einmal pro Paar berechnen
                    agent1 = self.agents[agent1_id]
                    agent2 = self.agents[agent2_id]
                    similarity = self._calculate_belief_similarity(agent1, agent2)
                    similarity_matrix[i][j] = similarity
                    similarity_matrix[j][i] = similarity  # Symmetrisch
        
        # Schwellenwert zur Cluster-Identifikation
        similarity_threshold = 0.7  # Agenten mit >70% Überzeugungsähnlichkeit werden geclustert
        clusters = []
        
        # Einfachen Clustering-Ansatz verwenden
        for i, agent_id in enumerate(agent_ids):
            # Alle Agenten mit hoher Ähnlichkeit finden
            similar_agents = [agent_ids[j] for j in range(len(agent_ids)) 
                            if similarity_matrix[i][j] > similarity_threshold]
            
            # Prüfen, ob dieser Agent bereits in einem Cluster ist
            already_clustered = False
            for cluster in clusters:
                if agent_id in cluster["agents"]:
                    already_clustered = True
                    break
                    
            if not already_clustered and len(similar_agents) > 1:  # Mindestens der Agent selbst und ein weiterer
                # Gemeinsame starke Überzeugungen in diesem Cluster identifizieren
                common_beliefs = {}
                for cluster_agent_id in similar_agents:
                    agent = self.agents[cluster_agent_id]
                    for belief_name, belief in agent.beliefs.items():
                        if belief.strength > 0.7:  # Nur starke Überzeugungen berücksichtigen
                            if belief_name not in common_beliefs:
                                common_beliefs[belief_name] = 0
                            common_beliefs[belief_name] += 1
                
                # Überzeugungen filtern, die bei mindestens der Hälfte des Clusters vorkommen
                threshold = len(similar_agents) / 2
                defining_beliefs = {belief: count for belief, count in common_beliefs.items() 
                                if count >= threshold}
                
                clusters.append({
                    "agents": similar_agents,
                    "size": len(similar_agents),
                    "average_similarity": np.mean([similarity_matrix[i][agent_ids.index(a)] 
                                                for a in similar_agents if a != agent_id]),
                    "defining_beliefs": defining_beliefs
                })
        
        # Cluster nach Größe sortieren
        clusters.sort(key=lambda x: x["size"], reverse=True)
        
        return clusters

    def analyze_results(self, results: Dict) -> Dict:
        """
        Analysiert die Simulationsergebnisse und extrahiert wichtige Einsichten.
        
        Args:
            results: Ergebnisse der Simulation (von run_robust_simulation)
            
        Returns:
            Dictionary mit Analysen
        """
        analysis = {
            "belief_evolution": {},
            "decision_patterns": {},
            "social_influence_patterns": {},
            "polarization_metrics": [],
            "opinion_clusters": [],
            "cognitive_patterns": {},
            "neural_processing_insights": {},
            "robustness_metrics": {}
        }
        
        # 1. Entwicklung der Überzeugungen über die Zeit
        for agent_id, agent in self.agents.items():
            analysis["belief_evolution"][agent_id] = {
                belief_name: {
                    "strength": strengths,
                    "certainty": agent.belief_certainty_history.get(belief_name, [])
                } for belief_name, strengths in agent.belief_strength_history.items()
            }
        
        # 2. Entscheidungsmuster analysieren
        decision_counts = {}
        processing_type_decisions = {
            NeuralProcessingType.SYSTEMATIC: {},
            NeuralProcessingType.INTUITIVE: {},
            NeuralProcessingType.ASSOCIATIVE: {},
            NeuralProcessingType.EMOTIONAL: {},
            NeuralProcessingType.ANALOGICAL: {},
            NeuralProcessingType.NARRATIVE: {}
        }
        
        for step_decisions in results["decisions"]:
            for agent_id, decision in step_decisions.items():
                if "scenario_id" not in decision or "chosen_option" not in decision:
                    continue
                    
                scenario_id = decision["scenario_id"]
                option = decision["chosen_option"]
                
                # Option-Zählung für alle Agenten
                if scenario_id not in decision_counts:
                    decision_counts[scenario_id] = {}
                if option not in decision_counts[scenario_id]:
                    decision_counts[scenario_id][option] = 0
                    
                decision_counts[scenario_id][option] += 1
                
                # Option-Zählung nach kognitivem Verarbeitungsstil
                agent = self.agents.get(agent_id)
                if agent:
                    proc_type = agent.cognitive_architecture.primary_processing
                    
                    if scenario_id not in processing_type_decisions[proc_type]:
                        processing_type_decisions[proc_type][scenario_id] = {}
                    if option not in processing_type_decisions[proc_type][scenario_id]:
                        processing_type_decisions[proc_type][scenario_id][option] = 0
                        
                    processing_type_decisions[proc_type][scenario_id][option] += 1
                
        analysis["decision_patterns"]["option_counts"] = decision_counts
        analysis["cognitive_patterns"]["processing_type_decisions"] = processing_type_decisions
        
        # 3. Polarisierung über die Zeit messen
        for step in range(len(results["belief_changes"])):
            polarization = self._calculate_polarization_at_step(step)
            analysis["polarization_metrics"].append(polarization)
            
        # 4. Meinungscluster am Ende der Simulation identifizieren
        final_clusters = self._identify_belief_clusters()
        analysis["opinion_clusters"] = final_clusters
        
        # 5. Einfluss kognitiver Verarbeitungsstile analysieren
        cognitive_influence = self._analyze_cognitive_style_influence()
        analysis["neural_processing_insights"]["cognitive_influence"] = cognitive_influence
        
        # 6. Robustheit bewerten
        if "validation" in results:
            analysis["robustness_metrics"] = {
                "error_count": len(results["validation"].get("errors", [])),
                "warning_count": len(results["validation"].get("warnings", [])),
                "simulation_stability": results["validation"].get("simulation_stability", "Unknown")
            }
            
            if "ensemble_statistics" in results:
                analysis["robustness_metrics"]["ensemble_variance"] = self._calculate_ensemble_statistics(
                    results["ensemble_statistics"])
        
        return analysis
    
    def _analyze_cognitive_style_influence(self) -> Dict:
        """Analysiert den Einfluss kognitiver Verarbeitungsstile auf Überzeugungen und Entscheidungen."""
        # Gruppen nach kognitivem Stil
        style_groups = {
            NeuralProcessingType.SYSTEMATIC: [],
            NeuralProcessingType.INTUITIVE: [],
            NeuralProcessingType.ASSOCIATIVE: [],
            NeuralProcessingType.EMOTIONAL: [],
            NeuralProcessingType.ANALOGICAL: [],
            NeuralProcessingType.NARRATIVE: []
        }
        
        # Agenten nach primärem Verarbeitungsstil gruppieren
        for agent_id, agent in self.agents.items():
            style = agent.cognitive_architecture.primary_processing
            style_groups[style].append(agent_id)
        
        # Überzeugungscharakteristiken je Gruppe berechnen
        style_belief_characteristics = {}
        
        for style, agent_ids in style_groups.items():
            if not agent_ids:
                continue
                
            # Überzeugungsstärken je kognitiver Stilgruppe
            belief_strengths = {}
            belief_certainties = {}
            
            for agent_id in agent_ids:
                agent = self.agents[agent_id]
                for belief_name, belief in agent.beliefs.items():
                    if belief_name not in belief_strengths:
                        belief_strengths[belief_name] = []
                        belief_certainties[belief_name] = []
                        
                    belief_strengths[belief_name].append(belief.strength)
                    belief_certainties[belief_name].append(belief.certainty)
            
            # Durchschnittswerte berechnen
            style_belief_stats = {}
            for belief_name in belief_strengths:
                if belief_strengths[belief_name]:
                    avg_strength = np.mean(belief_strengths[belief_name])
                    avg_certainty = np.mean(belief_certainties[belief_name])
                    
                    style_belief_stats[belief_name] = {
                        "avg_strength": avg_strength,
                        "avg_certainty": avg_certainty,
                        "strength_variance": np.var(belief_strengths[belief_name]),
                        "count": len(belief_strengths[belief_name])
                    }
            
            style_belief_characteristics[str(style)] = style_belief_stats
        
        # Vergleich: Wie stark weichen die kognitiven Stilgruppen voneinander ab?
        style_divergence = self._calculate_style_divergence(style_belief_characteristics)
        
        return {
            "style_belief_characteristics": style_belief_characteristics,
            "style_divergence": style_divergence
        }
    
    def _calculate_style_divergence(self, style_belief_stats: Dict) -> Dict:
        """Berechnet, wie stark verschiedene kognitive Stile in ihren Überzeugungen divergieren."""
        style_divergence = {}
        
        # Alle Stilpaare vergleichen
        styles = list(style_belief_stats.keys())
        
        for i, style1 in enumerate(styles):
            for style2 in styles[i+1:]:
                # Gemeinsame Überzeugungen
                common_beliefs = set(style_belief_stats[style1].keys()) & set(style_belief_stats[style2].keys())
                
                if not common_beliefs:
                    continue
                
                # Durchschnittliche Abweichung in der Stärke
                strength_diffs = []
                certainty_diffs = []
                
                for belief in common_beliefs:
                    strength_diff = abs(style_belief_stats[style1][belief]["avg_strength"] - 
                                       style_belief_stats[style2][belief]["avg_strength"])
                    strength_diffs.append(strength_diff)
                    
                    certainty_diff = abs(style_belief_stats[style1][belief]["avg_certainty"] - 
                                        style_belief_stats[style2][belief]["avg_certainty"])
                    certainty_diffs.append(certainty_diff)
                
                # Speichern der Divergenzen
                pair_key = f"{style1}_vs_{style2}"
                style_divergence[pair_key] = {
                    "avg_strength_diff": np.mean(strength_diffs),
                    "max_strength_diff": np.max(strength_diffs),
                    "avg_certainty_diff": np.mean(certainty_diffs),
                    "belief_count": len(common_beliefs),
                    "most_divergent_belief": max(common_beliefs, 
                                               key=lambda b: abs(style_belief_stats[style1][b]["avg_strength"] - 
                                                              style_belief_stats[style2][b]["avg_strength"]))
                }
        
        return style_divergence
    

    
    def _calculate_ensemble_statistics(self, ensemble_stats: Dict) -> Dict:
        """Berechnet Statistiken über die Ensemble-Durchläufe."""
        result = {
            "avg_variance": 0.0,
            "high_variance_beliefs": []
        }
        
        # Durchschnittliche Varianz
        if "variance" in ensemble_stats:
            variances = []
            high_variance = []
            
            for belief_key, var_list in ensemble_stats["variance"].items():
                avg_var = np.mean(var_list) if var_list else 0
                variances.append(avg_var)
                
                # Überzeugungen mit hoher Varianz identifizieren
                if avg_var > 0.05:  # Schwellenwert für hohe Varianz
                    high_variance.append((belief_key, avg_var))
            
            if variances:
                result["avg_variance"] = np.mean(variances)
                
            # Top-Überzeugungen mit höchster Varianz
            high_variance.sort(key=lambda x: x[1], reverse=True)
            result["high_variance_beliefs"] = high_variance[:5]  # Top 5
        
        return result
    # Removed visualize_neural_processing
    # Removed visualize_belief_network
    # Removed visualize_cognitive_style_comparison
    # Removed visualize_social_network

    def save_simulation(self, filename: str):
        """Speichert die Simulation in einer Datei."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_simulation(cls, filename: str) -> 'NeuralEthicalSociety':
        """Lädt eine Simulation aus einer Datei."""
        with open(filename, 'rb') as f:
            return pickle.load(f)


def create_example_neural_society() -> NeuralEthicalSociety:
    """Erstellt eine Beispielgesellschaft mit neurokognitiven Modellen."""
    society = NeuralEthicalSociety()
    
    # Überzeugungsvorlagen hinzufügen
    society.add_belief_template(
        "individual_freedom", "Freiheit", 
        {
            "government_control": (0.7, -1),
            "free_speech": (0.8, 1),
            "free_market": (0.6, 1)
        },
        {
            "liberty": 0.9,
            "independence": 0.8,
            "autonomy": 0.7
        },
        0.6  # Positive emotionale Valenz
    )
    
    society.add_belief_template(
        "government_control", "Freiheit",
        {
            "individual_freedom": (0.7, -1),
            "social_welfare": (0.6, 1),
            "market_regulation": (0.5, 1)
        },
        {
            "order": 0.8,
            "security": 0.7,
            "stability": 0.6
        },
        -0.2  # Leicht negative emotionale Valenz
    )
    
    society.add_belief_template(
        "free_speech", "Freiheit",
        {
            "individual_freedom": (0.8, 1),
            "hate_speech_laws": (0.7, -1)
        },
        {
            "expression": 0.9,
            "democracy": 0.6,
            "debate": 0.7
        },
        0.7  # Positive emotionale Valenz
    )
    
    society.add_belief_template(
        "hate_speech_laws", "Gerechtigkeit",
        {
            "free_speech": (0.7, -1),
            "equality": (0.6, 1)
        },
        {
            "protection": 0.8,
            "respect": 0.7,
            "dignity": 0.9
        },
        0.4  # Positive emotionale Valenz
    )
    
    society.add_belief_template(
        "equality", "Gerechtigkeit",
        {
            "meritocracy": (0.5, -1),
            "social_welfare": (0.6, 1)
        },
        {
            "fairness": 0.9,
            "justice": 0.8,
            "rights": 0.7
        },
        0.8  # Stark positive emotionale Valenz
    )
    
    society.add_belief_template(
        "meritocracy", "Wirtschaft",
        {
            "equality": (0.5, -1),
            "free_market": (0.7, 1)
        },
        {
            "effort": 0.8,
            "achievement": 0.9,
            "reward": 0.7
        },
        0.5  # Positive emotionale Valenz
    )
    
    society.add_belief_template(
        "free_market", "Wirtschaft",
        {
            "individual_freedom": (0.6, 1),
            "market_regulation": (0.8, -1),
            "meritocracy": (0.7, 1)
        },
        {
            "commerce": 0.8,
            "competition": 0.9,
            "efficiency": 0.7
        },
        0.3  # Leicht positive emotionale Valenz
    )
    
    society.add_belief_template(
        "market_regulation", "Wirtschaft",
        {
            "free_market": (0.8, -1),
            "government_control": (0.5, 1),
            "social_welfare": (0.6, 1)
        },
        {
            "oversight": 0.8,
            "fairness": 0.7,
            "consumer_protection": 0.9
        },
        0.1  # Neutral-positive emotionale Valenz
    )
    
    society.add_belief_template(
        "social_welfare", "Wohlfahrt",
        {
            "equality": (0.6, 1),
            "government_control": (0.6, 1),
            "market_regulation": (0.6, 1)
        },
        {
            "care": 0.9,
            "support": 0.8,
            "community": 0.7
        },
        0.6  # Positive emotionale Valenz
    )
    
    society.add_belief_template(
        "traditional_values", "Tradition",
        {
            "progressivism": (0.8, -1),
            "religiosity": (0.7, 1)
        },
        {
            "heritage": 0.9,
            "stability": 0.7,
            "family": 0.8
        },
        0.4  # Positive emotionale Valenz
    )
    
    society.add_belief_template(
        "progressivism", "Fortschritt",
        {
            "traditional_values": (0.8, -1),
            "science_trust": (0.6, 1)
        },
        {
            "change": 0.8,
            "innovation": 0.9,
            "adaptation": 0.7
        },
        0.5  # Positive emotionale Valenz
    )
    
    society.add_belief_template(
        "religiosity", "Religion",
        {
            "traditional_values": (0.7, 1),
            "science_trust": (0.5, -1)
        },
        {
            "faith": 0.9,
            "transcendence": 0.8,
            "spirituality": 0.7
        },
        0.7  # Positive emotionale Valenz
    )
    
    society.add_belief_template(
        "science_trust", "Wissenschaft",
        {
            "progressivism": (0.6, 1),
            "religiosity": (0.5, -1)
        },
        {
            "evidence": 0.9,
            "research": 0.8,
            "knowledge": 0.9
        },
        0.6  # Positive emotionale Valenz
    )
    
    # Diverse Gesellschaft generieren
    society.generate_diverse_society(
        num_archetypes=6,  # 6 verschiedene "Prototypen"
        agents_per_archetype=4,  # 4 ähnliche Agenten pro Archetyp
        similarity_range=(0.6, 0.9),  # Ähnlichkeitsbereich
        randomize_cognitive_styles=False  # Gezielte Verteilung der Denkstile
    )
    
    # Ethische Szenarien erstellen
    
    # Meinungsfreiheit vs. Hassrede
    hate_speech_scenario = EthicalScenario(
        scenario_id="hate_speech",
        description="Eine kontroverse Person hält eine öffentliche Rede mit potenziell beleidigenden Inhalten.",
        relevant_beliefs={
            "free_speech": 0.9,
            "hate_speech_laws": 0.9,
            "individual_freedom": 0.5,
            "equality": 0.7
        },
        options={
            "allow_speech": {
                "free_speech": 0.9,
                "hate_speech_laws": -0.7,
                "individual_freedom": 0.6
            },
            "restrict_speech": {
                "hate_speech_laws": 0.8,
                "equality": 0.6,
                "free_speech": -0.7
            },
            "monitor_but_allow": {
                "free_speech": 0.5,
                "hate_speech_laws": 0.4,
                "individual_freedom": 0.3,
                "equality": 0.3
            }
        },
        option_attributes={
            "allow_speech": {
                "risks": 0.7,
                "group_norms": {
                    "Liberals": -0.5,
                    "Conservatives": 0.3
                }
            },
            "restrict_speech": {
                "risks": 0.5,
                "group_norms": {
                    "Liberals": 0.6,
                    "Conservatives": -0.4
                }
            },
            "monitor_but_allow": {
                "risks": 0.3,
                "group_norms": {
                    "Liberals": 0.2,
                    "Conservatives": 0.1
                }
            }
        },
        outcome_feedback={
            "allow_speech": {
                "free_speech": 0.05,
                "hate_speech_laws": -0.05,
                "equality": -0.05
            },
            "restrict_speech": {
                "free_speech": -0.05,
                "hate_speech_laws": 0.05,
                "equality": 0.05
            },
            "monitor_but_allow": {
                "free_speech": 0.02,
                "hate_speech_laws": 0.02
            }
        },
        moral_implications={
            "allow_speech": {
                "liberty": 0.8,
                "fairness": -0.3,
                "care": -0.5
            },
            "restrict_speech": {
                "liberty": -0.6,
                "fairness": 0.7,
                "care": 0.6
            },
            "monitor_but_allow": {
                "liberty": 0.3,
                "fairness": 0.3,
                "care": 0.2
            }
        }
    )
    society.add_scenario(hate_speech_scenario)
    
    # Marktregulierung
    market_scenario = EthicalScenario(
        scenario_id="market_regulation",
        description="Eine neue Technologie entwickelt sich schnell, ohne klare Regulierung.",
        relevant_beliefs={
            "free_market": 0.8,
            "market_regulation": 0.8,
            "government_control": 0.6,
            "progressivism": 0.5,
            "science_trust": 0.4
        },
        options={
            "deregulate": {
                "free_market": 0.8,
                "individual_freedom": 0.6,
                "government_control": -0.5
            },
            "strict_regulation": {
                "market_regulation": 0.8,
                "government_control": 0.7,
                "free_market": -0.6
            },
            "moderate_oversight": {
                "market_regulation": 0.4,
                "free_market": 0.3,
                "science_trust": 0.5
            }
        },
        option_attributes={
            "deregulate": {
                "risks": 0.8,
                "group_norms": {
                    "Economic Conservatives": 0.7,
                    "Progressives": -0.5
                }
            },
            "strict_regulation": {
                "risks": 0.4,
                "group_norms": {
                    "Economic Conservatives": -0.6,
                    "Progressives": 0.5
                }
            },
            "moderate_oversight": {
                "risks": 0.5,
                "group_norms": {
                    "Economic Conservatives": 0.1,
                    "Progressives": 0.3
                }
            }
        },
        moral_implications={
            "deregulate": {
                "liberty": 0.7,
                "fairness": -0.4,
                "authority": -0.5
            },
            "strict_regulation": {
                "liberty": -0.6,
                "fairness": 0.5,
                "authority": 0.7
            },
            "moderate_oversight": {
                "liberty": 0.2,
                "fairness": 0.4,
                "authority": 0.3
            }
        }
    )
    society.add_scenario(market_scenario)
    
    # Soziale Wohlfahrt
    welfare_scenario = EthicalScenario(
        scenario_id="welfare_policy",
        description="Reform des sozialen Sicherheitssystems wird diskutiert.",
        relevant_beliefs={
            "social_welfare": 0.9,
            "equality": 0.8,
            "government_control": 0.7,
            "free_market": 0.6,
            "meritocracy": 0.6
        },
        options={
            "expand_programs": {
                "social_welfare": 0.8,
                "equality": 0.7,
                "government_control": 0.6,
                "meritocracy": -0.4
            },
            "reduce_programs": {
                "free_market": 0.7,
                "meritocracy": 0.8,
                "social_welfare": -0.7,
                "government_control": -0.5
            },
            "targeted_programs": {
                "social_welfare": 0.5,
                "equality": 0.5,
                "meritocracy": 0.4,
                "government_control": 0.3
            }
        },
        option_attributes={
            "expand_programs": {
                "risks": 0.5,
                "group_norms": {
                    "Progressives": 0.8,
                    "Economic Conservatives": -0.7
                }
            },
            "reduce_programs": {
                "risks": 0.6,
                "group_norms": {
                    "Progressives": -0.8,
                    "Economic Conservatives": 0.7
                }
            },
            "targeted_programs": {
                "risks": 0.3,
                "group_norms": {
                    "Progressives": 0.3,
                    "Economic Conservatives": 0.2
                }
            }
        },
        moral_implications={
            "expand_programs": {
                "care": 0.8,
                "fairness": 0.7,
                "liberty": -0.4
            },
            "reduce_programs": {
                "care": -0.6,
                "fairness": -0.5,
                "liberty": 0.7
            },
            "targeted_programs": {
                "care": 0.5,
                "fairness": 0.6,
                "liberty": 0.2
            }
        }
    )
    society.add_scenario(welfare_scenario)
    
    # Daten für Narrativ-orientierte Agenten ergänzen
    for scenario_id, scenario in society.scenarios.items():
        scenario.narrative_elements = {
            "characters": ["Bürger", "Politiker", "Experten"],
            "conflict": "Werte im Konflikt: Freiheit vs. Sicherheit vs. Gerechtigkeit",
            "context": "Moderne demokratische Gesellschaft mit unterschiedlichen Interessen",
            "coherence": 0.7
        }
    
    return society


# Beispiel für einen Testlauf
def run_demo():
    # Beispielgesellschaft erstellen
    society = create_example_neural_society()
    
    # Robuste Simulation durchführen
    print("\nRobuste Simulation wird ausgeführt...")
    results = society.run_robust_simulation(
        num_steps=15, 
        scenario_probability=0.3,
        social_influence_probability=0.4,
        reflection_probability=0.2
    )
    
    # Ergebnisse analysieren
    print("\nAnalyse der Ergebnisse...")
    analysis = society.analyze_results(results)
    
    # Entwicklung der Überzeugungen visualisieren
    print("\nEntwicklung der Überzeugung 'free_speech':")
    for agent_id in list(society.agents.keys())[:3]:  # Erste 3 Agenten
        agent = society.agents[agent_id]
        style = agent.cognitive_architecture.primary_processing
        print(f"Agent {agent_id} (Stil: {style})")
        
    # Einen Agenten mit systematischem und einen mit intuitivem Denkstil auswählen
    systematic_agent = None
    intuitive_agent = None
    
    for agent_id, agent in society.agents.items():
        if agent.cognitive_architecture.primary_processing == NeuralProcessingType.SYSTEMATIC and not systematic_agent:
            systematic_agent = agent_id
        elif agent.cognitive_architecture.primary_processing == NeuralProcessingType.INTUITIVE and not intuitive_agent:
            intuitive_agent = agent_id
            
        if systematic_agent and intuitive_agent:
            break
    
    # Neuronale Verarbeitung visualisieren
    # print("\nVisualisierung der neuronalen Verarbeitung eines systematischen Denkers:")
    # if systematic_agent:
    #     visualize_neural_processing(society, systematic_agent) # Updated and commented
    
    # if intuitive_agent:
    #     print("\nVisualisierung der neuronalen Verarbeitung eines intuitiven Denkers:")
    #     visualize_neural_processing(society, intuitive_agent) # Updated and commented
    
    # Kognitive Stile vergleichen
    # print("\nVergleich der kognitiven Stile:")
    # visualize_cognitive_style_comparison(society) # Updated and commented
    
    # Soziales Netzwerk nach kognitivem Stil
    # print("\nSoziales Netzwerk nach kognitivem Stil:")
    # visualize_social_network(society, color_by='cognitive_style') # Updated and commented

    print("\n--- Detailed Analysis Output ---")
    if "polarization_metrics" in analysis and analysis["polarization_metrics"]:
        print("\nFinal Polarization Metrics:")
        # Print the last entry, assuming it's the final state
        final_polarization = analysis["polarization_metrics"][-1]
        for belief, metrics in final_polarization.items():
            print(f"  Belief: {belief}, Bimodality: {metrics.get('bimodality', 'N/A'):.2f}, Variance: {metrics.get('variance', 'N/A'):.2f}, Entropy: {metrics.get('entropy', 'N/A'):.2f}")

    if "opinion_clusters" in analysis and analysis["opinion_clusters"]:
        print("\nOpinion Clusters (Final):")
        for i, cluster in enumerate(analysis["opinion_clusters"]):
            print(f"  Cluster {i+1}: Size {cluster.get('size', 'N/A')}, Agents: {cluster.get('agents', [])[:3]}...") # Print first 3 agents
            # Optionally print some defining beliefs if the structure allows and is not too verbose
            if "defining_beliefs" in cluster and cluster["defining_beliefs"]:
                 print(f"    Defining beliefs (sample): {dict(list(cluster['defining_beliefs'].items())[:2])}")


    if "robustness_metrics" in analysis:
        print("\nRobustness Metrics:")
        print(f"  Error Count: {analysis['robustness_metrics'].get('error_count', 'N/A')}")
        print(f"  Warning Count: {analysis['robustness_metrics'].get('warning_count', 'N/A')}")
        print(f"  Simulation Stability: {analysis['robustness_metrics'].get('simulation_stability', 'N/A')}")

    if "decision_patterns" in analysis and "option_counts" in analysis["decision_patterns"]:
        print("\nDecision Patterns (Option Counts - Sample):")
        for scenario_id, options in list(analysis["decision_patterns"]["option_counts"].items())[:2]: # Sample 2 scenarios
            print(f"  Scenario: {scenario_id}")
            for option, count in options.items():
                print(f"    Option: {option}, Count: {count}")
    print("--- End of Detailed Analysis Output ---")
    
    return society, results, analysis


if __name__ == "__main__":
    society, results, analysis = run_demo()
