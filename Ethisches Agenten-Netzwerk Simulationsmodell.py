import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import cosine
from sklearn.manifold import TSNE
import seaborn as sns
from tqdm import tqdm
import random
from typing import List, Dict, Tuple, Optional, Set, Union
import pickle
import os

class EthicalBelief:
    """Repräsentiert eine einzelne ethische Überzeugung oder Wert."""
    
    def __init__(self, name: str, category: str, initial_strength: float = 0.5):
        """
        Initialisiert eine ethische Überzeugung.
        
        Args:
            name: Beschreibender Name der Überzeugung
            category: Kategorie der Überzeugung (z.B. 'Gerechtigkeit', 'Freiheit')
            initial_strength: Anfängliche Stärke der Überzeugung (0-1)
        """
        self.name = name
        self.category = category
        self.strength = np.clip(initial_strength, 0.0, 1.0)
        
        # Verbindungen zu anderen Überzeugungen (Belief -> (Einfluss, Polarität))
        # Polarität: +1 bedeutet positive Verstärkung, -1 bedeutet negative Beeinflussung
        self.connections = {}
        
    def add_connection(self, belief_name: str, influence_strength: float, polarity: int):
        """Fügt eine Verbindung zu einer anderen Überzeugung hinzu."""
        self.connections[belief_name] = (np.clip(influence_strength, 0.0, 1.0), np.sign(polarity))
        
    def update_strength(self, new_strength: float):
        """Aktualisiert die Stärke der Überzeugung."""
        self.strength = np.clip(new_strength, 0.0, 1.0)


class EthicalAgent:
    """Repräsentiert einen menschlichen Agenten mit ethischen Überzeugungen."""
    
    def __init__(self, agent_id: str, personality_traits: Dict[str, float] = None):
        """
        Initialisiert einen ethischen Agenten.
        
        Args:
            agent_id: Eindeutige ID des Agenten
            personality_traits: Persönlichkeitsmerkmale des Agenten
        """
        self.agent_id = agent_id
        self.beliefs = {}  # name -> EthicalBelief
        
        # Persönlichkeitsmerkmale (Offenheit, Gewissenhaftigkeit, etc.)
        self.personality_traits = personality_traits or {
            "openness": np.random.beta(5, 5),          # Offenheit für neue Ideen
            "agreeableness": np.random.beta(5, 5),     # Verträglichkeit
            "conscientiousness": np.random.beta(5, 5),  # Gewissenhaftigkeit
            "stability": np.random.beta(5, 5),         # Emotionale Stabilität
            "extroversion": np.random.beta(5, 5)       # Extroversion
        }
        
        # Historische Entscheidungen und Überzeugungsstärken
        self.decision_history = []
        self.belief_strength_history = {}
        
        # Soziales Netzwerk - IDs anderer Agenten und Stärke der Verbindung
        self.social_connections = {}  # agent_id -> connection_strength
        
        # Kognitiver Stil (wie der Agent Informationen verarbeitet)
        self.cognitive_style = {
            "analytical_thinking": np.random.beta(5, 5),  # Analytisches vs. intuitives Denken
            "need_for_cognition": np.random.beta(5, 5),   # Bedürfnis nach kognitiver Aktivität
            "dogmatism": np.random.beta(3, 7),            # Dogmatismus vs. Offenheit für Komplexität
            "ambiguity_tolerance": np.random.beta(5, 5)   # Ambiguitätstoleranz
        }
        
        # Gruppenidentitäten
        self.group_identities = {}  # group_name -> identification_strength
        
    def add_belief(self, belief: EthicalBelief):
        """Fügt eine ethische Überzeugung hinzu."""
        self.beliefs[belief.name] = belief
        self.belief_strength_history[belief.name] = [belief.strength]
        
    def update_belief(self, belief_name: str, new_strength: float):
        """Aktualisiert die Stärke einer Überzeugung."""
        if belief_name in self.beliefs:
            self.beliefs[belief_name].update_strength(new_strength)
            self.belief_strength_history[belief_name].append(new_strength)
    
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
    
    def get_belief_categories(self) -> Dict[str, List[str]]:
        """Gibt ein Dictionary mit Kategorien als Schlüssel und Listen von Überzeugungsnamen zurück."""
        categories = {}
        for belief in self.beliefs.values():
            if belief.category not in categories:
                categories[belief.category] = []
            categories[belief.category].append(belief.name)
        return categories
    
    def calculate_cognitive_dissonance(self) -> float:
        """Berechnet die kognitive Dissonanz basierend auf widersprüchlichen Überzeugungen."""
        dissonance = 0.0
        processed_pairs = set()
        
        for belief_name, belief in self.beliefs.items():
            for other_name, (influence, polarity) in belief.connections.items():
                if other_name in self.beliefs and (belief_name, other_name) not in processed_pairs:
                    # Dissonanz entsteht, wenn starke Überzeugungen gegensätzlich verbunden sind
                    if polarity < 0:
                        dissonance += belief.strength * self.beliefs[other_name].strength * influence * abs(polarity)
                    processed_pairs.add((belief_name, other_name))
                    processed_pairs.add((other_name, belief_name))
                    
        return dissonance
    
    def make_decision(self, scenario: 'EthicalScenario') -> Dict[str, Union[str, float, Dict]]:
        """
        Trifft eine Entscheidung in einem ethischen Szenario.
        
        Returns:
            Dict mit der Entscheidung und Begründungen
        """
        # Relevante Überzeugungen für dieses Szenario identifizieren
        relevant_beliefs = {}
        for belief_name, relevance in scenario.relevant_beliefs.items():
            if belief_name in self.beliefs:
                relevant_beliefs[belief_name] = (self.beliefs[belief_name].strength * relevance)
        
        # Optionen bewerten
        option_scores = {}
        for option_name, option_impacts in scenario.options.items():
            score = 0
            justifications = {}
            
            for belief_name, impact in option_impacts.items():
                if belief_name in self.beliefs:
                    belief_score = self.beliefs[belief_name].strength * impact
                    score += belief_score
                    justifications[belief_name] = belief_score
            
            # Persönlichkeits- und Denkstil-Einflüsse
            if "risks" in scenario.option_attributes.get(option_name, {}):
                risk_aversion = 0.7 - 0.4 * self.personality_traits["openness"]
                risk_adjustment = -scenario.option_attributes[option_name]["risks"] * risk_aversion
                score += risk_adjustment
                justifications["risk_consideration"] = risk_adjustment
                
            option_scores[option_name] = {
                "score": score,
                "justifications": justifications
            }
        
        # Beste Option auswählen (mit etwas Zufall für menschliche Unberechenbarkeit)
        # Bei analytischerem Denkstil ist die Entscheidung weniger zufällig
        randomness_factor = 1.0 - self.cognitive_style["analytical_thinking"]
        decision_noise = np.random.normal(0, randomness_factor, len(option_scores))
        
        options = list(option_scores.keys())
        scores = [option_scores[opt]["score"] for opt in options]
        noisy_scores = np.array(scores) + decision_noise
        chosen_option = options[np.argmax(noisy_scores)]
        
        decision = {
            "scenario_id": scenario.scenario_id,
            "chosen_option": chosen_option,
            "option_scores": option_scores,
            "cognitive_dissonance": self.calculate_cognitive_dissonance(),
            "timestamp": len(self.decision_history)
        }
        
        # Entscheidung zur Historie hinzufügen
        self.decision_history.append(decision)
        
        return decision
    
    def update_beliefs_from_experience(self, scenario: 'EthicalScenario', chosen_option: str) -> Dict[str, float]:
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
                    old_strength = self.beliefs[belief_name].strength
                    # Lernrate basierend auf Persönlichkeit
                    learning_rate = 0.05 * (0.5 + 0.5 * self.personality_traits["openness"])
                    
                    # Neue Stärke basierend auf Feedback
                    new_strength = old_strength + learning_rate * feedback
                    self.update_belief(belief_name, new_strength)
                    belief_changes[belief_name] = new_strength - old_strength
        
        # Propagation der Änderungen durch das Netzwerk von Überzeugungen
        propagated_changes = self._propagate_belief_changes(belief_changes)
        belief_changes.update(propagated_changes)
        
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
        
        for belief_name, change in initial_changes.items():
            if belief_name in self.beliefs:
                belief = self.beliefs[belief_name]
                
                # Änderungen an verbundene Überzeugungen weitergeben
                for connected_belief, (influence, polarity) in belief.connections.items():
                    if connected_belief in self.beliefs:
                        # Stärke der Änderung basierend auf Verbindungsstärke und Polarität
                        connected_change = change * influence * polarity * 0.5
                        
                        # Aktualisieren der verbundenen Überzeugung
                        old_strength = self.beliefs[connected_belief].strength
                        new_strength = old_strength + connected_change
                        self.update_belief(connected_belief, new_strength)
                        
                        if connected_belief not in initial_changes:
                            propagated_changes[connected_belief] = new_strength - old_strength
        
        return propagated_changes
    
    def update_from_social_influence(self, other_agent: 'EthicalAgent') -> Dict[str, float]:
        """
        Aktualisiert Überzeugungen basierend auf dem sozialen Einfluss eines anderen Agenten.
        
        Returns:
            Dictionary mit Überzeugungsänderungen
        """
        if other_agent.agent_id not in self.social_connections:
            return {}
        
        connection_strength = self.social_connections[other_agent.agent_id]
        belief_changes = {}
        
        # Soziale Lernrate basierend auf Persönlichkeit
        base_social_learning_rate = 0.02
        agreeableness_factor = 0.5 + 0.5 * self.personality_traits["agreeableness"]
        social_learning_rate = base_social_learning_rate * agreeableness_factor * connection_strength
        
        # Überzeugungen vergleichen und aktualisieren
        common_beliefs = set(self.beliefs.keys()).intersection(set(other_agent.beliefs.keys()))
        for belief_name in common_beliefs:
            # Stärke des Einflusses hängt von der Differenz der Überzeugungsstärken ab
            my_strength = self.beliefs[belief_name].strength
            other_strength = other_agent.beliefs[belief_name].strength
            strength_diff = other_strength - my_strength
            
            # Gewichtung des Einflusses basierend auf Gruppenidentität
            group_weight = 1.0
            for group, my_identity in self.group_identities.items():
                if group in other_agent.group_identities:
                    other_identity = other_agent.group_identities[group]
                    # Stärkerer Einfluss bei gemeinsamer Gruppenidentität
                    if my_identity > 0.5 and other_identity > 0.5:
                        group_weight *= 1.0 + 0.5 * min(my_identity, other_identity)
            
            # Aktualisierung basierend auf sozialem Einfluss
            change = strength_diff * social_learning_rate * group_weight
            
            # Kognitive Faktoren berücksichtigen
            if abs(strength_diff) > 0.3:  # Große Meinungsunterschiede
                # Dogmatischere Menschen ändern ihre Meinung weniger
                change *= (1.0 - 0.7 * self.cognitive_style["dogmatism"])
                
            # Aktualisieren der Überzeugung
            new_strength = my_strength + change
            self.update_belief(belief_name, new_strength)
            belief_changes[belief_name] = change
            
        return belief_changes
        
    def __str__(self):
        """String-Repräsentation des Agenten."""
        return f"Agent {self.agent_id} mit {len(self.beliefs)} Überzeugungen"


class EthicalScenario:
    """Repräsentiert ein ethisches Szenario oder Dilemma."""
    
    def __init__(self, scenario_id: str, description: str, 
                 relevant_beliefs: Dict[str, float],
                 options: Dict[str, Dict[str, float]],
                 option_attributes: Dict[str, Dict[str, float]] = None,
                 outcome_feedback: Dict[str, Dict[str, float]] = None):
        """
        Initialisiert ein ethisches Szenario.
        
        Args:
            scenario_id: Eindeutige ID des Szenarios
            description: Beschreibung des Szenarios
            relevant_beliefs: Dictionary mit relevanten Überzeugungen und ihrer Relevanz
            options: Dictionary mit Optionen und ihren Auswirkungen auf Überzeugungen
            option_attributes: Zusätzliche Attribute für jede Option (z.B. Risiko)
            outcome_feedback: Feedback für jede Option, wie sie Überzeugungen beeinflusst
        """
        self.scenario_id = scenario_id
        self.description = description
        self.relevant_beliefs = relevant_beliefs
        self.options = options
        self.option_attributes = option_attributes or {}
        self.outcome_feedback = outcome_feedback or {}


class EthicalSociety:
    """Repräsentiert eine Gesellschaft von ethischen Agenten."""
    
    def __init__(self):
        """Initialisiert eine Gesellschaft von ethischen Agenten."""
        self.agents = {}  # agent_id -> EthicalAgent
        self.scenarios = {}  # scenario_id -> EthicalScenario
        self.belief_templates = {}  # Template-Überzeugungen für die Agentenerzeugung
        self.social_network = nx.Graph()  # Soziales Netzwerk
        self.groups = {}  # group_name -> set of agent_ids
        
    def add_agent(self, agent: EthicalAgent):
        """Fügt einen Agenten zur Gesellschaft hinzu."""
        self.agents[agent.agent_id] = agent
        self.social_network.add_node(agent.agent_id)
        
    def add_scenario(self, scenario: EthicalScenario):
        """Fügt ein Szenario zur Gesellschaft hinzu."""
        self.scenarios[scenario.scenario_id] = scenario
        
    def add_belief_template(self, name: str, category: str, 
                           connections: Dict[str, Tuple[float, int]] = None):
        """
        Fügt eine Template-Überzeugung hinzu.
        
        Args:
            name: Name der Überzeugung
            category: Kategorie der Überzeugung
            connections: Verbindungen zu anderen Überzeugungen (name -> (strength, polarity))
        """
        self.belief_templates[name] = (category, connections or {})
        
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
                             connection_probability: float = 0.3) -> EthicalAgent:
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
        agent = EthicalAgent(agent_id)
        
        # Zufällige Anzahl an Überzeugungen auswählen
        num_beliefs = np.random.randint(min_beliefs, max_beliefs + 1)
        belief_names = list(self.belief_templates.keys())
        selected_belief_names = np.random.choice(belief_names, 
                                               size=min(num_beliefs, len(belief_names)), 
                                               replace=False)
        
        # Überzeugungen hinzufügen
        for belief_name in selected_belief_names:
            category, connections = self.belief_templates[belief_name]
            
            # Anfangsstärke basierend auf der gewählten Verteilung
            if belief_distribution == "normal":
                initial_strength = np.clip(np.random.normal(0.5, 0.15), 0.1, 0.9)
            elif belief_distribution == "uniform":
                initial_strength = np.random.uniform(0.1, 0.9)
            elif belief_distribution == "beta":
                initial_strength = np.random.beta(2, 2)
            else:
                initial_strength = 0.5
                
            # Überzeugung erstellen
            belief = EthicalBelief(belief_name, category, initial_strength)
            
            # Verbindungen hinzufügen (nur zu anderen ausgewählten Überzeugungen)
            for conn_name, (conn_strength, polarity) in connections.items():
                if conn_name in selected_belief_names:
                    # Variabilität in Verbindungsstärken
                    actual_strength = conn_strength * np.random.uniform(0.8, 1.2)
                    belief.add_connection(conn_name, actual_strength, polarity)
            
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
    
    def generate_similar_agent(self, base_agent: EthicalAgent, agent_id: str, 
                              similarity: float = 0.8) -> EthicalAgent:
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
        new_agent = EthicalAgent(agent_id)
        
        # Persönlichkeitsmerkmale anpassen
        for trait, value in base_agent.personality_traits.items():
            # Wert in der Nähe des Basis-Agenten, aber mit etwas Variation
            variation = np.random.normal(0, (1 - similarity) * 0.2)
            new_value = np.clip(value + variation, 0.0, 1.0)
            new_agent.personality_traits[trait] = new_value
            
        # Kognitiven Stil anpassen
        for style, value in base_agent.cognitive_style.items():
            variation = np.random.normal(0, (1 - similarity) * 0.2)
            new_value = np.clip(value + variation, 0.0, 1.0)
            new_agent.cognitive_style[style] = new_value
            
        # Überzeugungen kopieren und variieren
        for belief_name, belief in base_agent.beliefs.items():
            # Mit zunehmender Unähnlichkeit steigt die Wahrscheinlichkeit, eine Überzeugung auszulassen
            if np.random.random() < similarity:
                # Variieren der Überzeugungsstärke
                variation = np.random.normal(0, (1 - similarity) * 0.3)
                new_strength = np.clip(belief.strength + variation, 0.0, 1.0)
                
                # Neue Überzeugung erstellen
                new_belief = EthicalBelief(belief.name, belief.category, new_strength)
                
                # Verbindungen kopieren und variieren
                for conn_name, (conn_strength, polarity) in belief.connections.items():
                    # Mit geringer Wahrscheinlichkeit Polarität umkehren
                    if np.random.random() < 0.1 * (1 - similarity):
                        polarity *= -1
                        
                    # Verbindungsstärke variieren
                    variation = np.random.normal(0, (1 - similarity) * 0.2)
                    new_conn_strength = np.clip(conn_strength + variation, 0.0, 1.0)
                    
                    new_belief.add_connection(conn_name, new_conn_strength, polarity)
                
                new_agent.add_belief(new_belief)
                
        # Gruppenzugehörigkeiten mit Variation kopieren
        for group, identification in base_agent.group_identities.items():
            # Variieren der Identifikationsstärke
            variation = np.random.normal(0, (1 - similarity) * 0.3)
            new_identification = np.clip(identification + variation, 0.0, 1.0)
            new_agent.add_group_identity(group, new_identification)
            
        return new_agent
    
    def generate_population_cluster(self, base_agents: List[EthicalAgent], 
                                  num_agents: int, similarity_range: Tuple[float, float]) -> List[EthicalAgent]:
        """
        Generiert eine Gruppe von Agenten basierend auf mehreren Basis-Agenten.
        
        Args:
            base_agents: Liste von Basis-Agenten
            num_agents: Anzahl zu generierender Agenten
            similarity_range: Bereich der Ähnlichkeit (min, max)
            
        Returns:
            Liste neuer Agenten
        """
        new_agents = []
        
        for i in range(num_agents):
            # Zufälligen Basis-Agenten auswählen
            base_agent = np.random.choice(base_agents)
            
            # Zufällige Ähnlichkeit im angegebenen Bereich
            similarity = np.random.uniform(similarity_range[0], similarity_range[1])
            
            # Neuen Agenten erstellen
            agent_id = f"agent_{len(self.agents) + len(new_agents) + 1}"
            new_agent = self.generate_similar_agent(base_agent, agent_id, similarity)
            new_agents.append(new_agent)
            
        return new_agents
    
    def generate_realistic_social_network(self, connection_density: float = 0.1,
                                        group_connection_boost: float = 0.3,
                                        belief_similarity_factor: float = 0.5):
        """
        Generiert ein realistisches soziales Netzwerk basierend auf Gruppen und Überzeugungsähnlichkeit.
        
        Args:
            connection_density: Grundlegende Dichte von Verbindungen
            group_connection_boost: Erhöhte Verbindungswahrscheinlichkeit innerhalb von Gruppen
            belief_similarity_factor: Einfluss der Überzeugungsähnlichkeit auf Verbindungen
        """
        agent_ids = list(self.agents.keys())
        
        # Berechnen von Ähnlichkeiten zwischen Agenten basierend auf Überzeugungen
        similarity_matrix = {}
        for i, agent1_id in enumerate(agent_ids):
            for agent2_id in agent_ids[i+1:]:
                agent1 = self.agents[agent1_id]
                agent2 = self.agents[agent2_id]
                
                # Überzeugungsähnlichkeit berechnen
                belief_sim = self._calculate_belief_similarity(agent1, agent2)
                
                # Gruppenzugehörigkeitsähnlichkeit berechnen
                group_sim = self._calculate_group_similarity(agent1, agent2)
                
                # Gewichtete Gesamtähnlichkeit
                total_sim = belief_similarity_factor * belief_sim + (1 - belief_similarity_factor) * group_sim
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
            
            # Verbindung mit berechneter Wahrscheinlichkeit erstellen
            if np.random.random() < min(prob, 0.95):  # Max 95% Wahrscheinlichkeit
                # Verbindungsstärke basierend auf Ähnlichkeit
                strength = 0.3 + 0.7 * similarity
                self.add_social_connection(agent1_id, agent2_id, strength)
    
    def _calculate_belief_similarity(self, agent1: EthicalAgent, agent2: EthicalAgent) -> float:
        """Berechnet die Ähnlichkeit der Überzeugungen zwischen zwei Agenten."""
        # Gemeinsame Überzeugungen finden
        common_beliefs = set(agent1.beliefs.keys()) & set(agent2.beliefs.keys())
        
        if not common_beliefs:
            return 0.0
        
        # Ähnlichkeit basierend auf Überzeugungsstärken
        similarity = 0.0
        for belief_name in common_beliefs:
            strength1 = agent1.beliefs[belief_name].strength
            strength2 = agent2.beliefs[belief_name].strength
            # Je näher die Stärken, desto ähnlicher
            similarity += 1.0 - abs(strength1 - strength2)
            
        return similarity / len(common_beliefs)
    
    def _calculate_group_similarity(self, agent1: EthicalAgent, agent2: EthicalAgent) -> float:
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
    
    def run_simulation(self, num_steps: int, 
                      scenario_probability: float = 0.2,
                      social_influence_probability: float = 0.3) -> Dict:
        """
        Führt die Simulation über mehrere Zeitschritte aus.
        
        Args:
            num_steps: Anzahl der Simulationsschritte
            scenario_probability: Wahrscheinlichkeit, dass ein Agent in einem Schritt ein Szenario erlebt
            social_influence_probability: Wahrscheinlichkeit für sozialen Einfluss in einem Schritt
            
        Returns:
            Dictionary mit Simulationsergebnissen
        """
        results = {
            "decisions": [],
            "belief_changes": [],
            "social_influences": []
        }
        
        # Für jeden Zeitschritt
        for step in tqdm(range(num_steps), desc="Simulation Steps"):
            step_results = {
                "step": step,
                "decisions": {},
                "belief_changes": {},
                "social_influences": {}
            }
            
            # Für jeden Agenten
            for agent_id, agent in self.agents.items():
                # 1. Mögliches Szenario erleben
                if np.random.random() < scenario_probability and self.scenarios:
                    # Zufälliges Szenario auswählen
                    scenario_id = np.random.choice(list(self.scenarios.keys()))
                    scenario = self.scenarios[scenario_id]
                    
                    # Entscheidung treffen
                    decision = agent.make_decision(scenario)
                    step_results["decisions"][agent_id] = decision
                    
                    # Überzeugungen basierend auf Erfahrung aktualisieren
                    belief_changes = agent.update_beliefs_from_experience(
                        scenario, decision["chosen_option"])
                    step_results["belief_changes"][agent_id] = belief_changes
                
                # 2. Möglicher sozialer Einfluss
                if np.random.random() < social_influence_probability and agent.social_connections:
                    # Zufälligen verbundenen Agenten auswählen
                    connected_id = np.random.choice(list(agent.social_connections.keys()))
                    connected_agent = self.agents[connected_id]
                    
                    # Überzeugungen basierend auf sozialem Einfluss aktualisieren
                    social_changes = agent.update_from_social_influence(connected_agent)
                    if social_changes:
                        if agent_id not in step_results["social_influences"]:
                            step_results["social_influences"][agent_id] = {}
                        step_results["social_influences"][agent_id][connected_id] = social_changes
            
            # Ergebnisse für diesen Schritt speichern
            results["decisions"].append(step_results["decisions"])
            results["belief_changes"].append(step_results["belief_changes"])
            results["social_influences"].append(step_results["social_influences"])
            
        return results
    
    def analyze_results(self, results: Dict) -> Dict:
        """
        Analysiert die Simulationsergebnisse.
        
        Args:
            results: Ergebnisse der Simulation (von run_simulation)
            
        Returns:
            Dictionary mit Analysen
        """
        analysis = {
            "belief_evolution": {},
            "decision_patterns": {},
            "social_influence_patterns": {},
            "polarization_metrics": [],
            "opinion_clusters": []
        }
        
        # 1. Entwicklung der Überzeugungen über die Zeit
        for agent_id, agent in self.agents.items():
            analysis["belief_evolution"][agent_id] = {
                belief_name: strengths 
                for belief_name, strengths in agent.belief_strength_history.items()
            }
        
        # 2. Entscheidungsmuster analysieren
        decision_counts = {}
        for step_decisions in results["decisions"]:
            for agent_id, decision in step_decisions.items():
                scenario_id = decision["scenario_id"]
                option = decision["chosen_option"]
                
                if scenario_id not in decision_counts:
                    decision_counts[scenario_id] = {}
                if option not in decision_counts[scenario_id]:
                    decision_counts[scenario_id][option] = 0
                    
                decision_counts[scenario_id][option] += 1
                
        analysis["decision_patterns"]["option_counts"] = decision_counts
        
        # 3. Polarisierung über die Zeit messen
        for step in range(len(results["belief_changes"])):
            polarization = self._calculate_polarization_at_step(step)
            analysis["polarization_metrics"].append(polarization)
            
        # 4. Meinungscluster am Ende der Simulation identifizieren
        final_clusters = self._identify_belief_clusters()
        analysis["opinion_clusters"] = final_clusters
        
        return analysis
    
    def _calculate_polarization_at_step(self, step: int) -> Dict:
        """Berechnet Polarisierungsmetriken für einen bestimmten Zeitschritt."""
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
                    # Historischen Wert zum Zeitpunkt step nehmen (oder aktuellen, wenn nicht verfügbar)
                    history = agent.belief_strength_history.get(belief_name, [])
                    if step < len(history):
                        strength = history[step]
                    else:
                        strength = agent.beliefs[belief_name].strength
                    belief_strengths.append(strength)
            
            if belief_strengths:
                # Bimodalität als Polarisierungsmaß
                hist, _ = np.histogram(belief_strengths, bins=10, range=(0, 1))
                bimodality = self._calculate_bimodality(hist)
                
                # Varianz als Maß für Meinungsvielfalt
                variance = np.var(belief_strengths)
                
                polarization[belief_name] = {
                    "bimodality": bimodality,
                    "variance": variance
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
    
    def _identify_belief_clusters(self) -> List[Dict]:
        """Identifiziert Cluster von Agenten mit ähnlichen Überzeugungen."""
        # Überzeugungsvektoren für alle Agenten erstellen
        agent_vectors = {}
        all_beliefs = set()
        
        for agent in self.agents.values():
            all_beliefs.update(agent.beliefs.keys())
            
        belief_list = sorted(list(all_beliefs))
        
        for agent_id, agent in self.agents.items():
            vector = np.zeros(len(belief_list))
            for i, belief_name in enumerate(belief_list):
                if belief_name in agent.beliefs:
                    vector[i] = agent.beliefs[belief_name].strength
            agent_vectors[agent_id] = vector
            
        # Ähnlichkeitsmatrix erstellen
        agent_ids = list(agent_vectors.keys())
        similarity_matrix = np.zeros((len(agent_ids), len(agent_ids)))
        
        for i, id1 in enumerate(agent_ids):
            for j, id2 in enumerate(agent_ids):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    # Kosinus-Ähnlichkeit zwischen Vektoren
                    similarity_matrix[i, j] = 1.0 - cosine(agent_vectors[id1], agent_vectors[id2])
        
        # Clustering mit spektraler Clusteranalyse
        try:
            from sklearn.cluster import SpectralClustering
            
            # Schätze optimale Clusterzahl
            if len(agent_ids) <= 5:
                n_clusters = 2
            else:
                n_clusters = min(5, len(agent_ids) // 3)
            
            clustering = SpectralClustering(
                n_clusters=n_clusters, 
                affinity='precomputed',
                random_state=42
            ).fit(similarity_matrix)
            
            # Ergebnisse sammeln
            clusters = []
            for i in range(n_clusters):
                cluster_agents = [agent_ids[j] for j in range(len(agent_ids)) if clustering.labels_[j] == i]
                
                # Repräsentative Überzeugungen für diesen Cluster berechnen
                cluster_beliefs = {}
                for belief_name in belief_list:
                    values = []
                    for agent_id in cluster_agents:
                        agent = self.agents[agent_id]
                        if belief_name in agent.beliefs:
                            values.append(agent.beliefs[belief_name].strength)
                            
                    if values:
                        cluster_beliefs[belief_name] = np.mean(values)
                
                clusters.append({
                    "cluster_id": i,
                    "agent_ids": cluster_agents,
                    "size": len(cluster_agents),
                    "representative_beliefs": cluster_beliefs
                })
                
            return clusters
            
        except ImportError:
            # Falls sklearn nicht verfügbar ist, einfache Lösung
            return [{"cluster_id": 0, "agent_ids": agent_ids, "size": len(agent_ids)}]
    
    def visualize_belief_network(self, agent_id: str, min_connection_strength: float = 0.2):
        """Visualisiert das Netzwerk von Überzeugungen eines Agenten."""
        if agent_id not in self.agents:
            print(f"Agent {agent_id} nicht gefunden.")
            return
            
        agent = self.agents[agent_id]
        G = nx.DiGraph()
        
        # Knoten für jede Überzeugung hinzufügen
        for belief_name, belief in agent.beliefs.items():
            G.add_node(belief_name, strength=belief.strength, category=belief.category)
            
            # Kanten für Verbindungen hinzufügen
            for conn_name, (strength, polarity) in belief.connections.items():
                if strength >= min_connection_strength and conn_name in agent.beliefs:
                    G.add_edge(belief_name, conn_name, weight=strength, polarity=polarity)
        
        # Netzwerk zeichnen
        plt.figure(figsize=(12, 10))
        
        # Positionen berechnen
        pos = nx.spring_layout(G, seed=42)
        
        # Knoten zeichnen, Farbe basierend auf Kategorie
        categories = set(nx.get_node_attributes(G, 'category').values())
        colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
        category_colors = dict(zip(categories, colors))
        
        for category, color in category_colors.items():
            node_list = [node for node, data in G.nodes(data=True) if data['category'] == category]
            strengths = [G.nodes[node]['strength'] for node in node_list]
            nx.draw_networkx_nodes(
                G, pos, 
                nodelist=node_list, 
                node_color=[color] * len(node_list),
                node_size=[s * 500 for s in strengths],
                alpha=0.8,
                label=category
            )
        
        # Kanten zeichnen, rot für negative, grün für positive
        pos_edges = [(u, v) for u, v, d in G.edges(data=True) if d['polarity'] > 0]
        neg_edges = [(u, v) for u, v, d in G.edges(data=True) if d['polarity'] < 0]
        
        edge_weights = [G[u][v]['weight'] * 2 for u, v in pos_edges]
        nx.draw_networkx_edges(G, pos, edgelist=pos_edges, width=edge_weights, 
                              edge_color='green', alpha=0.6, arrows=True)
        
        edge_weights = [G[u][v]['weight'] * 2 for u, v in neg_edges]
        nx.draw_networkx_edges(G, pos, edgelist=neg_edges, width=edge_weights, 
                              edge_color='red', alpha=0.6, arrows=True, style='dashed')
        
        # Knotenbeschriftungen
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
        
        plt.title(f"Überzeugungsnetzwerk für Agent {agent_id}")
        plt.legend()
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def visualize_social_network(self, color_by: str = 'group'):
        """
        Visualisiert das soziale Netzwerk der Agenten.
        
        Args:
            color_by: Bestimmt, wie die Knoten gefärbt werden ('group' oder 'belief')
        """
        plt.figure(figsize=(12, 10))
        
        # Positionen berechnen
        pos = nx.spring_layout(self.social_network, seed=42)
        
        if color_by == 'group':
            # Färben nach Gruppenzugehörigkeit (primäre Gruppe für jeden Agenten)
            colors = []
            for agent_id in self.social_network.nodes():
                agent = self.agents[agent_id]
                
                # Primäre Gruppe finden (mit höchster Identifikation)
                primary_group = None
                max_id = 0.0
                
                for group, id_strength in agent.group_identities.items():
                    if id_strength > max_id:
                        max_id = id_strength
                        primary_group = group
                
                colors.append(hash(primary_group) % 10 if primary_group else 0)
        
        elif color_by == 'belief':
            # Bestimme die wichtigste Überzeugung im Netzwerk
            all_beliefs = set()
            for agent in self.agents.values():
                all_beliefs.update(agent.beliefs.keys())
                
            # Wähle eine repräsentative Überzeugung
            if all_beliefs:
                rep_belief = list(all_beliefs)[0]
                
                colors = []
                for agent_id in self.social_network.nodes():
                    agent = self.agents[agent_id]
                    if rep_belief in agent.beliefs:
                        # Farbe basierend auf Stärke der Überzeugung
                        colors.append(agent.beliefs[rep_belief].strength)
                    else:
                        colors.append(0.0)
            else:
                colors = [0.0] * len(self.social_network.nodes())
        
        else:
            # Standardfarbe
            colors = [0.5] * len(self.social_network.nodes())
            
        # Knotengröße basierend auf Anzahl der Verbindungen
        node_size = [300 * (1 + self.social_network.degree(node)) for node in self.social_network.nodes()]
        
        # Kanten basierend auf Verbindungsstärke
        edge_widths = [2 * self.social_network[u][v]['weight'] for u, v in self.social_network.edges()]
        
        # Netzwerk zeichnen
        nodes = nx.draw_networkx_nodes(
            self.social_network, pos, 
            node_size=node_size,
            node_color=colors, 
            cmap=plt.cm.viridis, 
            alpha=0.8
        )
        
        edges = nx.draw_networkx_edges(
            self.social_network, pos,
            width=edge_widths,
            alpha=0.5
        )
        
        # Kleinere Knotenbeschriftungen
        nx.draw_networkx_labels(
            self.social_network, pos, 
            font_size=8, 
            font_family='sans-serif'
        )
        
        plt.title("Soziales Netzwerk der Agenten")
        plt.axis('off')
        plt.colorbar(nodes)
        plt.tight_layout()
        plt.show()
    
    def visualize_belief_distribution(self, belief_name: str):
        """Visualisiert die Verteilung einer bestimmten Überzeugung in der Gesellschaft."""
        belief_values = []
        agent_ids = []
        
        for agent_id, agent in self.agents.items():
            if belief_name in agent.beliefs:
                belief_values.append(agent.beliefs[belief_name].strength)
                agent_ids.append(agent_id)
                
        if not belief_values:
            print(f"Keine Agenten mit der Überzeugung '{belief_name}' gefunden.")
            return
            
        plt.figure(figsize=(12, 6))
        
        # Histogramm
        plt.subplot(1, 2, 1)
        plt.hist(belief_values, bins=10, alpha=0.7, color='skyblue')
        plt.title(f"Verteilung von '{belief_name}'")
        plt.xlabel("Überzeugungsstärke")
        plt.ylabel("Anzahl der Agenten")
        
        # Balkendiagramm für einzelne Agenten
        plt.subplot(1, 2, 2)
        y_pos = np.arange(len(agent_ids))
        
        colors = plt.cm.viridis(np.array(belief_values))
        plt.barh(y_pos, belief_values, alpha=0.8, color=colors)
        plt.yticks(y_pos, agent_ids)
        plt.xlabel("Überzeugungsstärke")
        plt.title("Überzeugungsstärke nach Agent")
        
        plt.tight_layout()
        plt.show()
    
    def visualize_belief_evolution(self, results_analysis: Dict, belief_name: str, 
                                 agent_ids: List[str] = None):
        """
        Visualisiert die Entwicklung einer Überzeugung über die Zeit.
        
        Args:
            results_analysis: Analyse-Ergebnisse (von analyze_results)
            belief_name: Name der zu visualisierenden Überzeugung
            agent_ids: Liste der Agenten-IDs (None für alle Agenten)
        """
        if "belief_evolution" not in results_analysis:
            print("Keine Daten zur Überzeugungsentwicklung gefunden.")
            return
            
        belief_evolution = results_analysis["belief_evolution"]
        
        # Wenn keine Agenten angegeben, alle nehmen
        if agent_ids is None:
            agent_ids = list(belief_evolution.keys())
            
        # Nur Agenten mit der gesuchten Überzeugung
        agent_ids = [agent_id for agent_id in agent_ids 
                  if agent_id in belief_evolution and 
                  belief_name in belief_evolution[agent_id]]
                  
        if not agent_ids:
            print(f"Keine Agenten mit der Überzeugung '{belief_name}' gefunden.")
            return
            
        plt.figure(figsize=(12, 6))
        
        for agent_id in agent_ids:
            strengths = belief_evolution[agent_id][belief_name]
            plt.plot(strengths, label=agent_id)
            
        plt.title(f"Entwicklung der Überzeugung '{belief_name}' über die Zeit")
        plt.xlabel("Simulationsschritte")
        plt.ylabel("Überzeugungsstärke")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def visualize_polarization(self, results_analysis: Dict):
        """
        Visualisiert die Polarisierung über die Zeit.
        
        Args:
            results_analysis: Analyse-Ergebnisse (von analyze_results)
        """
        if "polarization_metrics" not in results_analysis:
            print("Keine Polarisierungsmetriken gefunden.")
            return
            
        polarization_data = results_analysis["polarization_metrics"]
        
        if not polarization_data:
            print("Keine Polarisierungsdaten verfügbar.")
            return
            
        # Alle Überzeugungen in den Daten finden
        all_beliefs = set()
        for step_data in polarization_data:
            all_beliefs.update(step_data.keys())
            
        # Für jede Überzeugung ein Diagramm erstellen
        plt.figure(figsize=(15, 10))
        rows = int(np.ceil(len(all_beliefs) / 2))
        
        for i, belief_name in enumerate(sorted(all_beliefs)):
            plt.subplot(rows, 2, i+1)
            
            bimodality = []
            variance = []
            
            for step_data in polarization_data:
                if belief_name in step_data:
                    bimodality.append(step_data[belief_name]["bimodality"])
                    variance.append(step_data[belief_name]["variance"])
                else:
                    bimodality.append(np.nan)
                    variance.append(np.nan)
                    
            steps = np.arange(len(polarization_data))
            
            plt.plot(steps, bimodality, label="Bimodalität", color="blue")
            plt.plot(steps, variance, label="Varianz", color="red")
            plt.title(f"Polarisierung für '{belief_name}'")
            plt.xlabel("Simulationsschritte")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.show()

    def save_simulation(self, filename: str):
        """Speichert die Simulation in einer Datei."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_simulation(cls, filename: str) -> 'EthicalSociety':
        """Lädt eine Simulation aus einer Datei."""
        with open(filename, 'rb') as f:
            return pickle.load(f)


def create_example_society() -> EthicalSociety:
    """Erstellt eine Beispielgesellschaft für Demonstrationen."""
    society = EthicalSociety()
    
    # Überzeugungsvorlagen hinzufügen
    society.add_belief_template(
        "individual_freedom", "Freiheit", 
        {
            "government_control": (0.7, -1),
            "free_speech": (0.8, 1),
            "free_market": (0.6, 1)
        }
    )
    
    society.add_belief_template(
        "government_control", "Freiheit",
        {
            "individual_freedom": (0.7, -1),
            "social_welfare": (0.6, 1),
            "market_regulation": (0.5, 1)
        }
    )
    
    society.add_belief_template(
        "free_speech", "Freiheit",
        {
            "individual_freedom": (0.8, 1),
            "hate_speech_laws": (0.7, -1)
        }
    )
    
    society.add_belief_template(
        "hate_speech_laws", "Gerechtigkeit",
        {
            "free_speech": (0.7, -1),
            "equality": (0.6, 1)
        }
    )
    
    society.add_belief_template(
        "equality", "Gerechtigkeit",
        {
            "meritocracy": (0.5, -1),
            "social_welfare": (0.6, 1)
        }
    )
    
    society.add_belief_template(
        "meritocracy", "Wirtschaft",
        {
            "equality": (0.5, -1),
            "free_market": (0.7, 1)
        }
    )
    
    society.add_belief_template(
        "free_market", "Wirtschaft",
        {
            "individual_freedom": (0.6, 1),
            "market_regulation": (0.8, -1),
            "meritocracy": (0.7, 1)
        }
    )
    
    society.add_belief_template(
        "market_regulation", "Wirtschaft",
        {
            "free_market": (0.8, -1),
            "government_control": (0.5, 1),
            "social_welfare": (0.6, 1)
        }
    )
    
    society.add_belief_template(
        "social_welfare", "Wohlfahrt",
        {
            "equality": (0.6, 1),
            "government_control": (0.6, 1),
            "market_regulation": (0.6, 1)
        }
    )
    
    society.add_belief_template(
        "traditional_values", "Tradition",
        {
            "progressivism": (0.8, -1),
            "religiosity": (0.7, 1)
        }
    )
    
    society.add_belief_template(
        "progressivism", "Fortschritt",
        {
            "traditional_values": (0.8, -1),
            "science_trust": (0.6, 1)
        }
    )
    
    society.add_belief_template(
        "religiosity", "Religion",
        {
            "traditional_values": (0.7, 1),
            "science_trust": (0.5, -1)
        }
    )
    
    society.add_belief_template(
        "science_trust", "Wissenschaft",
        {
            "progressivism": (0.6, 1),
            "religiosity": (0.5, -1)
        }
    )
    
    # Archetyp-Agenten erstellen
    
    # Libertärer Agent
    libertarian = society.generate_random_agent("libertarian_archetype")
    # Überzeugungen anpassen
    libertarian.beliefs["individual_freedom"].update_strength(0.9)
    libertarian.beliefs["free_market"].update_strength(0.85)
    if "government_control" in libertarian.beliefs:
        libertarian.beliefs["government_control"].update_strength(0.2)
    if "market_regulation" in libertarian.beliefs:
        libertarian.beliefs["market_regulation"].update_strength(0.15)
    society.add_agent(libertarian)
    
    # Progressive Agent
    progressive = society.generate_random_agent("progressive_archetype")
    progressive.beliefs["progressivism"].update_strength(0.85)
    progressive.beliefs["equality"].update_strength(0.8)
    if "science_trust" in progressive.beliefs:
        progressive.beliefs["science_trust"].update_strength(0.9)
    if "traditional_values" in progressive.beliefs:
        progressive.beliefs["traditional_values"].update_strength(0.2)
    society.add_agent(progressive)
    
    # Konservativer Agent
    conservative = society.generate_random_agent("conservative_archetype")
    conservative.beliefs["traditional_values"].update_strength(0.8)
    if "religiosity" in conservative.beliefs:
        conservative.beliefs["religiosity"].update_strength(0.7)
    if "progressivism" in conservative.beliefs:
        conservative.beliefs["progressivism"].update_strength(0.3)
    society.add_agent(conservative)
    
    # Sozialistischer Agent
    socialist = society.generate_random_agent("socialist_archetype")
    socialist.beliefs["equality"].update_strength(0.9)
    socialist.beliefs["social_welfare"].update_strength(0.85)
    if "market_regulation" in socialist.beliefs:
        socialist.beliefs["market_regulation"].update_strength(0.8)
    if "free_market" in socialist.beliefs:
        socialist.beliefs["free_market"].update_strength(0.2)
    society.add_agent(socialist)
    
    # Archetypen in Gruppen einteilen
    society.add_group("Libertarians", ["libertarian_archetype"])
    society.add_group("Progressives", ["progressive_archetype"])
    society.add_group("Conservatives", ["conservative_archetype"])
    society.add_group("Socialists", ["socialist_archetype"])
    
    # Mehr Agenten erstellen
    archetypes = [libertarian, progressive, conservative, socialist]
    new_agents = society.generate_population_cluster(archetypes, 20, (0.7, 0.9))
    
    # Neue Agenten hinzufügen
    for agent in new_agents:
        society.add_agent(agent)
        
        # Gruppenzugehörigkeit basierend auf Ähnlichkeit
        similarities = []
        for archetype in archetypes:
            sim = society._calculate_belief_similarity(agent, archetype)
            similarities.append((archetype.agent_id.split("_")[0], sim))
        
        # Zur ähnlichsten Gruppe hinzufügen
        most_similar = max(similarities, key=lambda x: x[1])
        society.add_group(f"{most_similar[0].capitalize()}s", [agent.agent_id])
    
    # Soziales Netzwerk generieren
    society.generate_realistic_social_network(connection_density=0.1, 
                                            group_connection_boost=0.4,
                                            belief_similarity_factor=0.6)
    
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
            "allow_speech": {"risks": 0.7},
            "restrict_speech": {"risks": 0.5},
            "monitor_but_allow": {"risks": 0.3}
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
            "deregulate": {"risks": 0.8},
            "strict_regulation": {"risks": 0.4},
            "moderate_oversight": {"risks": 0.5}
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
        }
    )
    society.add_scenario(welfare_scenario)
    
    return society


# Beispiel für einen Testlauf
def run_demo():
    # Beispielgesellschaft erstellen
    society = create_example_society()
    
    # Visualisierung des Überzeugungsnetzwerks eines Agenten
    print("Visualisierung des Überzeugungsnetzwerks des libertären Archetyps:")
    society.visualize_belief_network("libertarian_archetype")
    
    # Visualisierung des sozialen Netzwerks
    print("\nVisualisierung des sozialen Netzwerks:")
    society.visualize_social_network(color_by='group')
    
    # Beispiel für die Analyse einer Überzeugung
    print("\nVerteilung der Überzeugung 'individual_freedom':")
    society.visualize_belief_distribution("individual_freedom")
    
    # Simulation durchführen
    print("\nSimulation wird ausgeführt...")
    results = society.run_simulation(num_steps=20, 
                                  scenario_probability=0.3,
                                  social_influence_probability=0.4)
    
    # Ergebnisse analysieren
    print("\nAnalyse der Ergebnisse...")
    analysis = society.analyze_results(results)
    
    # Entwicklung der Überzeugungen visualisieren
    print("\nEntwicklung der Überzeugung 'free_speech':")
    society.visualize_belief_evolution(analysis, "free_speech")
    
    # Polarisierung visualisieren
    print("\nPolarisierung über die Zeit:")
    society.visualize_polarization(analysis)
    
    return society, results, analysis


if __name__ == "__main__":
    society, results, analysis = run_demo()
