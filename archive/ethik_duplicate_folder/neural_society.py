import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import pickle
import os
from scipy.stats import entropy
from typing import List, Dict, Tuple, Optional, Set, Union, Callable

# Import aus anderen Modulen
from neural_core import NeuralProcessingType, CognitiveArchitecture, NeuralEthicalBelief, EthicalScenario
from neural_agent import NeuralEthicalAgent

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
        if group_name not in self.groups:
            self.groups[group_name] = set()
            
        self.groups[group_name].update(agent_ids)
        
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
    
    # Weitere Methoden für die Gesellschaftssimulation...