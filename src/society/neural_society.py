import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import pickle
import os
from scipy.stats import entropy
from typing import List, Dict, Tuple, Optional, Set, Union, Callable

# Import aus anderen Modulen
from neural_types import NeuralProcessingType
from cognitive_architecture import CognitiveArchitecture
from beliefs import NeuralEthicalBelief
from scenarios import EthicalScenario
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
    
    # === ERWEITERTE SOZIALE DYNAMIK ===
    
    def update_social_dynamics(self, scenario: EthicalScenario = None) -> Dict[str, float]:
        """
        Führt einen Schritt der sozialen Dynamik durch.
        
        Args:
            scenario: Optionales Szenario für kontextuelle Interaktionen
            
        Returns:
            Dictionary mit Metriken der sozialen Dynamik
        """
        dynamics_metrics = {}
        
        # Phase 1: Soziales Lernen zwischen verbundenen Agenten
        social_learning_changes = self._perform_social_learning()
        dynamics_metrics["social_learning_intensity"] = np.mean(list(social_learning_changes.values())) if social_learning_changes else 0.0
        
        # Phase 2: Gruppenbildung und -auflösung
        group_changes = self._update_group_dynamics()
        dynamics_metrics["group_formation_rate"] = group_changes.get("formations", 0)
        dynamics_metrics["group_dissolution_rate"] = group_changes.get("dissolutions", 0)
        
        # Phase 3: Meinungsführerschaft und Einfluss
        leadership_changes = self._update_opinion_leadership()
        dynamics_metrics["leadership_changes"] = len(leadership_changes)
        
        # Phase 4: Netzwerk-Evolution
        network_changes = self._evolve_social_network()
        dynamics_metrics["network_edge_changes"] = network_changes.get("edge_changes", 0)
        
        # Phase 5: Polarisierung und Konsensbildung
        polarization_metrics = self._measure_polarization()
        dynamics_metrics.update(polarization_metrics)
        
        # Phase 6: Kulturelle Übertragung
        cultural_transmission = self._perform_cultural_transmission()
        dynamics_metrics["cultural_transmission_rate"] = cultural_transmission
        
        self.current_step += 1
        self.validation_metrics["polarization_history"].append(polarization_metrics)
        
        return dynamics_metrics
    
    def _perform_social_learning(self) -> Dict[str, float]:
        """Führt soziales Lernen zwischen allen verbundenen Agenten durch."""
        learning_changes = {}
        
        # Für jeden Agenten mit seinen direkten Nachbarn lernen
        for agent_id, agent in self.agents.items():
            neighbors = list(self.social_network.neighbors(agent_id))
            neighbor_agents = [self.agents[nid] for nid in neighbors if nid in self.agents]
            
            if neighbor_agents:
                # Erweiterte soziale Lernmechanismen anwenden
                changes = agent.advanced_social_learning(neighbor_agents)
                if changes:
                    total_change = sum(abs(change) for change in changes.values())
                    learning_changes[agent_id] = total_change
        
        return learning_changes
    
    def _update_group_dynamics(self) -> Dict[str, int]:
        """Aktualisiert Gruppenbildung und -auflösung."""
        group_changes = {"formations": 0, "dissolutions": 0}
        
        # Suche nach neuen Gruppenbildungen basierend auf Ähnlichkeit
        agent_ids = list(self.agents.keys())
        
        for i, agent_id1 in enumerate(agent_ids):
            for agent_id2 in agent_ids[i+1:]:
                agent1 = self.agents[agent_id1]
                agent2 = self.agents[agent_id2]
                
                # Berechne Ähnlichkeit zwischen Agenten
                similarity = agent1._calculate_agent_similarity(agent2)
                
                # Wenn sehr ähnlich und noch nicht in einer Gruppe, bilde neue Gruppe
                if similarity > 0.8:
                    shared_groups = set(agent1.group_identities.keys()).intersection(
                        set(agent2.group_identities.keys()))
                    
                    if not shared_groups:
                        # Neue Gruppe basierend auf Ähnlichkeit bilden
                        new_group_name = f"dynamic_group_{self.current_step}_{len(self.groups)}"
                        self.add_group(new_group_name, [agent_id1, agent_id2], 0.7, 0.9)
                        group_changes["formations"] += 1
        
        # Prüfe Gruppenauflösungen (wenn Mitglieder zu unterschiedlich werden)
        groups_to_dissolve = []
        for group_name, member_ids in self.groups.items():
            if len(member_ids) < 2:
                continue
                
            # Berechne durchschnittliche Intragruppen-Ähnlichkeit
            similarities = []
            member_list = list(member_ids)  # Convert set to list for indexing
            for i, agent_id1 in enumerate(member_list):
                for agent_id2 in member_list[i+1:]:
                    if agent_id1 in self.agents and agent_id2 in self.agents:
                        agent1 = self.agents[agent_id1]
                        agent2 = self.agents[agent_id2]
                        similarity = agent1._calculate_agent_similarity(agent2)
                        similarities.append(similarity)
            
            if similarities and np.mean(similarities) < 0.4:
                groups_to_dissolve.append(group_name)
        
        # Löse Gruppen auf
        for group_name in groups_to_dissolve:
            if group_name in self.groups:
                del self.groups[group_name]
                group_changes["dissolutions"] += 1
        
        return group_changes
    
    def _update_opinion_leadership(self) -> List[str]:
        """Aktualisiert die Meinungsführerschaft basierend auf den aktuellen Überzeugungen."""
        leadership_changes = []
        
        # Finde führende Agenten in jeder Gruppe (oder insgesamt)
        for group_name, member_ids in self.groups.items():
            if len(member_ids) < 2:
                continue
            
            # Berechne den Einfluss jedes Mitglieds in der Gruppe
            influences = {}
            for agent_id in member_ids:
                if agent_id in self.agents:
                    agent = self.agents[agent_id]
                    # Einfluss als Funktion der Überzeugungsstärke und -gewissheit
                    total_influence = sum(belief.strength * belief.certainty for belief in agent.beliefs.values())
                    influences[agent_id] = total_influence
            
            # Finde den oder die führenden Agenten mit dem höchsten Einfluss
            if influences:
                max_influence = max(influences.values())
                leaders = [aid for aid, inf in influences.items() if inf == max_influence]
                
                # Zufällige Auswahl eines Führers bei mehreren Kandidaten
                chosen_leader = np.random.choice(leaders)
                leadership_changes.append(chosen_leader)
                
                # Optional: Führer erhalten einen Identitätsbonus
                if chosen_leader in self.agents:
                    self.agents[chosen_leader].add_group_identity(group_name, 1.0)
        
        return leadership_changes
    
    def _evolve_social_network(self) -> Dict[str, int]:
        """Entwickelt das soziale Netzwerk weiter, indem neue Verbindungen basierend auf Ähnlichkeit hinzugefügt oder entfernt werden."""
        network_changes = {"edge_changes": 0}
        
        # Für alle Agentenpaare, die noch nicht verbunden sind, füge Verbindung hinzu, wenn sie ähnlich genug sind
        for agent_id1, agent1 in self.agents.items():
            for agent_id2, agent2 in self.agents.items():
                if agent_id1 != agent_id2 and not self.social_network.has_edge(agent_id1, agent_id2):
                    # Berechne Ähnlichkeit zwischen den Agenten
                    similarity = agent1._calculate_agent_similarity(agent2)
                    
                    # Wenn die Ähnlichkeit hoch genug ist, füge eine Verbindung hinzu
                    if similarity > 0.7:
                        self.add_social_connection(agent_id1, agent_id2, similarity)
                        network_changes["edge_changes"] += 1
        
        # Entferne zufällig Verbindungen, um das Netzwerk dynamisch zu halten
        for agent_id, agent in self.agents.items():
            if len(agent.social_connections) > 3:  # Maximal 3 Verbindungen beibehalten
                connections_to_remove = random.sample(agent.social_connections, len(agent.social_connections) - 3)
                for conn_id in connections_to_remove:
                    self.social_network.remove_edge(agent_id, conn_id)
                    agent.remove_social_connection(conn_id)
                    network_changes["edge_changes"] += 1
        
        return network_changes
    
    def _measure_polarization(self) -> Dict[str, float]:
        """Misst die Polarisierung innerhalb der Gesellschaft."""
        polarization_metrics = {}
        
        # Berechne die Verteilung der Überzeugungen in der Gesellschaft
        all_beliefs = np.array([belief.strength for agent in self.agents.values() for belief in agent.beliefs.values()])
        
        # Polarisierung als Entropie der Überzeugungsverteilung
        if len(all_beliefs) > 1:
            belief_entropy = entropy(np.histogram(all_beliefs, bins=10, range=(0, 1))[0], base=2)
        else:
            belief_entropy = 0
        
        polarization_metrics["belief_entropy"] = belief_entropy
        
        # Zusätzliche Polarisierungsmaße können hier hinzugefügt werden
        
        return polarization_metrics
    
    def _perform_cultural_transmission(self) -> float:
        """Führt kulturelle Übertragung zwischen Agenten durch."""
        transmission_rate = 0.0
        
        # Kulturelle Übertragung erfolgt hauptsächlich durch soziale Interaktionen
        for agent_id, agent in self.agents.items():
            neighbors = list(self.social_network.neighbors(agent_id))
            neighbor_agents = [self.agents[nid] for nid in neighbors if nid in self.agents]
            
            if neighbor_agents:
                # Auswahl eines zufälligen Nachbarn für die kulturelle Übertragung
                chosen_neighbor = np.random.choice(neighbor_agents)
                
                # Kulturelle Merkmale vergleichen und übertragen
                for belief in agent.beliefs.values():
                    if np.random.random() < 0.1:  # 10%ige Wahrscheinlichkeit für den Transfer jedes Merkmals
                        # Übertrage die Stärke der Überzeugung mit etwas Zufallsrauschen
                        transferred_strength = np.clip(belief.strength + np.random.normal(0, 0.05), 0, 1)
                        chosen_neighbor.add_belief(NeuralEthicalBelief(belief.name, belief.category, transferred_strength, belief.certainty, belief.emotional_valence))
                        transmission_rate += 0.1  # Erhöhe die Übertragungsrate
        
        return transmission_rate
    
    def validate_society(self) -> Dict[str, float]:
        """
        Führt eine Validierung der Gesellschaft durch.
        
        Returns:
            Dictionary mit Validierungsergebnissen
        """
        validation_results = {}
        
        # 1. Überprüfe die Verteilung der Überzeugungen
        all_beliefs = np.array([belief.strength for agent in self.agents.values() for belief in agent.beliefs.values()])
        if len(all_beliefs) > 1:
            belief_histogram, _ = np.histogram(all_beliefs, bins=10, range=(0, 1))
            belief_entropy = entropy(belief_histogram, base=2)
        else:
            belief_entropy = 0
        
        validation_results["belief_entropy"] = belief_entropy
        
        # 2. Analyse der Entscheidungsfindungsmuster
        decision_patterns = self._analyze_decision_patterns()
        validation_results["decision_patterns"] = decision_patterns
        
        # 3. Polarisierungsmessung
        polarization_metrics = self._measure_polarization()
        validation_results.update(polarization_metrics)
        
        # 4. Netzwerkstruktur-Analyse
        network_structure = self._analyze_network_structure()
        validation_results["network_density"] = network_structure.get("density", 0)
        validation_results["average_clustering"] = network_structure.get("average_clustering", 0)
        
        # 5. Überprüfe auf Validierungsfehler
        validation_errors = self.validation_metrics["validation_errors"]
        validation_results["validation_errors"] = validation_errors
        
        return validation_results
    
    def _analyze_decision_patterns(self) -> Dict[str, float]:
        """Analysiert die Entscheidungsfindungsmuster der Agenten."""
        patterns = {}
        
        for agent_id, agent in self.agents.items():
            # Beispielmuster: Durchschnittliche Überzeugungsstärke und -gewissheit
            avg_strength = np.mean([belief.strength for belief in agent.beliefs.values()])
            avg_certainty = np.mean([belief.certainty for belief in agent.beliefs.values()])
            
            patterns[agent_id] = {"avg_strength": avg_strength, "avg_certainty": avg_certainty}
        
        return patterns
    
    def _analyze_network_structure(self) -> Dict[str, float]:
        """Analysiert die Struktur des sozialen Netzwerks."""
        density = nx.density(self.social_network)
        average_clustering = nx.average_clustering(self.social_network)
        
        return {"density": density, "average_clustering": average_clustering}
    
    def save_society_state(self, file_path: str):
        """Speichert den aktuellen Zustand der Gesellschaft in einer Datei."""
        with open(file_path, "wb") as file:
            pickle.dump(self, file)
    
    @classmethod
    def load_society_state(cls, file_path: str) -> "NeuralEthicalSociety":
        """Lädt den Zustand einer Gesellschaft aus einer Datei."""
        with open(file_path, "rb") as file:
            society = pickle.load(file)
            return society
    
    def run_simulation_step(self):
        """Führt einen vollständigen Simulationsschritt für die Gesellschaft durch."""
        # 1. Aktualisiere die sozialen Dynamiken
        self.update_social_dynamics()
        
        # 2. Führe individuelle Entscheidungen und Lernprozesse der Agenten durch
        for agent in self.agents.values():
            agent.decide_and_learn()
        
        # 3. Optional: Periodische Validierung der Gesellschaft
        if self.current_step % 10 == 0:
            self.validate_society()
        
    def run_full_simulation(self, num_steps: int, validation_interval: int = 10):
        """
        Führt eine vollständige Simulation über eine gegebene Anzahl von Schritten durch.
        
        Args:
            num_steps: Anzahl der Simulationsschritte
            validation_interval: Intervall für die Validierung der Gesellschaft (in Schritten)
        """
        for step in range(num_steps):
            self.run_simulation_step()
            
            if step % validation_interval == 0:
                validation_results = self.validate_society()
                print(f"Validierungsergebnisse nach Schritt {step}: {validation_results}")