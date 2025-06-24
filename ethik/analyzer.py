import numpy as np
import pandas as pd
import copy
import logging
from typing import List, Dict, Any, Optional

# Importiere notwendige Klassen aus dem ethik Paket
from .neural_types import NeuralProcessingType
# Annahme: NeuralEthicalSociety und NeuralEthicalAgent werden für Typ-Hinweise benötigt
# und sind im ethik-Paket verfügbar (z.B. ethik.neural_society, ethik.agents)
from .neural_society import NeuralEthicalSociety
from .agents import NeuralEthicalAgent
from .constants import ANALYSIS_BELIEF_CLUSTER_THRESHOLD, ANALYSIS_CLUSTER_MIN_SIZE, ANALYSIS_STRONG_BELIEF_THRESHOLD

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
        # Bestimme die Anzahl der Schritte aus einer der Ergebnislisten, die pro Schritt Daten enthält.
        # 'agent_states' ist ein guter Kandidat, da er für jeden Schritt einen Eintrag haben sollte.
        agent_states_history = results.get("agent_states", [])
        self.num_steps = len(agent_states_history) if agent_states_history else 0

        self.agent_ids = list(society.agents.keys())

        # Berechne einige Metriken direkt bei der Initialisierung
        self.final_agent_states = agent_states_history[-1] if agent_states_history else {}
        self.belief_evolution_df = self._create_belief_evolution_dataframe()
        self.final_polarization = self._calculate_final_polarization()


    def _create_belief_evolution_dataframe(self) -> Optional[pd.DataFrame]:
        """Erstellt ein DataFrame zur Verfolgung der Belief-Entwicklung."""
        records = []
        agent_states_history = self.results.get("agent_states", [])
        if not agent_states_history: return None

        for step, step_states in enumerate(agent_states_history):
            for agent_id, state in step_states.items():
                 if agent_id in self.society.agents:
                     agent_instance = self.society.agents[agent_id]
                     agent_arch = agent_instance.cognitive_architecture
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
         decision_counts: Dict[str, Dict[str, int]] = {}
         for step_decisions_list in self.results.get("decisions", []):
             # decisions kann eine Liste von Dictionaries sein, wenn aus Ensemble kombiniert
             # oder direkt ein Dictionary pro Schritt.
             actual_step_decisions = step_decisions_list if isinstance(step_decisions_list, dict) else {}
             if not isinstance(step_decisions_list, dict) and isinstance(step_decisions_list, list) and len(step_decisions_list) > 0:
                 # Nehme an, es ist eine Liste von step_decision dicts (wenn nicht weiter genestet)
                  actual_step_decisions = step_decisions_list[0] if len(step_decisions_list) > 0 and isinstance(step_decisions_list[0], dict) else {}


             for agent_id, decision in actual_step_decisions.items():
                 scen_id = decision.get("scenario_id")
                 option = decision.get("chosen_option")
                 if scen_id and option:
                      if scenario_id is not None and scen_id != scenario_id:
                           continue
                      if scen_id not in decision_counts: decision_counts[scen_id] = {}
                      decision_counts[scen_id][option] = decision_counts[scen_id].get(option, 0) + 1
         return decision_counts

    def get_cognitive_style_decision_summary(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        """Zählt Entscheidungen gruppiert nach primärem kognitiven Stil."""
        style_counts: Dict[str, Dict[str, Dict[str, int]]] = {style: {} for style in NeuralProcessingType.ALL_TYPES}

        for step_decisions_list in self.results.get("decisions", []):
            actual_step_decisions = step_decisions_list if isinstance(step_decisions_list, dict) else {}
            if not isinstance(step_decisions_list, dict) and isinstance(step_decisions_list, list) and len(step_decisions_list) > 0:
                 actual_step_decisions = step_decisions_list[0] if len(step_decisions_list) > 0 and isinstance(step_decisions_list[0], dict) else {}

            for agent_id, decision in actual_step_decisions.items():
                 scen_id = decision.get("scenario_id")
                 option = decision.get("chosen_option")
                 if scen_id and option and agent_id in self.society.agents:
                      agent = self.society.agents[agent_id]
                      style = agent.cognitive_architecture.primary_processing
                      if scen_id not in style_counts[style]: style_counts[style][scen_id] = {}
                      style_counts[style][scen_id][option] = style_counts[style][scen_id].get(option, 0) + 1
        return {s: data for s, data in style_counts.items() if data}


    def _calculate_final_polarization(self) -> Optional[Dict[str, Dict[str, float]]]:
        """Berechnet Polarisierungsmetriken für den finalen Zustand."""
        if not self.final_agent_states: return None
        temp_agents = {}
        initial_agents = self.society.agents
        for agent_id, state in self.final_agent_states.items():
            if agent_id in initial_agents:
                # Erstelle eine Kopie des ursprünglichen Agenten, um seine Struktur beizubehalten
                temp_agent = copy.deepcopy(initial_agents[agent_id])
                # Aktualisiere nur die Belief-Zustände aus den Simulationsergebnissen
                temp_agent.beliefs = {} # Alte Belief-Objekte entfernen
                for b_name, b_data in state.get("beliefs", {}).items():
                    # Erstelle neue Belief-Objekte mit den Werten aus dem state
                    # Annahme: NeuralEthicalBelief ist importiert oder hier verfügbar
                    from .beliefs import NeuralEthicalBelief # Sicherstellen, dass es verfügbar ist
                    temp_belief = NeuralEthicalBelief(name=b_name, category="unknown", # Kategorie ist hier nicht kritisch
                                                      initial_strength=b_data.get("strength", 0.5),
                                                      certainty=b_data.get("certainty", 0.5),
                                                      emotional_valence=b_data.get("valence", 0.0))
                    temp_belief.activation = b_data.get("activation", 0.0)
                    temp_agent.beliefs[b_name] = temp_belief
                temp_agents[agent_id] = temp_agent

        if not temp_agents : return None
        return self.society._calculate_polarization(temp_agents)


    def get_polarization_trend(self, metric: str = 'bimodality') -> Optional[pd.DataFrame]:
        """Gibt die Entwicklung der Polarisierung über die Zeit zurück."""
        polarization_history = []
        agent_states_history = self.results.get("agent_states", [])
        if not agent_states_history: return None

        initial_agents_structure = self.society.agents

        for step, step_states in enumerate(agent_states_history):
            current_step_temp_agents = {}
            for agent_id, state_data in step_states.items():
                if agent_id in initial_agents_structure:
                    agent_shell = copy.deepcopy(initial_agents_structure[agent_id])
                    agent_shell.beliefs = {} # Leere Beliefs, um sie mit Step-Daten zu füllen
                    for b_name, b_data in state_data.get("beliefs", {}).items():
                        from .beliefs import NeuralEthicalBelief
                        agent_shell.beliefs[b_name] = NeuralEthicalBelief(
                            name=b_name, category="unknown", # Kategorie für Analyse hier nicht primär
                            initial_strength=b_data.get("strength",0.5),
                            certainty=b_data.get("certainty",0.5),
                            emotional_valence=b_data.get("valence",0.0)
                        )
                    current_step_temp_agents[agent_id] = agent_shell

            if not current_step_temp_agents: continue

            step_polarization = self.society._calculate_polarization(current_step_temp_agents) # Ruft Methode der Society auf
            if step_polarization:
                for belief, metrics_dict in step_polarization.items():
                     polarization_history.append({
                         "step": step,
                         "belief": belief,
                         "metric": metric, # Die angeforderte Metrik
                         "value": metrics_dict.get(metric) # Wert der Metrik
                     })

        if not polarization_history: return None
        df = pd.DataFrame(polarization_history)
        return df[df['value'].notna()]

    def get_ensemble_variance(self, result_type: str = "belief_strength_variance") -> Optional[pd.DataFrame]:
        """Extrahiert die Ensemble-Varianz als DataFrame."""
        variance_data = self.results.get("ensemble_stats", {}).get(result_type, [])
        if not variance_data: return None

        records = []
        for step, step_variance_data in enumerate(variance_data):
             # step_variance_data ist ein Dict {agent_id: {belief_name: variance}}
             for agent_id, belief_variances_dict in step_variance_data.items():
                 for belief, variance_value in belief_variances_dict.items():
                      records.append({
                          "step": step,
                          "agent_id": agent_id,
                          "belief": belief,
                          "variance": variance_value
                      })
        if not records: return None
        return pd.DataFrame(records)

    def identify_belief_clusters(self, step: int = -1) -> List[Dict[str, Any]]:
        """Identifiziert Agenten-Cluster basierend auf Belief-Ähnlichkeit zu einem bestimmten Zeitpunkt."""
        agent_states_history = self.results.get("agent_states", [])
        if not agent_states_history: return []

        if step == -1:
            agents_to_cluster_snapshot = self.final_agent_states
        elif 0 <= step < len(agent_states_history):
            agents_to_cluster_snapshot = agent_states_history[step]
        else:
            logging.error(f"Ungültiger Schritt {step} für Cluster-Analyse. Max Schritt: {len(agent_states_history)-1}")
            return []

        if not agents_to_cluster_snapshot: return []

        temp_agents_for_clustering = {}
        initial_agents_structure = self.society.agents
        for agent_id, state_data in agents_to_cluster_snapshot.items():
            if agent_id in initial_agents_structure:
                agent_shell = copy.deepcopy(initial_agents_structure[agent_id])
                agent_shell.beliefs = {}
                for b_name, b_data in state_data.get("beliefs", {}).items():
                    from .beliefs import NeuralEthicalBelief
                    agent_shell.beliefs[b_name] = NeuralEthicalBelief(
                        name=b_name, category="unknown",
                        initial_strength=b_data.get("strength",0.5)
                    )
                temp_agents_for_clustering[agent_id] = agent_shell

        if len(temp_agents_for_clustering) < 2: return []

        agent_ids_list = list(temp_agents_for_clustering.keys())
        similarity_matrix = np.zeros((len(agent_ids_list), len(agent_ids_list)))
        for i, id1 in enumerate(agent_ids_list):
            for j, id2 in enumerate(agent_ids_list):
                if i == j: similarity_matrix[i, j] = 1.0
                elif i < j:
                    sim = self.society._calculate_belief_similarity(temp_agents_for_clustering[id1], temp_agents_for_clustering[id2])
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim

        clusters = []
        processed_agents_set = set()

        for i, agent_id_i in enumerate(agent_ids_list):
            if agent_id_i in processed_agents_set: continue

            current_cluster_members = {agent_id_i}
            # Finde Agenten, die ähnlich zu agent_id_i sind
            for j, agent_id_j in enumerate(agent_ids_list):
                if i == j or agent_id_j in processed_agents_set: continue
                if similarity_matrix[i, j] > ANALYSIS_BELIEF_CLUSTER_THRESHOLD:
                    current_cluster_members.add(agent_id_j)

            if len(current_cluster_members) >= ANALYSIS_CLUSTER_MIN_SIZE:
                # Verfeinere Cluster: Füge Agenten hinzu, die zu *irgendeinem* Mitglied des aktuellen Clusters ähnlich sind
                # Dies ist eine einfache Form des iterativen Hinzufügens
                newly_added_to_cluster = True
                while newly_added_to_cluster:
                    newly_added_to_cluster = False
                    potential_additions = []
                    for existing_member_id_idx, existing_member_id in enumerate(agent_ids_list):
                        if existing_member_id not in current_cluster_members or existing_member_id in processed_agents_set : continue

                        member_idx_in_matrix = agent_ids_list.index(existing_member_id)

                        for prospect_id_idx, prospect_id in enumerate(agent_ids_list):
                            if prospect_id in current_cluster_members or prospect_id in processed_agents_set : continue

                            if similarity_matrix[member_idx_in_matrix, prospect_id_idx] > ANALYSIS_BELIEF_CLUSTER_THRESHOLD:
                                potential_additions.append(prospect_id)

                    if potential_additions:
                        for pa_id in potential_additions:
                            if pa_id not in current_cluster_members:
                                current_cluster_members.add(pa_id)
                                newly_added_to_cluster = True
                                processed_agents_set.add(pa_id) # Markiere als verarbeitet, sobald zu einem Cluster hinzugefügt

                if len(current_cluster_members) >= ANALYSIS_CLUSTER_MIN_SIZE :
                    processed_agents_set.update(current_cluster_members)

                    # Berechne durchschnittliche Ähnlichkeit
                    cluster_sim_values = []
                    member_list_for_sim = list(current_cluster_members)
                    for k_idx_loop in range(len(member_list_for_sim)):
                        for l_idx_loop in range(k_idx_loop + 1, len(member_list_for_sim)):
                            original_k_idx = agent_ids_list.index(member_list_for_sim[k_idx_loop])
                            original_l_idx = agent_ids_list.index(member_list_for_sim[l_idx_loop])
                            cluster_sim_values.append(similarity_matrix[original_k_idx, original_l_idx])
                    avg_similarity_in_cluster = np.mean(cluster_sim_values) if cluster_sim_values else 1.0

                    defining_beliefs_for_cluster = self._find_defining_beliefs(list(current_cluster_members), temp_agents_for_clustering)

                    clusters.append({
                        "agents": list(current_cluster_members),
                        "size": len(current_cluster_members),
                        "average_similarity": avg_similarity_in_cluster,
                        "defining_beliefs": defining_beliefs_for_cluster,
                        "step": step if step != -1 else self.num_steps -1
                    })

        clusters.sort(key=lambda x: x["size"], reverse=True)
        return clusters

    def _find_defining_beliefs(self, cluster_agent_ids: List[str], agent_snapshots: Dict[str, NeuralEthicalAgent]) -> Dict[str, float]:
        """Identifiziert Beliefs, die für einen Cluster charakteristisch sind."""
        belief_strengths_in_cluster: Dict[str, List[float]] = {}
        for agent_id_in_cluster in cluster_agent_ids:
            if agent_id_in_cluster in agent_snapshots:
                 agent_snapshot = agent_snapshots[agent_id_in_cluster]
                 for belief_name_key, belief_obj in agent_snapshot.beliefs.items():
                     if belief_name_key not in belief_strengths_in_cluster:
                         belief_strengths_in_cluster[belief_name_key] = []
                     belief_strengths_in_cluster[belief_name_key].append(belief_obj.strength)

        defining_beliefs_map: Dict[str, float] = {}
        for belief_name_key, strengths_list in belief_strengths_in_cluster.items():
            # Belief muss bei einem signifikanten Anteil der Cluster-Mitglieder vorhanden sein
            if len(strengths_list) >= len(cluster_agent_ids) * 0.75:
                mean_strength_val = np.mean(strengths_list)
                # Charakteristisch, wenn stark ausgeprägt (hoch oder niedrig im Vergleich zum Mittelwert 0.5)
                if mean_strength_val > ANALYSIS_STRONG_BELIEF_THRESHOLD or \
                   mean_strength_val < (1.0 - ANALYSIS_STRONG_BELIEF_THRESHOLD):
                     defining_beliefs_map[belief_name_key] = mean_strength_val

        return dict(sorted(defining_beliefs_map.items(), key=lambda item: abs(item[1] - 0.5), reverse=True))
