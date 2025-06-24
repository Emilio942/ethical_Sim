import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Union, Callable

# Import from project modules
from .neural_types import NeuralProcessingType
from .cognitive_architecture import CognitiveArchitecture
from .beliefs import NeuralEthicalBelief
from .scenarios import EthicalScenario # Corrected to relative import

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