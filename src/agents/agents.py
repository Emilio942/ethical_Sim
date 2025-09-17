import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Union, Callable

# Import from project modules - try different approaches for compatibility
try:
    # Try relative imports first (when used as a package)
    from ..core.neural_types import NeuralProcessingType
    from ..core.cognitive_architecture import CognitiveArchitecture
    from ..core.beliefs import NeuralEthicalBelief
    from ..scenarios.scenarios import EthicalScenario
except ImportError:
    try:
        # Try absolute imports from the src package
        from core.neural_types import NeuralProcessingType
        from core.cognitive_architecture import CognitiveArchitecture
        from core.beliefs import NeuralEthicalBelief
        from scenarios.scenarios import EthicalScenario
    except ImportError:
        # Fallback to direct imports
        from neural_types import NeuralProcessingType
        from cognitive_architecture import CognitiveArchitecture
        from beliefs import NeuralEthicalBelief
        from scenarios import EthicalScenario

class NeuralEthicalAgent:
    """Repräsentiert einen ethischen Agenten mit neuronalen Verarbeitungsmodellen."""
    
    def __init__(self, agent_id: str, personality_traits: Dict[str, float] = None):
        """
        Initialisiert einen neuronalen ethischen Agenten.
        
        Args:
            agent_id: Eindeutige ID des Agenten
            personality_traits: Persönlichkeitsmerkmale des Agenten
            
        Raises:
            ValueError: Wenn agent_id leer oder None ist
        """
        # Input-Validierung
        if not agent_id or not isinstance(agent_id, str) or agent_id.strip() == "":
            raise ValueError("agent_id must be a non-empty string")
        
        self.agent_id = agent_id.strip()
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
        
        # === ERWEITERTE LERNMECHANISMEN ===
        
        # Reinforcement Learning Komponenten
        self.rl_system = {
            "action_values": {},  # action_type -> value estimate
            "learning_rate": 0.1 + 0.1 * self.personality_traits.get("openness", 0.5),
            "exploration_rate": 0.2 + 0.2 * self.personality_traits.get("openness", 0.5),
            "discount_factor": 0.8 + 0.15 * self.personality_traits.get("conscientiousness", 0.5),
            "experience_buffer": [],  # (state, action, reward, next_state)
            "policy_gradient": {},  # für komplexere Strategien
        }
        
        # Multi-Kriterien Entscheidungssystem
        self.decision_criteria = {
            "utility_weights": {
                "personal_benefit": 0.3,
                "social_good": 0.3, 
                "moral_alignment": 0.25,
                "risk_minimization": 0.15
            },
            "criteria_uncertainty": {  # Unsicherheit in den Gewichtungen
                "personal_benefit": 0.1,
                "social_good": 0.1,
                "moral_alignment": 0.05,
                "risk_minimization": 0.15
            },
            "trade_off_history": [],  # Aufzeichnung von Trade-off Entscheidungen
            "regret_minimization": True,  # Bereuen-Minimierung aktiviert
        }
        
        # Unsicherheitsbehandlung
        self.uncertainty_handling = {
            "ambiguity_tolerance": self.personality_traits.get("openness", 0.5),
            "information_seeking_tendency": 0.5 + 0.3 * self.personality_traits.get("conscientiousness", 0.5),
            "uncertainty_estimates": {},  # belief_name -> uncertainty
            "confidence_calibration": 0.7,  # Wie gut kalibriert sind Konfidenzschätzungen
            "epistemic_humility": 0.3 + 0.4 * self.personality_traits.get("openness", 0.5),
        }
        
        # Erweiterte soziale Lernmechanismen  
        self.social_learning = {
            "trust_network": {},  # agent_id -> trust_level
            "reputation_system": {},  # agent_id -> reputation_score
            "social_proof_sensitivity": 0.4 + 0.4 * self.personality_traits.get("agreeableness", 0.5),
            "conformity_pressure": {},  # group -> conformity_strength
            "opinion_leadership": 0.3 + 0.4 * self.personality_traits.get("extroversion", 0.5),
            "expertise_recognition": {},  # domain -> List[expert_agent_ids]
        }
        
        # Temporale Dynamik
        self.temporal_dynamics = {
            "belief_momentum": {},  # belief_name -> momentum (Trägheit von Änderungen)
            "adaptation_rate": 0.1 + 0.1 * self.personality_traits.get("openness", 0.5),
            "memory_decay": 0.02,  # Rate des Vergessens
            "recency_bias": 0.3 + 0.3 * self.personality_traits.get("neuroticism", 0.5),
            "habituation_effects": {},  # Gewöhnung an wiederholte Stimuli
        }
        
        # Meta-kognitive Bewusstheit
        self.metacognition = {
            "thinking_about_thinking": self.personality_traits.get("conscientiousness", 0.5) > 0.7,
            "strategy_monitoring": 0.4 + 0.4 * self.personality_traits.get("conscientiousness", 0.5),
            "error_detection": 0.3 + 0.5 * self.personality_traits.get("conscientiousness", 0.5),
            "cognitive_load_awareness": 0.5,
            "strategy_switching_threshold": 0.6,
        }
        
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
        # Validate that belief has required attributes
        if not hasattr(belief, 'name'):
            raise ValueError("Belief must have a 'name' attribute")
        if not hasattr(belief, 'strength'):
            raise ValueError("Belief must have a 'strength' attribute")
        if not hasattr(belief, 'certainty'):
            raise ValueError("Belief must have a 'certainty' attribute")
            
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
                        # Safely get certainty attributes with fallback values
                        belief_certainty = getattr(belief, 'certainty', 1.0)
                        other_certainty = getattr(self.beliefs[other_name], 'certainty', 1.0)
                        certainty_weight = belief_certainty * other_certainty
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
            activations = {name: getattr(belief, 'activation', 0.0) for name, belief in self.beliefs.items()}
            
            # Aktivierung verbreiten
            for belief_name, belief in self.beliefs.items():
                current_activation = getattr(belief, 'activation', 0.0)
                if current_activation > 0.1:  # Mindestschwelle für Spreading
                    for conn_name, (strength, polarity) in belief.connections.items():
                        if conn_name in self.beliefs:
                            # Aktivierung weitergeben
                            spread_activation = current_activation * strength * 0.5
                            
                            # Polarität berücksichtigen (negative Verbindungen hemmen)
                            if polarity < 0:
                                # Hemmung statt Aktivierung
                                if hasattr(self.beliefs[conn_name], 'activation'):
                                    self.beliefs[conn_name].activation *= (1.0 - spread_activation * 0.3)
                            else:
                                # Aktivierung
                                if hasattr(self.beliefs[conn_name], 'activate'):
                                    self.beliefs[conn_name].activate(spread_activation, self.current_time)
            
            # Assoziative Aktivierung (falls relevant)
            if self.cognitive_architecture.primary_processing == NeuralProcessingType.ASSOCIATIVE:
                for belief_name, belief in self.beliefs.items():
                    current_activation = getattr(belief, 'activation', 0.0)
                    if current_activation > 0.1:
                        associated_concepts = getattr(belief, 'associated_concepts', {})
                        for concept, strength in associated_concepts.items():
                            # Assoziative Aktivierung zu verbundenen Konzepten
                            for other_name, other_belief in self.beliefs.items():
                                other_concepts = getattr(other_belief, 'associated_concepts', {})
                                if concept in other_concepts:
                                    assoc_strength = strength * other_concepts[concept]
                                    assoc_activation = current_activation * assoc_strength * 0.3
                                    if hasattr(other_belief, 'activate'):
                                        other_belief.activate(assoc_activation, self.current_time)
        
        # Decay nach Aktivierung
        for belief in self.beliefs.values():
            if hasattr(belief, 'activation'):
                belief.activation *= 0.8  # 20% Decay pro Runde
    
    def make_decision(self, scenario: 'EthicalScenario', enhanced: bool = True) -> Dict[str, Union[str, float, Dict]]:
        """
        Trifft eine Entscheidung in einem ethischen Szenario.
        
        Args:
            scenario: Das ethische Szenario
            enhanced: Ob erweiterte Lernmechanismen verwendet werden sollen
        
        Returns:
            Dict mit der Entscheidung und Begründungen
        """
        if enhanced:
            return self.enhanced_make_decision(scenario)
        
        # Fallback auf die ursprüngliche Implementierung
        return self._original_make_decision(scenario)
    
    def _original_make_decision(self, scenario: 'EthicalScenario') -> Dict[str, Union[str, float, Dict]]:
        """
        Trifft eine Entscheidung in einem ethischen Szenario (ursprüngliche Implementierung).
        
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
    
    # === ERWEITERTE LERNMECHANISMEN ===
    
    def reinforcement_learn_from_outcome(self, scenario: 'EthicalScenario', 
                                       chosen_option: str, 
                                       outcome_feedback: Dict[str, float]) -> Dict[str, float]:
        """
        Implementiert Reinforcement Learning basierend auf Entscheidungsresultaten.
        
        Args:
            scenario: Das ethische Szenario
            chosen_option: Die gewählte Option  
            outcome_feedback: Feedback über das Ergebnis (verschiedene Dimensionen)
            
        Returns:
            Dictionary mit gelernten Wertschätzungen
        """
        # Gesamtbelohnung aus verschiedenen Feedback-Dimensionen berechnen
        total_reward = 0.0
        reward_components = {}
        
        for dimension, value in outcome_feedback.items():
            # Gewichtung basierend auf persönlichen Präferenzen
            if dimension == "personal_outcome":
                weight = self.decision_criteria["utility_weights"]["personal_benefit"]
            elif dimension == "social_outcome": 
                weight = self.decision_criteria["utility_weights"]["social_good"]
            elif dimension == "moral_outcome":
                weight = self.decision_criteria["utility_weights"]["moral_alignment"]
            elif dimension == "risk_outcome":
                weight = self.decision_criteria["utility_weights"]["risk_minimization"]
                value = -abs(value)  # Risiko ist negativ
            else:
                weight = 0.1  # Unbekannte Dimensionen
                
            component_reward = weight * value
            total_reward += component_reward
            reward_components[dimension] = component_reward
        
        # Erfahrung zur RL-Historie hinzufügen
        experience = {
            "scenario_type": scenario.scenario_id.split('_')[0] if '_' in scenario.scenario_id else scenario.scenario_id,
            "action": chosen_option,
            "reward": total_reward,
            "reward_components": reward_components,
            "timestamp": self.current_time
        }
        self.rl_system["experience_buffer"].append(experience)
        
        # Aktionswerte aktualisieren (Q-Learning ähnlich)
        learning_rate = self.rl_system["learning_rate"]
        scenario_type = experience["scenario_type"]
        
        # Initialisierung falls neue Aktion
        if scenario_type not in self.rl_system["action_values"]:
            self.rl_system["action_values"][scenario_type] = {}
        if chosen_option not in self.rl_system["action_values"][scenario_type]:
            self.rl_system["action_values"][scenario_type][chosen_option] = 0.0
            
        # Q-Learning Update
        old_value = self.rl_system["action_values"][scenario_type][chosen_option]
        self.rl_system["action_values"][scenario_type][chosen_option] = (
            old_value + learning_rate * (total_reward - old_value)
        )
        
        # Policy Gradient Update für Strategien
        if scenario_type not in self.rl_system["policy_gradient"]:
            self.rl_system["policy_gradient"][scenario_type] = {}
        
        # Einfacher Policy Gradient: Wahrscheinlichkeit guter Aktionen erhöhen
        if total_reward > 0:
            current_prob = self.rl_system["policy_gradient"][scenario_type].get(chosen_option, 0.5)
            self.rl_system["policy_gradient"][scenario_type][chosen_option] = min(0.9, current_prob + 0.05)
        elif total_reward < -0.1:
            current_prob = self.rl_system["policy_gradient"][scenario_type].get(chosen_option, 0.5)
            self.rl_system["policy_gradient"][scenario_type][chosen_option] = max(0.1, current_prob - 0.05)
        
        # Adaptives Lernen: Lernrate basierend auf Leistung anpassen
        recent_rewards = [exp["reward"] for exp in self.rl_system["experience_buffer"][-10:]]
        if len(recent_rewards) >= 5:
            reward_variance = np.var(recent_rewards)
            if reward_variance > 0.5:  # Hohe Varianz = mehr Exploration
                self.rl_system["exploration_rate"] = min(0.8, self.rl_system["exploration_rate"] + 0.02)
            else:  # Niedrige Varianz = weniger Exploration
                self.rl_system["exploration_rate"] = max(0.05, self.rl_system["exploration_rate"] - 0.01)
        
        # Buffer-Größe begrenzen (nur letzte 100 Erfahrungen behalten)
        if len(self.rl_system["experience_buffer"]) > 100:
            self.rl_system["experience_buffer"] = self.rl_system["experience_buffer"][-100:]
            
        return {
            "learned_value": self.rl_system["action_values"][scenario_type][chosen_option],
            "total_reward": total_reward,
            "exploration_rate": self.rl_system["exploration_rate"],
            "experience_count": len(self.rl_system["experience_buffer"])
        }
    
    def multi_criteria_decision_making(self, scenario: 'EthicalScenario', 
                                     option_evaluations: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Erweiterte multi-kriteria Entscheidungsfindung mit expliziten Trade-offs.
        
        Args:
            scenario: Das ethische Szenario
            option_evaluations: Bewertungen jeder Option auf verschiedenen Kriterien
            
        Returns:
            Finale Bewertungen für jede Option
        """
        final_scores = {}
        trade_off_analysis = {}
        
        # Kriterien-Gewichte dynamisch anpassen basierend auf Kontext
        context_weights = self.decision_criteria["utility_weights"].copy()
        
        # Anpassung basierend auf Szenario-Eigenschaften
        if hasattr(scenario, 'context') and scenario.context:
            if "high_stakes" in scenario.context:
                context_weights["risk_minimization"] *= 1.5
            if "social_pressure" in scenario.context:
                context_weights["social_good"] *= 1.3
            if "personal_values" in scenario.context:
                context_weights["moral_alignment"] *= 1.4
        
        # Normalisierung der Gewichte
        total_weight = sum(context_weights.values())
        context_weights = {k: v/total_weight for k, v in context_weights.items()}
        
        for option_name, criteria_scores in option_evaluations.items():
            weighted_score = 0.0
            option_trade_offs = {}
            
            # Gewichtete Summe mit Unsicherheitsberücksichtigung
            for criterion, score in criteria_scores.items():
                if criterion in context_weights:
                    weight = context_weights[criterion]
                    uncertainty = self.decision_criteria["criteria_uncertainty"].get(criterion, 0.1)
                    
                    # Unsicherheitsanpassung: Reduktion des Gewichts bei hoher Unsicherheit
                    adjusted_weight = weight * (1.0 - uncertainty)
                    contribution = adjusted_weight * score
                    weighted_score += contribution
                    
                    option_trade_offs[criterion] = {
                        "raw_score": score,
                        "weight": weight,
                        "uncertainty": uncertainty,
                        "contribution": contribution
                    }
            
            # Regret-Minimierung: Vermeidung von Entscheidungen mit hohem Bedauernspotential
            if self.decision_criteria["regret_minimization"]:
                # Worst-case Analyse für jedes Kriterium
                worst_case_regret = 0.0
                for criterion, score in criteria_scores.items():
                    if criterion in context_weights:
                        # Wie schlecht könnte diese Entscheidung in diesem Kriterium sein?
                        max_possible_score = 1.0  # Annahme: Scores normalisiert auf [0,1]
                        regret = max_possible_score - score
                        worst_case_regret += context_weights[criterion] * regret
                
                # Regret-Penalty
                regret_penalty = worst_case_regret * 0.2  # 20% Gewichtung
                weighted_score -= regret_penalty
                option_trade_offs["regret_penalty"] = regret_penalty
            
            # Portfolio-Theorie: Diversifikation der Entscheidungsdimensionen
            score_variance = np.var(list(criteria_scores.values())) if len(criteria_scores) > 1 else 0
            diversification_bonus = (1.0 - score_variance) * 0.1  # Bonus für ausgewogene Scores
            weighted_score += diversification_bonus
            
            final_scores[option_name] = weighted_score
            trade_off_analysis[option_name] = option_trade_offs
        
        # Trade-off Historie aktualisieren
        self.decision_criteria["trade_off_history"].append({
            "scenario": scenario.scenario_id,
            "weights_used": context_weights,
            "trade_offs": trade_off_analysis,
            "timestamp": self.current_time
        })
        
        return final_scores
    
    def handle_uncertainty_and_ambiguity(self, scenario: 'EthicalScenario', 
                                       belief_activations: Dict[str, float]) -> Dict[str, float]:
        """
        Erweiterte Behandlung von Unsicherheit und Ambiguität in Entscheidungen.
        
        Args:
            scenario: Das ethische Szenario
            belief_activations: Aktivierte Überzeugungen mit Unsicherheit
            
        Returns:
            Angepasste Aktivierungen mit Unsicherheitsbehandlung
        """
        adjusted_activations = belief_activations.copy()
        uncertainty_info = {}
        
        # Globale Unsicherheitsschätzung für das Szenario
        scenario_uncertainty = 0.0
        if hasattr(scenario, 'uncertainty_level'):
            scenario_uncertainty = scenario.uncertainty_level
        else:
            # Implizite Unsicherheitsschätzung basierend auf Szenario-Eigenschaften
            scenario_uncertainty = (
                len(scenario.options) / 10.0 +  # Mehr Optionen = mehr Unsicherheit
                getattr(scenario, 'complexity', 0.5) +  # Explizite Komplexität
                0.1  # Basis-Unsicherheit
            )
            scenario_uncertainty = min(1.0, scenario_uncertainty)
        
        # Für jede aktivierte Überzeugung Unsicherheit schätzen
        for belief_name, activation in belief_activations.items():
            if belief_name in self.beliefs:
                belief = self.beliefs[belief_name]
                
                # Epistemic Uncertainty (Wissen über Wissen)
                knowledge_gaps = 1.0 - belief.certainty
                
                # Aleatorische Uncertainty (inhärente Zufälligkeit)
                inherent_randomness = scenario_uncertainty * 0.5
                
                # Modell-Uncertainty (Unsicherheit über die richtige Entscheidungsregel)
                model_uncertainty = 0.2  # Basis-Modellunsicherheit
                
                # Gesamtunsicherheit
                total_uncertainty = knowledge_gaps + inherent_randomness + model_uncertainty
                total_uncertainty = min(1.0, total_uncertainty)
                
                # Unsicherheits-toleranz des Agenten berücksichtigen
                tolerance = self.uncertainty_handling["ambiguity_tolerance"]
                
                if total_uncertainty > tolerance:
                    # Information-seeking behavior
                    if self.uncertainty_handling["information_seeking_tendency"] > 0.6:
                        # Suche nach zusätzlichen Informationen (simuliert)
                        info_gain = random.uniform(0.1, 0.3) * self.personality_traits.get("openness", 0.5)
                        total_uncertainty = max(0.1, total_uncertainty - info_gain)
                    
                    # Vorsichtkeits-Anpassung
                    if self.cognitive_architecture.primary_processing == NeuralProcessingType.SYSTEMATIC:
                        # Systematische Denker werden vorsichtiger bei Unsicherheit
                        uncertainty_penalty = total_uncertainty * 0.3
                        adjusted_activations[belief_name] = activation * (1.0 - uncertainty_penalty)
                    elif self.cognitive_architecture.primary_processing == NeuralProcessingType.INTUITIVE:
                        # Intuitive Denker vertrauen auf Bauchgefühl trotz Unsicherheit
                        uncertainty_penalty = total_uncertainty * 0.1
                        adjusted_activations[belief_name] = activation * (1.0 - uncertainty_penalty)
                
                # Konfidenz-Kalibrierung
                calibration = self.uncertainty_handling["confidence_calibration"]
                if calibration < 0.8:  # Schlecht kalibriert
                    # Overconfidence oder Underconfidence korrigieren
                    if belief.certainty > total_uncertainty:  # Overconfident
                        belief.update_certainty(belief.certainty * 0.9)
                    elif belief.certainty < total_uncertainty * 0.5:  # Underconfident
                        belief.update_certainty(min(1.0, belief.certainty * 1.1))
                
                # Unsicherheitsschätzung speichern
                self.uncertainty_handling["uncertainty_estimates"][belief_name] = total_uncertainty
                uncertainty_info[belief_name] = {
                    "total_uncertainty": total_uncertainty,
                    "knowledge_gaps": knowledge_gaps,
                    "scenario_uncertainty": scenario_uncertainty,
                    "adjusted_activation": adjusted_activations[belief_name]
                }
        
        # Epistemic Humility: Bei hoher Gesamtunsicherheit bescheidener werden
        avg_uncertainty = np.mean(list(self.uncertainty_handling["uncertainty_estimates"].values())) if self.uncertainty_handling["uncertainty_estimates"] else 0.5
        
        if avg_uncertainty > 0.7:
            humility_factor = self.uncertainty_handling["epistemic_humility"]
            for belief_name in adjusted_activations:
                adjusted_activations[belief_name] *= (1.0 - humility_factor * 0.2)
        
        return adjusted_activations
    
    def advanced_social_learning(self, other_agents: List['NeuralEthicalAgent'], 
                               scenario_context: Dict[str, float] = None) -> Dict[str, float]:
        """
        Erweiterte soziale Lernmechanismen mit Trust, Reputation und Expertise.
        
        Args:
            other_agents: Liste anderer Agenten im sozialen Netzwerk
            scenario_context: Kontext des aktuellen Szenarios
            
        Returns:
            Dictionary mit Überzeugungsänderungen durch soziales Lernen
        """
        belief_changes = {}
        social_influences = {}
        
        # Trust und Reputation aktualisieren
        self._update_trust_and_reputation(other_agents)
        
        # Expertise-Erkennung für das aktuelle Szenario
        domain_experts = self._identify_domain_experts(other_agents, scenario_context)
        
        for other_agent in other_agents:
            if other_agent.agent_id == self.agent_id:
                continue
                
            # Trust-Level bestimmen
            trust_level = self.social_learning["trust_network"].get(other_agent.agent_id, 0.5)
            reputation = self.social_learning["reputation_system"].get(other_agent.agent_id, 0.5)
            
            # Expertise-Bonus
            expertise_bonus = 1.0
            if other_agent.agent_id in domain_experts:
                expertise_bonus = 1.5 + 0.5 * domain_experts[other_agent.agent_id]
            
            # Social Proof: Wenn viele ähnliche Agenten ähnlich denken
            social_proof_factor = self._calculate_social_proof(other_agent, other_agents)
            
            # Homophilie: Ähnlichkeit verstärkt Einfluss
            similarity = self._calculate_agent_similarity(other_agent)
            homophily_factor = 1.0 + 0.3 * similarity
            
            # Gesamteinfluss berechnen
            base_influence = trust_level * reputation * expertise_bonus
            social_influence = base_influence * social_proof_factor * homophily_factor
            
            # Persönlichkeits-Moderation
            conformity_tendency = self.personality_traits.get("agreeableness", 0.5)
            independence = 1.0 - conformity_tendency
            
            # Bei hoher Unabhängigkeit weniger sozialer Einfluss
            final_influence = social_influence * (0.3 + 0.7 * conformity_tendency)
            
            # Gruppenkonformität
            for group_name, group_strength in self.group_identities.items():
                if group_name in other_agent.group_identities:
                    other_group_strength = other_agent.group_identities[group_name]
                    if group_strength > 0.6 and other_group_strength > 0.6:
                        # Starke gemeinsame Gruppenidentität
                        conformity_pressure = self.social_learning["conformity_pressure"].get(group_name, 0.3)
                        final_influence *= (1.0 + conformity_pressure)
            
            # Tatsächliches Lernen von Überzeugungen
            learning_changes = self._learn_from_agent(other_agent, final_influence)
            
            # Änderungen aggregieren
            for belief_name, change in learning_changes.items():
                if belief_name not in belief_changes:
                    belief_changes[belief_name] = 0.0
                belief_changes[belief_name] += change
                
            social_influences[other_agent.agent_id] = {
                "trust": trust_level,
                "reputation": reputation,
                "expertise_bonus": expertise_bonus,
                "social_proof": social_proof_factor,
                "similarity": similarity,
                "final_influence": final_influence,
                "belief_changes": learning_changes
            }
        
        # Opinion Leadership: Eigener Einfluss auf andere schätzen
        leadership_score = self.social_learning["opinion_leadership"]
        if leadership_score > 0.7:
            # Starke Meinungsführer sind weniger beeinflussbar
            for belief_name in belief_changes:
                belief_changes[belief_name] *= (1.0 - 0.3 * leadership_score)
        
        return belief_changes
    
    def _update_trust_and_reputation(self, other_agents: List['NeuralEthicalAgent']):
        """Aktualisiert Trust und Reputation basierend auf vergangenen Interaktionen."""
        for other_agent in other_agents:
            agent_id = other_agent.agent_id
            
            # Trust basierend auf Ähnlichkeit der Entscheidungen in der Vergangenheit
            decision_similarity = self._calculate_decision_similarity(other_agent)
            current_trust = self.social_learning["trust_network"].get(agent_id, 0.5)
            
            # Trust entwickelt sich langsam
            trust_learning_rate = 0.05
            new_trust = current_trust + trust_learning_rate * (decision_similarity - current_trust)
            self.social_learning["trust_network"][agent_id] = np.clip(new_trust, 0.0, 1.0)
            
            # Reputation basierend auf wahrgenommener Kompetenz
            competence = self._estimate_agent_competence(other_agent)
            current_reputation = self.social_learning["reputation_system"].get(agent_id, 0.5)
            
            reputation_learning_rate = 0.03
            new_reputation = current_reputation + reputation_learning_rate * (competence - current_reputation)
            self.social_learning["reputation_system"][agent_id] = np.clip(new_reputation, 0.0, 1.0)
    
    def _identify_domain_experts(self, other_agents: List['NeuralEthicalAgent'], 
                               scenario_context: Dict[str, float] = None) -> Dict[str, float]:
        """Identifiziert Expertise anderer Agenten für das aktuelle Szenario."""
        experts = {}
        
        if not scenario_context:
            return experts
            
        # Bestimme die relevante Domäne basierend auf dem Szenario
        domain = scenario_context.get("domain", "general")
        
        for other_agent in other_agents:
            expertise_score = 0.0
            
            # Expertise basierend auf früheren erfolgreichen Entscheidungen in ähnlichen Szenarien
            relevant_decisions = [
                decision for decision in other_agent.decision_history
                if domain in decision.get("scenario_id", "")
            ]
            
            if relevant_decisions:
                # Durchschnittliche Konfidenz und erfolgreiche Entscheidungen
                avg_confidence = np.mean([d.get("confidence", 0.5) for d in relevant_decisions])
                low_dissonance = np.mean([1.0 - d.get("cognitive_dissonance", 0.5) for d in relevant_decisions])
                
                expertise_score = (avg_confidence + low_dissonance) / 2.0
                
                # Bonus für umfangreiche Erfahrung
                experience_bonus = min(0.3, len(relevant_decisions) / 20.0)
                expertise_score += experience_bonus
            
            # Expertise basierend auf kognitiven Eigenschaften
            if other_agent.cognitive_architecture.primary_processing == NeuralProcessingType.SYSTEMATIC:
                expertise_score += 0.1  # Systematische Denker als kompetenter wahrgenommen
            
            if other_agent.personality_traits.get("conscientiousness", 0.5) > 0.8:
                expertise_score += 0.1  # Gewissenhafte als kompetenter wahrgenommen
                
            if expertise_score > 0.6:
                experts[other_agent.agent_id] = expertise_score
        
        return experts
    
    def _calculate_social_proof(self, target_agent: 'NeuralEthicalAgent', 
                              all_agents: List['NeuralEthicalAgent']) -> float:
        """Berechnet Social Proof: Wie viele ähnliche Agenten ähnlich denken."""
        if len(all_agents) < 3:
            return 1.0
            
        similar_agents = []
        for other_agent in all_agents:
            if other_agent.agent_id in [self.agent_id, target_agent.agent_id]:
                continue
                
            similarity = self._calculate_agent_similarity(other_agent)
            if similarity > 0.6:  # Schwellenwert für "ähnlich"
                similar_agents.append(other_agent)
        
        if not similar_agents:
            return 1.0
            
        # Wie viele der ähnlichen Agenten haben ähnliche Überzeugungen wie der target_agent?
        agreement_count = 0
        total_comparisons = 0
        
        for similar_agent in similar_agents:
            for belief_name in target_agent.beliefs:
                if belief_name in similar_agent.beliefs:
                    target_strength = target_agent.beliefs[belief_name].strength
                    similar_strength = similar_agent.beliefs[belief_name].strength
                    
                    # Agreement wenn Differenz < 0.3
                    if abs(target_strength - similar_strength) < 0.3:
                        agreement_count += 1
                    total_comparisons += 1
        
        if total_comparisons == 0:
            return 1.0
            
        agreement_ratio = agreement_count / total_comparisons
        
        # Social Proof Verstärkung basierend auf eigener Sensitivität
        social_proof_sensitivity = self.social_learning["social_proof_sensitivity"]
        
        return 1.0 + social_proof_sensitivity * agreement_ratio
    
    def _calculate_agent_similarity(self, other_agent: 'NeuralEthicalAgent') -> float:
        """Berechnet die Ähnlichkeit zu einem anderen Agenten."""
        # Persönlichkeitsähnlichkeit
        personality_similarity = 0.0
        for trait in self.personality_traits:
            if trait in other_agent.personality_traits:
                diff = abs(self.personality_traits[trait] - other_agent.personality_traits[trait])
                personality_similarity += 1.0 - diff
        personality_similarity /= len(self.personality_traits)
        
        # Kognitive Architektur-Ähnlichkeit
        cognitive_similarity = 0.0
        if self.cognitive_architecture.primary_processing == other_agent.cognitive_architecture.primary_processing:
            cognitive_similarity += 0.7
        if self.cognitive_architecture.secondary_processing == other_agent.cognitive_architecture.secondary_processing:
            cognitive_similarity += 0.3
            
        # Moral Foundations Ähnlichkeit
        moral_similarity = 0.0
        for foundation in self.moral_foundations:
            if foundation in other_agent.moral_foundations:
                diff = abs(self.moral_foundations[foundation] - other_agent.moral_foundations[foundation])
                moral_similarity += 1.0 - diff
        moral_similarity /= len(self.moral_foundations)
        
        # Gewichtete Gesamtähnlichkeit
        total_similarity = (0.4 * personality_similarity + 
                          0.3 * cognitive_similarity + 
                          0.3 * moral_similarity)
        
        return total_similarity
    
    def _calculate_decision_similarity(self, other_agent: 'NeuralEthicalAgent') -> float:
        """Berechnet Ähnlichkeit basierend auf vergangenen Entscheidungen."""
        if not self.decision_history or not other_agent.decision_history:
            return 0.5  # Neutrale Annahme
            
        # Finde gemeinsame Szenarien
        my_scenarios = {d["scenario_id"]: d["chosen_option"] for d in self.decision_history}
        other_scenarios = {d["scenario_id"]: d["chosen_option"] for d in other_agent.decision_history}
        
        common_scenarios = set(my_scenarios.keys()).intersection(set(other_scenarios.keys()))
        
        if not common_scenarios:
            return 0.5
            
        # Berechne Übereinstimmung
        agreements = sum(1 for scenario in common_scenarios 
                        if my_scenarios[scenario] == other_scenarios[scenario])
        
        return agreements / len(common_scenarios)
    
    def _estimate_agent_competence(self, other_agent: 'NeuralEthicalAgent') -> float:
        """Schätzt die Kompetenz eines anderen Agenten."""
        if not other_agent.decision_history:
            return 0.5
            
        # Durchschnittliche Konfidenz als Proxy für Kompetenz
        avg_confidence = np.mean([abs(d.get("confidence", 0.5)) for d in other_agent.decision_history])
        
        # Niedrige kognitive Dissonanz als Zeichen für Kompetenz
        avg_dissonance = np.mean([d.get("cognitive_dissonance", 0.5) for d in other_agent.decision_history])
        low_dissonance_score = 1.0 - avg_dissonance
        
        # Konsistenz der Entscheidungen
        consistency_score = self._calculate_decision_consistency(other_agent)
        
        # Gewichtete Kompetenzschätzung
        competence = (0.4 * avg_confidence + 
                     0.3 * low_dissonance_score + 
                     0.3 * consistency_score)
        
        return np.clip(competence, 0.0, 1.0)
    
    def _calculate_decision_consistency(self, other_agent: 'NeuralEthicalAgent') -> float:
        """Berechnet die Konsistenz der Entscheidungen eines Agenten."""
        if len(other_agent.decision_history) < 2:
            return 0.5
            
        # Gruppiere Entscheidungen nach Szenario-Typ
        scenario_decisions = {}
        for decision in other_agent.decision_history:
            scenario_type = decision["scenario_id"].split('_')[0] if '_' in decision["scenario_id"] else decision["scenario_id"]
            
            if scenario_type not in scenario_decisions:
                scenario_decisions[scenario_type] = []
            scenario_decisions[scenario_type].append(decision["chosen_option"])
        
        # Berechne Konsistenz für jeden Szenario-Typ
        consistency_scores = []
        for scenario_type, decisions in scenario_decisions.items():
            if len(decisions) > 1:
                # Häufigste Entscheidung
                from collections import Counter
                most_common = Counter(decisions).most_common(1)[0]
                consistency = most_common[1] / len(decisions)
                consistency_scores.append(consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 0.5
    
    def _learn_from_agent(self, other_agent: 'NeuralEthicalAgent', 
                         influence_strength: float) -> Dict[str, float]:
        """Lernt spezifische Überzeugungen von einem anderen Agenten."""
        belief_changes = {}
        
        # Gemeinsame Überzeugungen finden
        common_beliefs = set(self.beliefs.keys()).intersection(set(other_agent.beliefs.keys()))
        
        for belief_name in common_beliefs:
            my_belief = self.beliefs[belief_name]
            other_belief = other_agent.beliefs[belief_name]
            
            # Stärke-Differenz
            strength_diff = other_belief.strength - my_belief.strength
            
            # Gewissheits-gewichtete Anpassung
            other_certainty = other_belief.certainty
            my_certainty = my_belief.certainty
            
            # Wenn der andere gewisser ist, mehr Einfluss
            certainty_factor = other_certainty / (my_certainty + other_certainty + 0.1)
            
            # Finale Änderung
            base_change = strength_diff * influence_strength * 0.1  # Basis-Lernrate
            certainty_adjusted_change = base_change * certainty_factor
            
            # Widerstand gegen große Änderungen (konservatieve Anpassung)
            if abs(certainty_adjusted_change) > 0.2:
                certainty_adjusted_change *= 0.5
                
            belief_changes[belief_name] = certainty_adjusted_change
        
        return belief_changes
    
    def temporal_belief_dynamics(self, time_step: float = 1.0) -> Dict[str, float]:
        """
        Modelliert zeitliche Dynamik von Überzeugungen (Momentum, Decay, Habituation).
        
        Args:
            time_step: Zeitschritt für die Simulation
            
        Returns:
            Dictionary mit zeitlichen Anpassungen der Überzeugungen
        """
        belief_changes = {}
        
        # Zeit voranschreiten
        self.current_time += time_step
        
        for belief_name, belief in self.beliefs.items():
            old_strength = belief.strength
            temporal_change = 0.0
            
            # Momentum-Effekt: Überzeugungen in Bewegung bleiben in Bewegung
            if belief_name in self.temporal_dynamics["belief_momentum"]:
                momentum = self.temporal_dynamics["belief_momentum"][belief_name]
                momentum_effect = momentum * 0.1 * time_step
                temporal_change += momentum_effect
                
                # Momentum verlangsamt sich über Zeit (Reibung)
                self.temporal_dynamics["belief_momentum"][belief_name] *= 0.95
            
            # Memory Decay: Überzeugungen kehren langsam zum neutralen Wert zurück
            decay_rate = self.temporal_dynamics["memory_decay"]
            neutral_point = 0.5
            decay_effect = (neutral_point - old_strength) * decay_rate * time_step
            temporal_change += decay_effect
            
            # Habituation: Reduzierte Reaktion auf wiederholte Stimuli
            if belief_name in self.temporal_dynamics["habituation_effects"]:
                habituation_level = self.temporal_dynamics["habituation_effects"][belief_name]
                if habituation_level > 0.7:  # Hohe Habituation
                    # Reduktion der Reaktivität
                    temporal_change *= (1.0 - habituation_level * 0.3)
                    
                # Habituation klingt langsam ab
                self.temporal_dynamics["habituation_effects"][belief_name] *= 0.98
            
            # Adaptationsrate anwenden
            adaptation_rate = self.temporal_dynamics["adaptation_rate"]
            final_change = temporal_change * adaptation_rate
            
            # Überzeugung aktualisieren
            new_strength = np.clip(old_strength + final_change, 0.0, 1.0)
            if abs(final_change) > 0.001:  # Nur signifikante Änderungen
                self.update_belief(belief_name, new_strength)
                belief_changes[belief_name] = final_change
                
        # Recency Bias: Neuere Erinnerungen verstärken
        recency_bias = self.temporal_dynamics["recency_bias"]
        recent_memories = [m for m in self.episodic_memory if self.current_time - m["time"] < 5.0]
        
        for memory in recent_memories:
            if memory["type"] == "belief_change":
                belief_name = memory["belief"]
                if belief_name in self.beliefs:
                    # Verstärkung durch Recency
                    recency_factor = 1.0 - (self.current_time - memory["time"]) / 5.0
                    recency_boost = memory["change"] * recency_bias * recency_factor * 0.1
                    
                    current_strength = self.beliefs[belief_name].strength
                    new_strength = np.clip(current_strength + recency_boost, 0.0, 1.0)
                    self.update_belief(belief_name, new_strength)
                    
                    if belief_name in belief_changes:
                        belief_changes[belief_name] += recency_boost
                    else:
                        belief_changes[belief_name] = recency_boost
        
        return belief_changes
    
    def metacognitive_monitoring(self, decision_result: Dict[str, Union[str, float]]) -> Dict[str, float]:
        """
        Meta-kognitive Überwachung und Strategieanpassung.
        
        Args:
            decision_result: Ergebnis einer Entscheidung mit Konfidenz, Dissonanz, etc.
            
        Returns:
            Anpassungen der meta-kognitiven Parameter
        """
        if not self.metacognition["thinking_about_thinking"]:
            return {}
            
        adjustments = {}
        
        # Strategie-Monitoring: Wie gut funktioniert die aktuelle Denkstrategie?
        monitoring_level = self.metacognition["strategy_monitoring"]
        
        # Fehler-Erkennung basierend auf hoher Dissonanz oder niedriger Konfidenz
        cognitive_dissonance = decision_result.get("cognitive_dissonance", 0.0)
        confidence = abs(decision_result.get("confidence", 0.5))
        
        error_indicator = cognitive_dissonance + (1.0 - confidence)
        error_detected = error_indicator > self.metacognition["error_detection"]
        
        if error_detected and monitoring_level > 0.6:
            # Strategiewechsel erwägen
            switch_threshold = self.metacognition["strategy_switching_threshold"]
            
            if error_indicator > switch_threshold:
                # Wechsel zwischen primärer und sekundärer Verarbeitung
                old_balance = self.cognitive_architecture.processing_balance
                
                if old_balance > 0.6:  # Primär dominiert
                    new_balance = 0.4  # Mehr sekundäre Verarbeitung
                elif old_balance < 0.4:  # Sekundär dominiert
                    new_balance = 0.6  # Mehr primäre Verarbeitung
                else:
                    # Bei ausgewogener Balance, verstärke den weniger genutzten Typ
                    new_balance = 0.7 if random.random() > 0.5 else 0.3
                
                self.cognitive_architecture.processing_balance = new_balance
                adjustments["processing_balance"] = new_balance - old_balance
                
                # Aktualisiere Lernparameter
                if error_indicator > 0.8:  # Schwerwiegender Fehler
                    # Erhöhe Exploration
                    old_exploration = self.rl_system["exploration_rate"]
                    self.rl_system["exploration_rate"] = min(0.8, old_exploration * 1.2)
                    adjustments["exploration_increase"] = self.rl_system["exploration_rate"] - old_exploration
                    
                    # Erhöhe Lernrate vorübergehend
                    old_learning_rate = self.rl_system["learning_rate"]
                    self.rl_system["learning_rate"] = min(0.5, old_learning_rate * 1.1)
                    adjustments["learning_rate_increase"] = self.rl_system["learning_rate"] - old_learning_rate
        
        # Kognitive Belastung überwachen
        load_awareness = self.metacognition["cognitive_load_awareness"]
        
        # Schätze kognitive Belastung
        num_active_beliefs = sum(1 for belief in self.beliefs.values() if belief.activation > 0.1)
        working_memory_load = len(self.working_memory["contents"]) / self.working_memory["capacity"]
        recent_decisions = len([d for d in self.decision_history if self.current_time - d["timestamp"] < 3.0])
        
        cognitive_load = (num_active_beliefs / 20.0 + working_memory_load + recent_decisions / 10.0) / 3.0
        
        if cognitive_load > 0.8 and load_awareness > 0.6:
            # Belastung reduzieren
            # Vereinfache Entscheidungsstrategien
            if self.cognitive_architecture.primary_processing == NeuralProcessingType.SYSTEMATIC:
                # Systematisches Denken ist belastend, zu intuitivem wechseln
                old_balance = self.cognitive_architecture.processing_balance
                self.cognitive_architecture.processing_balance = max(0.3, old_balance - 0.2)
                adjustments["load_reduction"] = old_balance - self.cognitive_architecture.processing_balance
            
            # Arbeitsgedächtnis leeren
            overflow = max(0, len(self.working_memory["contents"]) - self.working_memory["capacity"])
            if overflow > 0:
                self.working_memory["contents"] = self.working_memory["contents"][overflow:]
                adjustments["memory_cleared"] = overflow
                
        # Selbst-Reflektion: Analyse der eigenen Denkprozesse
        if len(self.decision_history) >= 5:
            # Konsistenz der eigenen Entscheidungen analysieren
            recent_decisions = self.decision_history[-5:]
            
            # Varianz in Konfidenzlevels
            confidences = [abs(d.get("confidence", 0.5)) for d in recent_decisions]
            confidence_variance = np.var(confidences)
            
            # Hohe Varianz deutet auf inkonsistente Strategien hin
            if confidence_variance > 0.3:
                # Erhöhe Strategieüberwachung
                old_monitoring = self.metacognition["strategy_monitoring"]
                self.metacognition["strategy_monitoring"] = min(1.0, old_monitoring + 0.1)
                adjustments["increased_monitoring"] = 0.1
                
            # Dissonanz-Trend analysieren
            dissonances = [d.get("cognitive_dissonance", 0.0) for d in recent_decisions]
            avg_dissonance = np.mean(dissonances)
            
            if avg_dissonance > 0.6:  # Anhaltend hohe Dissonanz
                # Erhöhe Fehlererkennungssensitivität
                old_detection = self.metacognition["error_detection"]
                self.metacognition["error_detection"] = max(0.3, old_detection - 0.1)
                adjustments["improved_error_detection"] = old_detection - self.metacognition["error_detection"]
        
        # Episodisches Gedächtnis für Meta-kognition aktualisieren
        if adjustments:
            self.episodic_memory.append({
                "time": self.current_time,
                "type": "metacognitive_adjustment",
                "adjustments": adjustments,
                "trigger": {
                    "cognitive_dissonance": cognitive_dissonance,
                    "confidence": confidence,
                    "cognitive_load": cognitive_load
                }
            })
        
        return adjustments
    
    def enhanced_make_decision(self, scenario: 'EthicalScenario') -> Dict[str, Union[str, float, Dict]]:
        """
        Erweiterte Entscheidungsfindung mit allen neuen Lernmechanismen.
        
        Returns:
            Umfangreiches Entscheidungsergebnis mit Lernstatistiken
        """
        # === PHASE 1: VORBEREITUNG UND AKTIVIERUNG ===
        
        # Relevante Überzeugungen aktivieren
        seed_beliefs = list(scenario.relevant_beliefs.keys())
        activation_levels = [scenario.relevant_beliefs[b] for b in seed_beliefs 
                           if b in self.beliefs]
        seed_beliefs = [b for b in seed_beliefs if b in self.beliefs]
        
        # Spreading Activation durchführen
        self.spreading_activation(seed_beliefs, activation_levels)
        
        # Unsicherheitsbehandlung anwenden
        belief_inputs = {name: belief.strength for name, belief in self.beliefs.items()}
        adjusted_beliefs = self.handle_uncertainty_and_ambiguity(scenario, belief_inputs)
        
        # === PHASE 2: KOGNITIVE VERARBEITUNG ===
        
        # Kontext für die Verarbeitung bereitstellen
        context = {
            "scenario": scenario.scenario_id,
            "emotional_valence": np.mean([self.beliefs[b].emotional_valence for b in seed_beliefs 
                                       if b in self.beliefs]) if seed_beliefs else 0,
            "narrative_coherence": 0.7,
            "domain": scenario.scenario_id.split('_')[0] if '_' in scenario.scenario_id else "general"
        }
        
        # Verarbeitung durch kognitive Architektur
        processed_beliefs = self.cognitive_architecture.process_information(adjusted_beliefs, context)
        
        # === PHASE 3: REINFORCEMENT LEARNING INTEGRATION ===
        
        # RL-Werte in Entscheidung einbeziehen
        scenario_type = context["domain"]
        rl_adjustments = {}
        
        if scenario_type in self.rl_system["action_values"]:
            for option_name in scenario.options.keys():
                if option_name in self.rl_system["action_values"][scenario_type]:
                    rl_value = self.rl_system["action_values"][scenario_type][option_name]
                    rl_adjustments[option_name] = rl_value
        
        # === PHASE 4: MULTI-CRITERIA BEWERTUNG ===
        
        # Optionen bewerten
        option_evaluations = {}
        for option_name, option_impacts in scenario.options.items():
            criteria_scores = {
                "personal_benefit": 0.0,
                "social_good": 0.0,
                "moral_alignment": 0.0,
                "risk_minimization": 0.0
            }
            
            # Überzeugungsbeitrag
            for belief_name, impact in option_impacts.items():
                if belief_name in processed_beliefs:
                    belief_score = processed_beliefs[belief_name] * impact
                    criteria_scores["moral_alignment"] += belief_score * 0.5
                    criteria_scores["personal_benefit"] += belief_score * 0.3
                    criteria_scores["social_good"] += belief_score * 0.2
            
            # Moralische Grundlagen einbeziehen
            if option_name in scenario.moral_implications:
                for foundation, impact in scenario.moral_implications[option_name].items():
                    if foundation in self.moral_foundations:
                        moral_score = self.moral_foundations[foundation] * impact
                        criteria_scores["moral_alignment"] += moral_score * 0.5
            
            # Risikobewertung
            if "risks" in scenario.option_attributes.get(option_name, {}):
                risk_level = scenario.option_attributes[option_name]["risks"]
                criteria_scores["risk_minimization"] = 1.0 - risk_level
            else:
                criteria_scores["risk_minimization"] = 0.7  # Neutrale Annahme
            
            # RL-Werte einbeziehen
            if option_name in rl_adjustments:
                rl_bonus = rl_adjustments[option_name] * 0.3
                criteria_scores["personal_benefit"] += rl_bonus
            
            option_evaluations[option_name] = criteria_scores
        
        # Multi-Criteria Decision Making anwenden
        final_scores = self.multi_criteria_decision_making(scenario, option_evaluations)
        
        # === PHASE 5: FINALE ENTSCHEIDUNG ===
        
        # Option mit höchstem Score wählen (mit Exploration)
        options = list(final_scores.keys())
        scores = list(final_scores.values())
        
        # Exploration vs. Exploitation
        exploration_rate = self.rl_system["exploration_rate"]
        if random.random() < exploration_rate:
            # Exploration: Zufällige Wahl
            chosen_option = random.choice(options)
        else:
            # Exploitation: Beste Option
            best_index = np.argmax(scores)
            chosen_option = options[best_index]
        
        # Konfidenz berechnen
        if len(scores) > 1:
            sorted_scores = sorted(scores, reverse=True)
            score_diff = sorted_scores[0] - sorted_scores[1]
            confidence = np.tanh(score_diff * 2)
        else:
            confidence = 0.5
        
        # === PHASE 6: ERGEBNIS UND LERNEN ===
        
        # Entscheidung zusammenstellen
        decision_result = {
            "scenario_id": scenario.scenario_id,
            "chosen_option": chosen_option,
            "option_scores": final_scores,
            "criteria_evaluations": option_evaluations,
            "cognitive_dissonance": self.calculate_cognitive_dissonance(),
            "confidence": confidence,
            "rl_values": rl_adjustments,
            "uncertainty_info": self.uncertainty_handling["uncertainty_estimates"],
            "exploration_used": random.random() < exploration_rate,
            "timestamp": self.current_time,
            "processing_details": {
                "primary_processing": self.cognitive_architecture.primary_processing,
                "processing_balance": self.cognitive_architecture.processing_balance,
                "adjusted_beliefs": len(adjusted_beliefs),
                "criteria_weights": self.decision_criteria["utility_weights"]
            }
        }
        
        # Entscheidung zur Historie hinzufügen
        self.decision_history.append(decision_result)
        
        # Meta-kognitive Überwachung
        metacognitive_adjustments = self.metacognitive_monitoring(decision_result)
        if metacognitive_adjustments:
            decision_result["metacognitive_adjustments"] = metacognitive_adjustments
        
        # Temporale Dynamik aktualisieren
        temporal_changes = self.temporal_belief_dynamics()
        if temporal_changes:
            decision_result["temporal_changes"] = temporal_changes
        
        # Episodisches Gedächtnis aktualisieren
        if decision_result["cognitive_dissonance"] > 0.3 or abs(confidence) > 0.7:
            self.episodic_memory.append({
                "time": self.current_time,
                "type": "significant_decision",
                "scenario": scenario.scenario_id,
                "decision": chosen_option,
                "dissonance": decision_result["cognitive_dissonance"],
                "confidence": confidence,
                "enhanced_features": True
            })
        
        return decision_result
    
    def learn_from_decision_outcome(self, scenario: 'EthicalScenario', 
                                  chosen_option: str,
                                  outcome_feedback: Dict[str, float],
                                  other_agents: List['NeuralEthicalAgent'] = None) -> Dict[str, float]:
        """
        Umfassendes Lernen aus Entscheidungsresultaten mit allen erweiterten Mechanismen.
        
        Args:
            scenario: Das ethische Szenario
            chosen_option: Die gewählte Option
            outcome_feedback: Feedback über verschiedene Dimensionen des Ergebnisses
            other_agents: Andere Agenten für soziales Lernen
            
        Returns:
            Zusammenfassung aller Lernänderungen
        """
        learning_summary = {}
        
        # === REINFORCEMENT LEARNING ===
        rl_learning = self.reinforcement_learn_from_outcome(scenario, chosen_option, outcome_feedback)
        learning_summary["reinforcement_learning"] = rl_learning
        
        # === BELIEF UPDATES ===
        belief_updates = self.update_beliefs_from_experience(scenario, chosen_option)
        learning_summary["belief_updates"] = belief_updates
        
        # === CRITERIA WEIGHT LEARNING ===
        # Lerne über die Effektivität verschiedener Entscheidungskriterien
        total_outcome = sum(outcome_feedback.values()) / len(outcome_feedback)
        
        for criterion in self.decision_criteria["utility_weights"]:
            if criterion.replace("_", "").replace("minimization", "outcome") in outcome_feedback:
                criterion_outcome = outcome_feedback.get(criterion.replace("_", "").replace("minimization", "outcome"), 0)
                current_weight = self.decision_criteria["utility_weights"][criterion]
                
                # Positive Ergebnisse erhöhen das Gewicht, negative verringern es
                weight_adjustment = 0.02 * np.sign(criterion_outcome) * abs(criterion_outcome)
                new_weight = np.clip(current_weight + weight_adjustment, 0.05, 0.6)
                
                self.decision_criteria["utility_weights"][criterion] = new_weight
                learning_summary.setdefault("criteria_learning", {})[criterion] = weight_adjustment
        
        # Gewichte re-normalisieren
        total_weight = sum(self.decision_criteria["utility_weights"].values())
        for criterion in self.decision_criteria["utility_weights"]:
            self.decision_criteria["utility_weights"][criterion] /= total_weight
        
        # === UNCERTAINTY CALIBRATION ===
        # Verbessere Unsicherheitsschätzungen basierend auf tatsächlichen Ergebnissen
        predicted_uncertainty = self.uncertainty_handling["uncertainty_estimates"]
        actual_surprise = abs(total_outcome) if abs(total_outcome) > 0.5 else 1.0 - abs(total_outcome)
        
        # Kalibrierung anpassen
        old_calibration = self.uncertainty_handling["confidence_calibration"]
        if actual_surprise > 0.7:  # Überraschende Ergebnisse
            # Waren wir zu sicher? Kalibrierung verschlechtern
            self.uncertainty_handling["confidence_calibration"] = max(0.3, old_calibration - 0.05)
        elif actual_surprise < 0.3:  # Erwartete Ergebnisse
            # Gute Vorhersage, Kalibrierung verbessern
            self.uncertainty_handling["confidence_calibration"] = min(1.0, old_calibration + 0.02)
        
        learning_summary["uncertainty_calibration"] = {
            "old_calibration": old_calibration,
            "new_calibration": self.uncertainty_handling["confidence_calibration"],
            "actual_surprise": actual_surprise
        }
        
        # === SOCIAL LEARNING ===
        if other_agents:
            social_learning_changes = self.advanced_social_learning(other_agents, {"domain": scenario.scenario_id.split('_')[0]})
            learning_summary["social_learning"] = social_learning_changes
        
        # === TEMPORAL DYNAMICS ===
        # Momentum für veränderte Überzeugungen hinzufügen
        for belief_name, change in belief_updates.items():
            if abs(change) > 0.05:  # Signifikante Änderung
                current_momentum = self.temporal_dynamics["belief_momentum"].get(belief_name, 0.0)
                # Momentum in Richtung der Änderung verstärken
                new_momentum = current_momentum + np.sign(change) * min(0.3, abs(change))
                self.temporal_dynamics["belief_momentum"][belief_name] = np.clip(new_momentum, -0.5, 0.5)
        
        # === HABITUATION EFFECTS ===
        # Wiederholte ähnliche Szenarien führen zu Habituation
        scenario_type = scenario.scenario_id.split('_')[0] if '_' in scenario.scenario_id else scenario.scenario_id
        recent_similar = [d for d in self.decision_history[-10:] 
                         if scenario_type in d.get("scenario_id", "")]
        
        if len(recent_similar) > 3:  # Häufiges ähnliches Szenario
            for belief_name in scenario.relevant_beliefs:
                if belief_name in self.beliefs:
                    current_habituation = self.temporal_dynamics["habituation_effects"].get(belief_name, 0.0)
                    self.temporal_dynamics["habituation_effects"][belief_name] = min(0.9, current_habituation + 0.1)
        
        # === META-LEARNING ===
        # Lerne über die Effektivität verschiedener Lernstrategien
        learning_effectiveness = abs(total_outcome)
        
        if learning_effectiveness > 0.6:  # Gutes Ergebnis
            # Verstärke aktuelle Lernparameter
            self.rl_system["learning_rate"] = min(0.3, self.rl_system["learning_rate"] * 1.05)
            self.temporal_dynamics["adaptation_rate"] = min(0.3, self.temporal_dynamics["adaptation_rate"] * 1.05)
        elif learning_effectiveness < 0.3:  # Schlechtes Ergebnis
            # Reduziere Lernparameter (konservativer werden)
            self.rl_system["learning_rate"] = max(0.01, self.rl_system["learning_rate"] * 0.95)
            self.temporal_dynamics["adaptation_rate"] = max(0.05, self.temporal_dynamics["adaptation_rate"] * 0.95)
        
        learning_summary["meta_learning"] = {
            "learning_rate_adjustment": self.rl_system["learning_rate"],
            "adaptation_rate_adjustment": self.temporal_dynamics["adaptation_rate"],
            "effectiveness": learning_effectiveness
        }
        
        # === EPISODIC MEMORY UPDATE ===
        self.episodic_memory.append({
            "time": self.current_time,
            "type": "comprehensive_learning",
            "scenario": scenario.scenario_id,
            "chosen_option": chosen_option,
            "outcome_feedback": outcome_feedback,
            "learning_changes": learning_summary,
            "total_outcome": total_outcome
        })
        
        # Memory-Management: Alte Erinnerungen löschen
        if len(self.episodic_memory) > 200:
            self.episodic_memory = self.episodic_memory[-200:]
        
        return learning_summary

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