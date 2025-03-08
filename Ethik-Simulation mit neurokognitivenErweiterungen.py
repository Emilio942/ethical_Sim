# Fortsetzung der NeuralEthicalSociety-Klasse

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
    
    def visualize_neural_processing(self, agent_id: str):
        """Visualisiert die neuronale Verarbeitung eines Agenten."""
        if agent_id not in self.agents:
            print(f"Agent {agent_id} nicht gefunden.")
            return
            
        agent = self.agents[agent_id]
        
        # 1. Kognitive Architektur visualisieren
        plt.figure(figsize=(15, 10))
        
        # 1.1 Aktuelle Überzeugungsaktivierung visualisieren
        plt.subplot(2, 2, 1)
        beliefs = list(agent.beliefs.keys())
        activations = [agent.beliefs[b].activation for b in beliefs]
        
        # Sortieren nach Aktivierungsniveau
        sorted_indices = np.argsort(activations)[::-1]
        sorted_beliefs = [beliefs[i] for i in sorted_indices]
        sorted_activations = [activations[i] for i in sorted_indices]
        
        # Nur die Top 10 anzeigen, falls mehr vorhanden
        if len(sorted_beliefs) > 10:
            sorted_beliefs = sorted_beliefs[:10]
            sorted_activations = sorted_activations[:10]
        
        plt.barh(sorted_beliefs, sorted_activations, color='skyblue')
        plt.xlabel('Aktivierungsniveau')
        plt.title('Aktuelle Überzeugungsaktivierung')
        plt.grid(axis='x', alpha=0.3)
        
        # 1.2 Kognitive Biases visualisieren
        plt.subplot(2, 2, 2)
        biases = list(agent.cognitive_architecture.cognitive_biases.keys())
        bias_values = [agent.cognitive_architecture.cognitive_biases[b] for b in biases]
        
        plt.barh(biases, bias_values, color='salmon')
        plt.xlabel('Ausprägung')
        plt.title('Kognitive Biases')
        plt.grid(axis='x', alpha=0.3)
        
        # 1.3 Verarbeitungsstile visualisieren
        plt.subplot(2, 2, 3)
        styles = [
            f"Primär: {agent.cognitive_architecture.primary_processing}",
            f"Sekundär: {agent.cognitive_architecture.secondary_processing}"
        ]
        style_weights = [
            agent.cognitive_architecture.processing_balance,
            1.0 - agent.cognitive_architecture.processing_balance
        ]
        
        plt.barh(styles, style_weights, color='lightgreen')
        plt.xlabel('Gewichtung')
        plt.title('Kognitive Verarbeitungsstile')
        plt.grid(axis='x', alpha=0.3)
        
        # 1.4 Emotionale Parameter visualisieren
        plt.subplot(2, 2, 4)
        emotions = list(agent.cognitive_architecture.emotional_parameters.keys())
        emotion_values = [agent.cognitive_architecture.emotional_parameters[e] for e in emotions]
        
        plt.barh(emotions, emotion_values, color='gold')
        plt.xlabel('Ausprägung')
        plt.title('Emotionale Parameter')
        plt.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle(f"Neuronale Verarbeitung für Agent {agent_id}", fontsize=16)
        plt.subplots_adjust(top=0.9)
        plt.show()
        
        # 2. Überzeugungsnetzwerk mit Aktivierungsniveaus visualisieren
        self.visualize_belief_network(agent_id, show_activation=True)
    
    def visualize_belief_network(self, agent_id: str, min_connection_strength: float = 0.2, 
                               show_activation: bool = False):
        """
        Visualisiert das Netzwerk von Überzeugungen eines Agenten.
        
        Args:
            agent_id: ID des zu visualisierenden Agenten
            min_connection_strength: Minimale Verbindungsstärke für die Anzeige
            show_activation: Ob Aktivierungsniveaus angezeigt werden sollen
        """
        if agent_id not in self.agents:
            print(f"Agent {agent_id} nicht gefunden.")
            return
            
        agent = self.agents[agent_id]
        G = nx.DiGraph()
        
        # Knoten für jede Überzeugung hinzufügen
        for belief_name, belief in agent.beliefs.items():
            node_attrs = {
                'strength': belief.strength, 
                'category': belief.category,
                'certainty': belief.certainty
            }
            
            if show_activation:
                node_attrs['activation'] = belief.activation
                
            G.add_node(belief_name, **node_attrs)
            
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
            
            if show_activation:
                # Größe basierend auf Aktivierung
                node_sizes = [300 + 700 * G.nodes[node]['activation'] for node in node_list]
                
                # Farbtransparenz basierend auf Gewissheit
                alphas = [0.3 + 0.7 * G.nodes[node]['certainty'] for node in node_list]
                node_colors = [(*color[:3], alpha) for alpha in alphas]
            else:
                # Größe basierend auf Stärke
                node_sizes = [300 + 700 * G.nodes[node]['strength'] for node in node_list]
                node_colors = [color] * len(node_list)
            
            nx.draw_networkx_nodes(
                G, pos, 
                nodelist=node_list, 
                node_color=node_colors,
                node_size=node_sizes,
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
        
        title = f"Überzeugungsnetzwerk für Agent {agent_id}"
        if show_activation:
            title += " (mit Aktivierungsniveaus)"
            
        plt.title(title)
        plt.legend()
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def visualize_cognitive_style_comparison(self):
        """Visualisiert einen Vergleich der verschiedenen kognitiven Verarbeitungsstile."""
        # Agenten nach kognitivem Stil gruppieren
        style_groups = {
            NeuralProcessingType.SYSTEMATIC: [],
            NeuralProcessingType.INTUITIVE: [],
            NeuralProcessingType.ASSOCIATIVE: [],
            NeuralProcessingType.EMOTIONAL: [],
            NeuralProcessingType.ANALOGICAL: [],
            NeuralProcessingType.NARRATIVE: []
        }
        
        # Nur Stile mit Agenten berücksichtigen
        for agent_id, agent in self.agents.items():
            style = agent.cognitive_architecture.primary_processing
            style_groups[style].append(agent_id)
            
        active_styles = {s: agents for s, agents in style_groups.items() if agents}
        
        if not active_styles:
            print("Keine Agenten mit definierten kognitiven Stilen gefunden.")
            return
            
        # Häufige Überzeugungen identifizieren (für den Vergleich)
        all_beliefs = {}
        for agent in self.agents.values():
            for belief_name in agent.beliefs:
                if belief_name not in all_beliefs:
                    all_beliefs[belief_name] = 0
                all_beliefs[belief_name] += 1
        
        # Top 5 häufigste Überzeugungen auswählen
        common_beliefs = sorted(all_beliefs.items(), key=lambda x: x[1], reverse=True)[:5]
        common_belief_names = [b[0] for b in common_beliefs]
        
        # Durchschnittliche Überzeugungsstärken pro Stil und Überzeugung
        style_belief_strengths = {}
        
        for style, agent_ids in active_styles.items():
            style_belief_strengths[style] = {}
            
            for belief_name in common_belief_names:
                strengths = []
                
                for agent_id in agent_ids:
                    agent = self.agents[agent_id]
                    if belief_name in agent.beliefs:
                        strengths.append(agent.beliefs[belief_name].strength)
                
                if strengths:
                    style_belief_strengths[style][belief_name] = np.mean(strengths)
                else:
                    style_belief_strengths[style][belief_name] = 0
        
        # Visualisierung
        plt.figure(figsize=(12, 8))
        
        # Gruppierte Balkendiagramme erstellen
        x = np.arange(len(common_belief_names))
        width = 0.15  # Breite der Balken
        
        # Für jeden Stil einen Balken pro Überzeugung
        for i, (style, beliefs) in enumerate(style_belief_strengths.items()):
            strengths = [beliefs.get(b, 0) for b in common_belief_names]
            offset = width * (i - len(style_belief_strengths) / 2 + 0.5)
            plt.bar(x + offset, strengths, width, label=str(style))
        
        plt.xlabel('Überzeugungen')
        plt.ylabel('Durchschnittliche Stärke')
        plt.title('Überzeugungsstärke nach kognitivem Verarbeitungsstil')
        plt.xticks(x, common_belief_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def visualize_social_network(self, color_by: str = 'cognitive_style'):
        """
        Visualisiert das soziale Netzwerk der Agenten.
        
        Args:
            color_by: Bestimmt, wie die Knoten gefärbt werden 
                    ('group', 'belief', oder 'cognitive_style')
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
            
            color_map = plt.cm.tab10
            node_colors = [color_map(c/10) for c in colors]
            legend_elements = []
        
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
                
                color_map = plt.cm.viridis
                node_colors = colors
                legend_elements = []
            else:
                color_map = plt.cm.viridis
                node_colors = [0.5] * len(self.social_network.nodes())
                legend_elements = []
        
        elif color_by == 'cognitive_style':
            # Färben nach kognitivem Verarbeitungsstil
            style_colors = {
                NeuralProcessingType.SYSTEMATIC: 'blue',
                NeuralProcessingType.INTUITIVE: 'red',
                NeuralProcessingType.ASSOCIATIVE: 'green',
                NeuralProcessingType.EMOTIONAL: 'purple',
                NeuralProcessingType.ANALOGICAL: 'orange',
                NeuralProcessingType.NARRATIVE: 'brown'
            }
            
            node_colors = []
            for agent_id in self.social_network.nodes():
                agent = self.agents[agent_id]
                style = agent.cognitive_architecture.primary_processing
                node_colors.append(style_colors.get(style, 'gray'))
            
            # Legendenelemente erstellen
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor=color, label=str(style), markersize=10)
                for style, color in style_colors.items()
                if any(self.agents[agent_id].cognitive_architecture.primary_processing == style
                      for agent_id in self.social_network.nodes())
            ]
        
        else:
            # Standardfarbe
            color_map = plt.cm.viridis
            node_colors = [0.5] * len(self.social_network.nodes())
            legend_elements = []
            
        # Knotengröße basierend auf Anzahl der Verbindungen
        node_size = [300 * (1 + self.social_network.degree(node)) for node in self.social_network.nodes()]
        
        # Kanten basierend auf Verbindungsstärke
        edge_widths = [2 * self.social_network[u][v]['weight'] for u, v in self.social_network.edges()]
        
        # Netzwerk zeichnen
        if color_by in ['group', 'belief']:
            nodes = nx.draw_networkx_nodes(
                self.social_network, pos, 
                node_size=node_size,
                node_color=node_colors, 
                cmap=color_map, 
                alpha=0.8
            )
        else:
            nodes = nx.draw_networkx_nodes(
                self.social_network, pos, 
                node_size=node_size,
                node_color=node_colors, 
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
        
        if color_by in ['group', 'belief']:
            plt.colorbar(nodes)
        elif legend_elements:
            plt.legend(handles=legend_elements)
            
        plt.tight_layout()
        plt.show()

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
    if systematic_agent:
        print("\nVisualisierung der neuronalen Verarbeitung eines systematischen Denkers:")
        society.visualize_neural_processing(systematic_agent)
    
    if intuitive_agent:
        print("\nVisualisierung der neuronalen Verarbeitung eines intuitiven Denkers:")
        society.visualize_neural_processing(intuitive_agent)
    
    # Kognitive Stile vergleichen
    print("\nVergleich der kognitiven Stile:")
    society.visualize_cognitive_style_comparison()
    
    # Soziales Netzwerk nach kognitivem Stil
    print("\nSoziales Netzwerk nach kognitivem Stil:")
    society.visualize_social_network(color_by='cognitive_style')
    
    return society, results, analysis


if __name__ == "__main__":
    society, results, analysis = run_demo()
