import numpy as np
from typing import Dict, Optional

# Import from project modules
from core.neural_types import NeuralProcessingType
from core.config import config

class CognitiveArchitecture:
    """Modelliert die kognitive Architektur eines Agenten."""

    def __init__(
        self,
        primary_processing: str = NeuralProcessingType.SYSTEMATIC,
        secondary_processing: str = NeuralProcessingType.INTUITIVE,
        processing_balance: float = 0.5,
    ):
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
            "anchoring_bias": np.random.beta(5, 5),  # Ankereffekt
            "authority_bias": np.random.beta(5, 5),  # Autoritätsverzerrung
            "ingroup_bias": np.random.beta(5, 5),  # Eigengruppenfavorisierung
        }

        # Emotionale Parameter
        self.emotional_parameters = {
            "emotional_reactivity": np.random.beta(5, 5),  # Emotionale Reaktivität
            "emotional_regulation": np.random.beta(5, 5),  # Emotionsregulation
            "empathy": np.random.beta(5, 5),  # Empathie
            "negativity_bias": np.random.beta(6, 4),  # Negativitätsverzerrung
        }

        # Bayes'sche Verarbeitungsparameter
        self.bayesian_parameters = {
            "prior_strength": np.random.beta(5, 5),  # Stärke der Vorannahmen
            "evidence_threshold": np.random.beta(5, 5),  # Schwellenwert für Beweisannahme
            "update_rate": np.random.beta(5, 5),  # Geschwindigkeit des Belief-Updates
        }

        # Neuronale Aktivierungsfunktionen für verschiedene Verarbeitungstypen
        self.activation_functions = {
            NeuralProcessingType.SYSTEMATIC: self._systematic_activation,
            NeuralProcessingType.INTUITIVE: self._intuitive_activation,
            NeuralProcessingType.ASSOCIATIVE: self._associative_activation,
            NeuralProcessingType.ANALOGICAL: self._analogical_activation,
            NeuralProcessingType.EMOTIONAL: self._emotional_activation,
            NeuralProcessingType.NARRATIVE: self._narrative_activation,
        }

    def _systematic_activation(
        self, inputs: Dict[str, float], context: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Systematische, schrittweise Aktivierungsfunktion.
        Betont logische Konsistenz und sequentielle Verarbeitung.
        """
        results = {}
        for key, value in inputs.items():
            # Systematisches Denken reduziert den Einfluss von Verzerrungen
            bias_reduction = config.cognitive.BIAS_REDUCTION_FACTOR  # Reduktion der Verzerrungen

            # Angepasster Wert mit reduziertem Bias-Einfluss
            results[key] = value * (
                1.0 - self.cognitive_biases.get("confirmation_bias", 0) * (1.0 - bias_reduction)
            )
        return results

    def _intuitive_activation(
        self, inputs: Dict[str, float], context: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Intuitive, schnelle Aktivierungsfunktion.
        Betont Gesamteindrücke und unmittelbare Reaktionen.
        """
        results = {}
        for key, value in inputs.items():
            # Intuitive Reaktionen sind stärker durch Verfügbarkeitsheuristiken beeinflusst
            availability_effect = self.cognitive_biases.get("availability_bias", 0) * config.cognitive.AVAILABILITY_BIAS_WEIGHT

            # Angepasster Wert mit verstärktem Einfluss verfügbarer Informationen
            if context and key in context:
                context_influence = context[key] * availability_effect
                results[key] = value * (1.0 + context_influence)
            else:
                results[key] = value
        return results

    def _associative_activation(
        self, inputs: Dict[str, float], context: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Assoziative Aktivierungsfunktion mit netzwerkartiger Aktivierung.
        Aktiviert verbundene Konzepte basierend auf semantischer Nähe.
        """
        results = {}

        # Schwellenwert für Aktivierungsübertragung
        activation_threshold = config.cognitive.ACTIVATION_THRESHOLD

        for key, value in inputs.items():
            # Basisaktivierung
            results[key] = value

            # Aktivierung verbundener Konzepte (falls kontext-informationen verfügbar)
            if context:
                for other_key, other_value in context.items():
                    if (
                        other_key != key
                        and isinstance(other_value, (int, float))
                        and other_value > activation_threshold
                    ):
                        # Einfaches assoziatives Spreading
                        association_strength = other_value * config.cognitive.ASSOCIATION_DAMPING_FACTOR  # Dämpfungsfaktor
                        results[key] = max(results[key], value * (1.0 + association_strength))

        return results

    def _analogical_activation(
        self, inputs: Dict[str, float], context: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Aktivierungsfunktion basierend auf Analogien und Ähnlichkeiten.
        Betont strukturelle Ähnlichkeiten zwischen Konzepten.
        """
        results = {}

        # Parameter für die Analogiestärke
        analogy_strength = config.cognitive.ANALOGY_STRENGTH

        for key, value in inputs.items():
            # Basisaktivierung
            results[key] = value

            # Analogien verstärken ähnliche Konzepte (Simulation)
            if context:
                analogical_boost = (
                    sum(
                        config.cognitive.ANALOGY_CONTEXT_WEIGHT * v
                        for k, v in context.items()
                        if k != key and isinstance(v, (int, float))
                    )
                    * analogy_strength
                )
                results[key] *= 1.0 + analogical_boost

        return results

    def _emotional_activation(
        self, inputs: Dict[str, float], context: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Emotionsgesteuerte Aktivierungsfunktion.
        Betont emotionale Reaktionen und somatische Marker.
        """
        results = {}

        # Emotionale Reaktivität und Regulation beeinflussen die Aktivierung
        reactivity = self.emotional_parameters.get("emotional_reactivity", config.cognitive.DEFAULT_EMOTIONAL_REACTIVITY)
        regulation = self.emotional_parameters.get("emotional_regulation", config.cognitive.DEFAULT_EMOTIONAL_REGULATION)
        negativity_bias = self.emotional_parameters.get("negativity_bias", config.cognitive.DEFAULT_NEGATIVITY_BIAS)

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
                regulated_effect = emotional_effect * (1.0 - regulation * config.cognitive.EMOTIONAL_REGULATION_DAMPING)

                # Angepasster Wert mit emotionalem Einfluss
                results[key] = base_activation * (1.0 + regulated_effect)
            else:
                results[key] = base_activation

        return results

    def _narrative_activation(
        self, inputs: Dict[str, float], context: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Narrativbasierte Aktivierungsfunktion.
        Betont Kohärenz und Stimmigkeit in einer Geschichte.
        """
        results = {}

        # Kohärenz-Parameter simulieren
        coherence_bias = config.cognitive.COHERENCE_BIAS

        for key, value in inputs.items():
            # Basisaktivierung
            results[key] = value

            # Kohärenzeffekte (Simulation)
            if context and "narrative_coherence" in context:
                coherence = context["narrative_coherence"]  # 0 bis 1

                # Verstärkung für kohärente Narrative
                narrative_effect = coherence * coherence_bias
                results[key] *= 1.0 + narrative_effect

        return results

    def process_information(
        self, inputs: Dict[str, float], context: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
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
            secondary_results = self.activation_functions[self.secondary_processing](
                inputs, context
            )

            # Gewichtete Kombination der Ergebnisse
            results = {}
            for key in inputs.keys():
                primary_value = primary_results.get(key, 0)
                secondary_value = secondary_results.get(key, 0)

                # Gewichtete Kombination basierend auf processing_balance
                results[key] = (
                    self.processing_balance * primary_value
                    + (1.0 - self.processing_balance) * secondary_value
                )
        else:
            results = primary_results

        return results

    def apply_bayesian_update(
        self, prior_belief: float, evidence_strength: float, evidence_direction: int
    ) -> float:
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
            evidence_weight = config.cognitive.SYSTEMATIC_EVIDENCE_WEIGHT  # Systematisches Denken gewichtet Beweise stärker
        elif self.primary_processing == NeuralProcessingType.EMOTIONAL:
            evidence_weight = config.cognitive.EMOTIONAL_EVIDENCE_WEIGHT  # Emotionales Denken gewichtet Beweise weniger
        else:
            evidence_weight = config.cognitive.NEUTRAL_EVIDENCE_WEIGHT  # Neutrale Gewichtung

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
        anchoring_effect = adjusted_prior_strength * (prior_belief - 0.5) * config.cognitive.ANCHORING_EFFECT_WEIGHT

        # Neue Überzeugungsstärke
        new_belief = prior_belief + belief_change + anchoring_effect

        return float(np.clip(new_belief, 0.0, 1.0))

    def __str__(self):
        """String-Repräsentation der kognitiven Architektur."""
        return f"Kognitive Architektur: {self.primary_processing} ({self.processing_balance:.2f}) / {self.secondary_processing} ({1-self.processing_balance:.2f})"
