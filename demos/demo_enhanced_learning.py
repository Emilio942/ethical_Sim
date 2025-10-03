#!/usr/bin/env python3
"""
Enhanced Demo für die ethische Agenten-Simulation mit erweiterten Lernmechanismen.

Zeigt die neuen Features:
- Reinforcement Learning
- Multi-Criteria Decision Making
- Uncertainty Handling
- Advanced Social Learning
- Temporal Dynamics
- Metacognitive Monitoring
"""

import numpy as np
import random
from typing import List, Dict
from agents import NeuralEthicalAgent
from beliefs import NeuralEthicalBelief
from scenarios import ScenarioGenerator
from neural_society import NeuralEthicalSociety


def create_enhanced_agent(agent_id: str, agent_type: str = "balanced") -> NeuralEthicalAgent:
    """Erstellt einen Agenten mit spezifischen Eigenschaften für die Enhanced Demo."""

    if agent_type == "systematic":
        traits = {
            "openness": 0.9,
            "conscientiousness": 0.9,
            "extroversion": 0.5,
            "agreeableness": 0.6,
            "neuroticism": 0.3,
        }
    elif agent_type == "intuitive":
        traits = {
            "openness": 0.7,
            "conscientiousness": 0.4,
            "extroversion": 0.8,
            "agreeableness": 0.7,
            "neuroticism": 0.6,
        }
    elif agent_type == "social":
        traits = {
            "openness": 0.6,
            "conscientiousness": 0.6,
            "extroversion": 0.9,
            "agreeableness": 0.9,
            "neuroticism": 0.4,
        }
    else:  # balanced
        traits = {
            "openness": 0.6,
            "conscientiousness": 0.6,
            "extroversion": 0.6,
            "agreeableness": 0.6,
            "neuroticism": 0.5,
        }

    agent = NeuralEthicalAgent(agent_id, traits)

    # Standard-Überzeugungen hinzufügen
    agent.add_belief(
        NeuralEthicalBelief(
            "Utilitarismus", np.random.uniform(0.3, 0.9), np.random.uniform(0.6, 0.9)
        )
    )
    agent.add_belief(
        NeuralEthicalBelief("Deontologie", np.random.uniform(0.3, 0.9), np.random.uniform(0.6, 0.9))
    )
    agent.add_belief(
        NeuralEthicalBelief("Tugendethik", np.random.uniform(0.3, 0.9), np.random.uniform(0.6, 0.9))
    )

    return agent


def demo_reinforcement_learning():
    """Demonstriert Reinforcement Learning Mechanismen."""
    print("=" * 60)
    print("🧠 REINFORCEMENT LEARNING DEMO")
    print("=" * 60)

    agent = create_enhanced_agent("rl_learner", "systematic")
    generator = ScenarioGenerator()

    print(f"Agent erstellt: {agent.agent_id}")
    print(f"Initial exploration rate: {agent.rl_system['exploration_rate']:.3f}")
    print(f"Initial learning rate: {agent.rl_system['learning_rate']:.3f}")

    # Mehrere Entscheidungen in ähnlichen Szenarien
    for round_num in range(5):
        print(f"\n--- Runde {round_num + 1} ---")

        scenario = generator.create_scenario_from_template("trolley_problem")
        decision = agent.make_decision(scenario, enhanced=True)

        print(f"Entscheidung: {decision['chosen_option']}")
        print(f"Konfidenz: {decision['confidence']:.3f}")
        print(f"Exploration verwendet: {decision['exploration_used']}")

        # Simuliere Feedback (zufällig für Demo)
        outcome_feedback = {
            "personal_outcome": np.random.uniform(-0.5, 0.8),
            "social_outcome": np.random.uniform(-0.3, 0.9),
            "moral_outcome": np.random.uniform(-0.2, 0.7),
            "risk_outcome": np.random.uniform(-0.8, 0.3),
        }

        learning_result = agent.learn_from_decision_outcome(
            scenario, decision["chosen_option"], outcome_feedback
        )

        print(f"RL Belohnung: {learning_result['reinforcement_learning']['total_reward']:.3f}")
        print(f"Neue exploration rate: {agent.rl_system['exploration_rate']:.3f}")

        # Zeige gelernte Aktionswerte
        if "trolley" in agent.rl_system["action_values"]:
            trolley_values = agent.rl_system["action_values"]["trolley"]
            print(f"Gelernte Werte: {trolley_values}")


def demo_multi_criteria_decision():
    """Demonstriert Multi-Criteria Decision Making."""
    print("\n" + "=" * 60)
    print("⚖️  MULTI-CRITERIA DECISION MAKING DEMO")
    print("=" * 60)

    agent = create_enhanced_agent("mc_agent", "balanced")
    generator = ScenarioGenerator()

    print(f"Agent erstellt: {agent.agent_id}")
    print(f"Initial criteria weights: {agent.decision_criteria['utility_weights']}")

    scenario = generator.create_scenario_from_template("autonomous_vehicle")
    decision = agent.make_decision(scenario, enhanced=True)

    print(f"\nSzenario: {scenario.scenario_id}")
    print(f"Entscheidung: {decision['chosen_option']}")
    print(f"Konfidenz: {decision['confidence']:.3f}")

    # Detaillierte Kriterien-Analyse
    if "criteria_evaluations" in decision:
        print("\nKriterien-Bewertungen:")
        for option, criteria in decision["criteria_evaluations"].items():
            print(f"  {option}:")
            for criterion, score in criteria.items():
                print(f"    {criterion}: {score:.3f}")

    print(f"\nFinale Option-Scores: {decision['option_scores']}")

    # Lerne aus Feedback und zeige Gewichts-Anpassungen
    outcome_feedback = {
        "personal_outcome": 0.4,
        "social_outcome": 0.8,
        "moral_outcome": 0.6,
        "risk_outcome": 0.1,
    }

    learning_result = agent.learn_from_decision_outcome(
        scenario, decision["chosen_option"], outcome_feedback
    )

    print(f"\nNach dem Lernen:")
    print(f"Neue criteria weights: {agent.decision_criteria['utility_weights']}")
    if "criteria_learning" in learning_result:
        print(f"Gewichts-Änderungen: {learning_result['criteria_learning']}")


def demo_uncertainty_handling():
    """Demonstriert Uncertainty und Ambiguity Handling."""
    print("\n" + "=" * 60)
    print("🤔 UNCERTAINTY HANDLING DEMO")
    print("=" * 60)

    # Agent mit hoher Ambiguity Tolerance
    tolerant_agent = create_enhanced_agent("tolerant_agent", "intuitive")
    tolerant_agent.uncertainty_handling["ambiguity_tolerance"] = 0.9
    tolerant_agent.uncertainty_handling["epistemic_humility"] = 0.8

    # Agent mit niedriger Ambiguity Tolerance
    intolerant_agent = create_enhanced_agent("intolerant_agent", "systematic")
    intolerant_agent.uncertainty_handling["ambiguity_tolerance"] = 0.2
    intolerant_agent.uncertainty_handling["epistemic_humility"] = 0.3

    generator = ScenarioGenerator()
    scenario = generator.create_scenario_from_template("privacy_vs_security")

    print(f"Szenario: {scenario.scenario_id}")

    for agent, label in [
        (tolerant_agent, "Unsicherheits-Tolerant"),
        (intolerant_agent, "Unsicherheits-Intolerant"),
    ]:
        print(f"\n--- {label} Agent ---")
        print(f"Ambiguity tolerance: {agent.uncertainty_handling['ambiguity_tolerance']:.3f}")

        decision = agent.make_decision(scenario, enhanced=True)

        print(f"Entscheidung: {decision['chosen_option']}")
        print(f"Konfidenz: {decision['confidence']:.3f}")

        if "uncertainty_info" in decision:
            print(f"Unsicherheitsschätzungen: {len(decision['uncertainty_info'])} Überzeugungen")
            avg_uncertainty = (
                np.mean(list(decision["uncertainty_info"].values()))
                if decision["uncertainty_info"]
                else 0
            )
            print(f"Durchschnittliche Unsicherheit: {avg_uncertainty:.3f}")


def demo_social_learning():
    """Demonstriert Advanced Social Learning."""
    print("\n" + "=" * 60)
    print("👥 SOCIAL LEARNING DEMO")
    print("=" * 60)

    # Erstelle eine kleine Gesellschaft mit verschiedenen Agenten-Typen
    agents = [
        create_enhanced_agent("social_1", "social"),
        create_enhanced_agent("systematic_1", "systematic"),
        create_enhanced_agent("intuitive_1", "intuitive"),
        create_enhanced_agent("balanced_1", "balanced"),
    ]

    # Soziale Verbindungen erstellen
    for i, agent in enumerate(agents):
        for j, other_agent in enumerate(agents):
            if i != j:
                # Zufällige Verbindungsstärke
                connection_strength = np.random.uniform(0.3, 0.9)
                agent.add_social_connection(other_agent.agent_id, connection_strength)

    # Fokus-Agent für detaillierte Analyse
    focus_agent = agents[0]
    other_agents = agents[1:]

    print(f"Focus Agent: {focus_agent.agent_id}")
    print(f"Soziale Verbindungen: {focus_agent.social_connections}")

    # Vor sozialem Lernen
    print(f"\nVor sozialem Lernen:")
    for belief_name, belief in focus_agent.beliefs.items():
        print(f"  {belief_name}: {belief.strength:.3f}")

    # Soziales Lernen durchführen
    social_changes = focus_agent.advanced_social_learning(other_agents)

    print(f"\nNach sozialem Lernen:")
    for belief_name, belief in focus_agent.beliefs.items():
        print(f"  {belief_name}: {belief.strength:.3f}")

    print(f"\nSoziale Änderungen: {social_changes}")
    print(f"Trust Network: {focus_agent.social_learning['trust_network']}")


def demo_temporal_dynamics():
    """Demonstriert Temporal Belief Dynamics."""
    print("\n" + "=" * 60)
    print("⏰ TEMPORAL DYNAMICS DEMO")
    print("=" * 60)

    agent = create_enhanced_agent("temporal_agent", "balanced")

    print(f"Agent erstellt: {agent.agent_id}")

    # Verfolge eine spezifische Überzeugung über Zeit
    belief_name = "Utilitarismus"
    initial_strength = agent.beliefs[belief_name].strength

    print(f"\nInitiale Stärke von {belief_name}: {initial_strength:.3f}")

    # Simuliere zeitliche Entwicklung
    strengths_over_time = [initial_strength]

    for time_step in range(10):
        # Füge Momentum hinzu (simuliert anhaltende Einflüsse)
        if time_step == 3:
            agent.temporal_dynamics["belief_momentum"][belief_name] = 0.2
            print(f"  Zeit {time_step}: Momentum hinzugefügt")

        if time_step == 7:
            agent.temporal_dynamics["habituation_effects"][belief_name] = 0.8
            print(f"  Zeit {time_step}: Habituation-Effekt")

        # Zeitliche Dynamik anwenden
        temporal_changes = agent.temporal_belief_dynamics(1.0)

        current_strength = agent.beliefs[belief_name].strength
        strengths_over_time.append(current_strength)

        if belief_name in temporal_changes:
            change = temporal_changes[belief_name]
            print(f"  Zeit {time_step + 1}: {current_strength:.3f} (Δ: {change:.3f})")

    print(f"\nStärke-Verlauf: {[f'{s:.3f}' for s in strengths_over_time[:5]]}...")


def demo_metacognitive_monitoring():
    """Demonstriert Metacognitive Monitoring."""
    print("\n" + "=" * 60)
    print("🧐 METACOGNITIVE MONITORING DEMO")
    print("=" * 60)

    # Agent mit hoher metacognitiver Bewusstheit
    agent = create_enhanced_agent("meta_agent", "systematic")
    agent.metacognition["thinking_about_thinking"] = True
    agent.metacognition["strategy_monitoring"] = 0.9
    agent.metacognition["error_detection"] = 0.8

    print(f"Agent erstellt: {agent.agent_id}")
    print(f"Metacognitive features: {agent.metacognition}")

    generator = ScenarioGenerator()

    # Simuliere eine Sequenz von Entscheidungen mit unterschiedlichen Ergebnissen
    scenarios = ["trolley_problem", "autonomous_vehicle", "privacy_vs_security"]

    for i, template in enumerate(scenarios):
        print(f"\n--- Entscheidung {i + 1}: {template} ---")

        scenario = generator.create_scenario_from_template(template)
        decision = agent.make_decision(scenario, enhanced=True)

        print(f"Entscheidung: {decision['chosen_option']}")
        print(f"Konfidenz: {decision['confidence']:.3f}")
        print(f"Kognitive Dissonanz: {decision['cognitive_dissonance']:.3f}")

        # Simuliere unterschiedliche Qualität der Ergebnisse
        if i == 0:  # Gutes Ergebnis
            outcome_feedback = {
                "personal_outcome": 0.8,
                "social_outcome": 0.9,
                "moral_outcome": 0.7,
                "risk_outcome": 0.2,
            }
        elif i == 1:  # Schlechtes Ergebnis
            outcome_feedback = {
                "personal_outcome": -0.3,
                "social_outcome": -0.5,
                "moral_outcome": -0.2,
                "risk_outcome": -0.8,
            }
        else:  # Mittleres Ergebnis
            outcome_feedback = {
                "personal_outcome": 0.1,
                "social_outcome": 0.2,
                "moral_outcome": 0.3,
                "risk_outcome": 0.0,
            }

        learning_result = agent.learn_from_decision_outcome(
            scenario, decision["chosen_option"], outcome_feedback
        )

        if "metacognitive_adjustments" in decision:
            print(f"Metacognitive Anpassungen: {decision['metacognitive_adjustments']}")

        if "meta_learning" in learning_result:
            meta_info = learning_result["meta_learning"]
            print(f"Meta-Learning Effektivität: {meta_info['effectiveness']:.3f}")
            print(f"Angepasste Lernrate: {meta_info['learning_rate_adjustment']:.3f}")


def main():
    """Hauptdemo für erweiterte Lernmechanismen."""
    print("🚀 ENHANCED LEARNING MECHANISMS DEMO")
    print("Demonstriert alle neuen fortgeschrittenen Lernfähigkeiten der ethischen Agenten")

    try:
        demo_reinforcement_learning()
        demo_multi_criteria_decision()
        demo_uncertainty_handling()
        demo_social_learning()
        demo_temporal_dynamics()
        demo_metacognitive_monitoring()

        print("\n" + "=" * 60)
        print("✅ ALLE ENHANCED LEARNING DEMOS ERFOLGREICH ABGESCHLOSSEN!")
        print("=" * 60)
        print("\nDie erweiterten Lernmechanismen umfassen:")
        print("• Reinforcement Learning mit Exploration/Exploitation")
        print("• Multi-Criteria Decision Making mit Trade-off Analysis")
        print("• Uncertainty und Ambiguity Handling")
        print("• Advanced Social Learning mit Trust und Reputation")
        print("• Temporal Belief Dynamics mit Momentum und Habituation")
        print("• Metacognitive Monitoring und Strategy Switching")

    except Exception as e:
        print(f"❌ Fehler in der Enhanced Demo: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
