#!/usr/bin/env python3
"""
Test-Script für Visualisierungen
================================

Testet die Visualisierungsfunktionalität ohne interaktive Eingaben.
"""

import sys
import os
import random
import numpy as np

# Lokale Imports
try:
    from agents import NeuralEthicalAgent
    from scenarios import get_trolley_problem
    from neural_society import NeuralEthicalSociety
    from visualization import EthicalSimulationVisualizer

    print("✅ Alle Module erfolgreich importiert!")
except ImportError as e:
    print(f"❌ Import-Fehler: {e}")
    sys.exit(1)


def create_test_agents(num_agents: int = 5):
    """Erstellt Test-Agenten."""
    agents = []
    personality_templates = [
        {
            "openness": 0.9,
            "conscientiousness": 0.8,
            "extroversion": 0.3,
            "agreeableness": 0.6,
            "neuroticism": 0.2,
        },
        {
            "openness": 0.7,
            "conscientiousness": 0.6,
            "extroversion": 0.8,
            "agreeableness": 0.9,
            "neuroticism": 0.4,
        },
        {
            "openness": 0.5,
            "conscientiousness": 0.9,
            "extroversion": 0.6,
            "agreeableness": 0.5,
            "neuroticism": 0.3,
        },
        {
            "openness": 0.95,
            "conscientiousness": 0.4,
            "extroversion": 0.7,
            "agreeableness": 0.7,
            "neuroticism": 0.6,
        },
        {
            "openness": 0.3,
            "conscientiousness": 0.8,
            "extroversion": 0.4,
            "agreeableness": 0.6,
            "neuroticism": 0.3,
        },
    ]

    for i in range(num_agents):
        if i < len(personality_templates):
            traits = personality_templates[i]
        else:
            traits = None

        agent = NeuralEthicalAgent(f"test_agent_{i+1}", traits)
        agents.append(agent)

    return agents


def test_visualizations():
    """Testet alle Visualisierungsfunktionen."""
    print("🎨 Teste Visualisierungssystem")
    print("=" * 50)

    # Set matplotlib backend for headless environment
    import matplotlib

    matplotlib.use("Agg")  # Use non-interactive backend

    # Erstelle Test-Daten
    agents = create_test_agents(6)
    society = NeuralEthicalSociety()

    for agent in agents:
        society.add_agent(agent)

    scenario = get_trolley_problem()
    society.add_scenario(scenario)

    # Simuliere Entscheidungen
    decisions = {}
    options = list(scenario.options.keys())

    for agent in agents:
        decision = random.choice(options)
        confidence = random.uniform(0.3, 0.9)

        decisions[agent.agent_id] = {
            "decision": decision,
            "confidence": confidence,
            "processing_type": agent.cognitive_architecture.primary_processing,
            "personality": agent.personality_traits,
        }

    # Teste Visualisierungen
    try:
        visualizer = EthicalSimulationVisualizer()

        print("1. ✅ Teste Persönlichkeits-Plot...")
        visualizer.plot_agent_personalities(agents, "test_personalities.png")

        print("2. ✅ Teste Netzwerk-Plot...")
        visualizer.plot_social_network(society, "test_network.png")

        print("3. ✅ Teste Entscheidungs-Plot...")
        visualizer.plot_scenario_decisions(scenario, decisions, "test_decisions.png")

        print("4. ✅ Teste Dashboard...")
        visualizer.create_simulation_dashboard(society, scenario, decisions, "test_dashboard.png")

        print("5. ✅ Teste Batch-Speicherung...")
        output_dir = visualizer.save_all_plots(society, scenario, decisions)

        print(f"\n🎉 Alle Visualisierungstests erfolgreich!")
        print(f"📁 Dateien gespeichert in: {output_dir}")

        # Liste erstellte Dateien auf
        if os.path.exists(output_dir):
            files = os.listdir(output_dir)
            print(f"\nErstellte Dateien ({len(files)}):")
            for file in sorted(files):
                if file.endswith(".png"):
                    file_path = os.path.join(output_dir, file)
                    size = os.path.getsize(file_path)
                    print(f"  📊 {file} ({size:,} bytes)")

        return True

    except Exception as e:
        print(f"❌ Fehler beim Testen der Visualisierungen: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Seed für reproduzierbare Ergebnisse
    random.seed(42)
    np.random.seed(42)

    success = test_visualizations()

    if success:
        print("\n✅ Visualisierungstest erfolgreich abgeschlossen!")
        sys.exit(0)
    else:
        print("\n❌ Visualisierungstest fehlgeschlagen!")
        sys.exit(1)
