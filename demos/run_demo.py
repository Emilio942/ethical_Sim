#!/usr/bin/env python3
"""
Ethische Agenten Simulation - Demo Script
==========================================

Dieses Script demonstriert die Funktionalität der ethischen Agenten-Simulation.
Es zeigt verschiedene Szenarien und Agent-Interaktionen.
"""

import sys
import os
import random
import numpy as np
from typing import List, Dict

# Lokale Imports mit korrigiertem Pfad
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from agents.agents import NeuralEthicalAgent
    from scenarios.scenarios import EthicalScenario, ScenarioGenerator, get_trolley_problem
    from society.neural_society import NeuralEthicalSociety
    from core.neural_types import NeuralProcessingType
    from visualization.visualization import EthicalSimulationVisualizer, quick_dashboard

    print("✅ Alle Module erfolgreich importiert!")
except ImportError as e:
    print(f"❌ Import-Fehler: {e}")
    print("Bitte stellen Sie sicher, dass alle Module korrekt implementiert sind.")
    print("Versuche ohne Visualisierung fortzufahren...")
    try:
        from agents.agents import NeuralEthicalAgent
        from scenarios.scenarios import EthicalScenario, ScenarioGenerator, get_trolley_problem
        from society.neural_society import NeuralEthicalSociety
        from core.neural_types import NeuralProcessingType

        print("✅ Kern-Module importiert (ohne Visualisierung)")
        EthicalSimulationVisualizer = None
        quick_dashboard = None
    except ImportError as e2:
        print(f"❌ Kritischer Import-Fehler: {e2}")
        sys.exit(1)


def create_demo_agents(num_agents: int = 5) -> List[NeuralEthicalAgent]:
    """Erstellt eine Gruppe von Demo-Agenten mit verschiedenen Persönlichkeiten."""
    agents = []

    # Vordefinierte Persönlichkeitstypen für Demonstration
    personality_templates = [
        {
            "name": "Analytiker",
            "traits": {
                "openness": 0.9,
                "conscientiousness": 0.8,
                "extroversion": 0.3,
                "agreeableness": 0.6,
                "neuroticism": 0.2,
            },
        },
        {
            "name": "Empath",
            "traits": {
                "openness": 0.7,
                "conscientiousness": 0.6,
                "extroversion": 0.8,
                "agreeableness": 0.9,
                "neuroticism": 0.4,
            },
        },
        {
            "name": "Pragmatiker",
            "traits": {
                "openness": 0.5,
                "conscientiousness": 0.9,
                "extroversion": 0.6,
                "agreeableness": 0.5,
                "neuroticism": 0.3,
            },
        },
        {
            "name": "Kreativer",
            "traits": {
                "openness": 0.95,
                "conscientiousness": 0.4,
                "extroversion": 0.7,
                "agreeableness": 0.7,
                "neuroticism": 0.6,
            },
        },
        {
            "name": "Konservativer",
            "traits": {
                "openness": 0.3,
                "conscientiousness": 0.8,
                "extroversion": 0.4,
                "agreeableness": 0.6,
                "neuroticism": 0.3,
            },
        },
    ]

    for i in range(min(num_agents, len(personality_templates))):
        template = personality_templates[i]
        agent = NeuralEthicalAgent(
            agent_id=f"agent_{i+1}_{template['name'].lower()}",
            personality_traits=template["traits"],
        )
        agents.append(agent)
        print(f"✅ Agent erstellt: {agent.agent_id} ({template['name']})")

    # Falls mehr Agenten gewünscht als Templates vorhanden
    for i in range(len(personality_templates), num_agents):
        agent = NeuralEthicalAgent(agent_id=f"agent_{i+1}_random")
        agents.append(agent)
        print(f"✅ Agent erstellt: {agent.agent_id} (Zufällig)")

    return agents


def demonstrate_scenario_processing(agents: List[NeuralEthicalAgent], scenario: EthicalScenario):
    """Demonstriert, wie Agenten ein ethisches Szenario verarbeiten."""
    print(f"\n{'='*60}")
    print(f"SZENARIO: {scenario.scenario_id}")
    print(f"{'='*60}")
    print(f"Beschreibung: {scenario.description}")
    print(f"Verfügbare Optionen: {list(scenario.options.keys())}")
    print(f"Emotionale Valenz: {scenario.emotional_valence:.2f}")
    print(f"Komplexität: {scenario.complexity:.2f}")
    print(f"Zeitdruck: {scenario.time_pressure:.2f}")
    print(f"Unsicherheit: {scenario.uncertainty:.2f}")

    print(f"\n{'Agent Entscheidungen:':-^60}")

    # Sammle Entscheidungsdaten für Visualisierung
    decisions = {}

    for agent in agents:
        try:
            # Hier würde die Entscheidungslogik des Agenten aufgerufen werden
            # Da die vollständige Implementierung noch nicht abgeschlossen ist,
            # simulieren wir eine Entscheidung

            available_options = list(scenario.options.keys())
            decision = random.choice(available_options)
            confidence = random.uniform(0.3, 0.9)

            print(f"\n🤖 {agent.agent_id}:")
            print(f"   Entscheidung: {decision}")
            print(f"   Konfidenz: {confidence:.2f}")
            print(f"   Verarbeitungstyp: {agent.cognitive_architecture.primary_processing}")

            # Zeige Persönlichkeitseinfluss
            key_traits = []
            for trait, value in agent.personality_traits.items():
                if value > 0.7:
                    key_traits.append(f"{trait.capitalize()}: {value:.2f}")

            if key_traits:
                print(f"   Dominante Traits: {', '.join(key_traits)}")

            # Speichere für Visualisierung
            decisions[agent.agent_id] = {
                "decision": decision,
                "confidence": confidence,
                "processing_type": agent.cognitive_architecture.primary_processing,
                "personality": agent.personality_traits,
            }

        except Exception as e:
            print(f"❌ Fehler bei Agent {agent.agent_id}: {e}")

    # Visualisierung anzeigen (falls verfügbar)
    if EthicalSimulationVisualizer is not None:
        try:
            visualizer = EthicalSimulationVisualizer()
            visualizer.plot_scenario_decisions(scenario, decisions)
        except Exception as e:
            print(f"⚠️ Visualisierung nicht verfügbar: {e}")

    return decisions


def demonstrate_social_dynamics(society: NeuralEthicalSociety):
    """Demonstriert soziale Dynamiken in der Gesellschaft."""
    print(f"\n{'='*60}")
    print(f"SOZIALE DYNAMIKEN")
    print(f"{'='*60}")

    print(f"Anzahl Agenten: {len(society.agents)}")
    print(f"Netzwerk-Knoten: {society.social_network.number_of_nodes()}")
    print(f"Netzwerk-Kanten: {society.social_network.number_of_edges()}")

    # Analysiere Agent-Typen
    processing_types = {}
    for agent in society.agents.values():
        ptype = agent.cognitive_architecture.primary_processing
        processing_types[ptype] = processing_types.get(ptype, 0) + 1

    print(f"\nVerteilung der Verarbeitungstypen:")
    for ptype, count in processing_types.items():
        print(f"  {ptype}: {count} Agenten")


def run_basic_demo():
    """Führt eine grundlegende Demonstration aus."""
    print("🚀 Starte Ethische Agenten Simulation Demo")
    print("=" * 60)

    # 1. Agenten erstellen
    print("\n📋 Schritt 1: Agenten erstellen")
    agents = create_demo_agents(5)

    # 2. Gesellschaft erstellen
    print("\n🏛️ Schritt 2: Gesellschaft erstellen")
    society = NeuralEthicalSociety()

    for agent in agents:
        society.add_agent(agent)

    print(f"✅ Gesellschaft mit {len(society.agents)} Agenten erstellt")

    # 3. Szenarien testen
    print("\n🎭 Schritt 3: Szenarien testen")

    # Trolley Problem
    trolley_scenario = get_trolley_problem()
    society.add_scenario(trolley_scenario)
    demonstrate_scenario_processing(agents, trolley_scenario)

    # Zufälliges Szenario
    generator = ScenarioGenerator()
    random_scenario = generator.generate_random_scenario()
    society.add_scenario(random_scenario)
    demonstrate_scenario_processing(agents, random_scenario)

    # 4. Soziale Dynamiken
    print("\n👥 Schritt 4: Soziale Dynamiken")
    demonstrate_social_dynamics(society)

    print(f"\n{'Demo beendet!':-^60}")


def run_interactive_demo():
    """Führt eine interaktive Demonstration aus."""
    print("🎮 Interaktive Demo gestartet")
    print("=" * 60)

    generator = ScenarioGenerator()
    available_templates = generator.get_available_templates()

    print("\nVerfügbare Szenario-Templates:")
    for i, template in enumerate(available_templates, 1):
        print(f"  {i}. {template}")

    try:
        choice = input(
            f"\nWählen Sie ein Szenario (1-{len(available_templates)}) oder 'q' zum Beenden: "
        )

        if choice.lower() == "q":
            print("Demo beendet.")
            return

        choice_idx = int(choice) - 1
        if 0 <= choice_idx < len(available_templates):
            template_name = available_templates[choice_idx]
            scenario = generator.create_scenario_from_template(template_name)

            # Agenten erstellen
            num_agents = int(input("Anzahl Agenten (1-10): ") or "3")
            agents = create_demo_agents(min(max(num_agents, 1), 10))

            # Szenario verarbeiten
            demonstrate_scenario_processing(agents, scenario)
        else:
            print("Ungültige Auswahl.")

    except (ValueError, KeyboardInterrupt):
        print("\nDemo beendet.")


def run_batch_analysis():
    """Führt eine Batch-Analyse mehrerer Szenarien aus."""
    print("📊 Batch-Analyse gestartet")
    print("=" * 60)

    # Agenten erstellen
    agents = create_demo_agents(10)
    society = NeuralEthicalSociety()

    for agent in agents:
        society.add_agent(agent)

    # Alle verfügbaren Szenarien testen
    generator = ScenarioGenerator()
    templates = generator.get_available_templates()

    print(f"\nTeste {len(templates)} Szenarien mit {len(agents)} Agenten...")

    for template_name in templates:
        scenario = generator.create_scenario_from_template(template_name)
        society.add_scenario(scenario)
        demonstrate_scenario_processing(agents, scenario)

    # Abschließende Analyse
    demonstrate_social_dynamics(society)


def run_visualization_demo():
    """Führt eine Demonstration der Visualisierungsfunktionen aus."""
    print("🎨 Visualisierungs-Demo gestartet")
    print("=" * 60)

    if EthicalSimulationVisualizer is None:
        print("❌ Visualisierung nicht verfügbar. Bitte installieren Sie die benötigten Pakete:")
        print("pip install matplotlib seaborn networkx")
        return

    # Agenten und Gesellschaft erstellen
    agents = create_demo_agents(8)  # Mehr Agenten für bessere Visualisierung
    society = NeuralEthicalSociety()

    for agent in agents:
        society.add_agent(agent)

    # Szenario erstellen
    trolley_scenario = get_trolley_problem()
    society.add_scenario(trolley_scenario)

    # Entscheidungen sammeln
    decisions = demonstrate_scenario_processing(agents, trolley_scenario)

    print(f"\n🎨 Erzeuge Visualisierungen...")

    try:
        visualizer = EthicalSimulationVisualizer()

        print("1. Erstelle Agent-Persönlichkeiten Plot...")
        visualizer.plot_agent_personalities(agents)

        print("2. Erstelle Soziales Netzwerk Plot...")
        visualizer.plot_social_network(society)

        print("3. Erstelle Simulation Dashboard...")
        visualizer.create_simulation_dashboard(society, trolley_scenario, decisions)

        # Frage ob Plots gespeichert werden sollen
        save_choice = input("\nMöchten Sie alle Plots speichern? (j/n): ").lower()
        if save_choice in ["j", "ja", "y", "yes"]:
            output_dir = visualizer.save_all_plots(society, trolley_scenario, decisions)
            print(f"📁 Plots gespeichert in: {output_dir}")

        print("✅ Visualisierungs-Demo abgeschlossen!")

    except Exception as e:
        print(f"❌ Fehler bei der Visualisierung: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Hauptfunktion des Demo-Scripts."""
    print("🧠 Ethische Agenten Simulation")
    print("=" * 60)
    print("Wählen Sie einen Demo-Modus:")
    print("1. Basis-Demo (automatisch)")
    print("2. Interaktive Demo")
    print("3. Batch-Analyse")
    print("4. Visualisierungs-Demo")
    print("5. Beenden")

    try:
        choice = input("\nIhre Wahl (1-5): ").strip()

        if choice == "1":
            run_basic_demo()
        elif choice == "2":
            run_interactive_demo()
        elif choice == "3":
            run_batch_analysis()
        elif choice == "4":
            run_visualization_demo()
        elif choice == "5":
            print("Auf Wiedersehen!")
        else:
            print("Ungültige Auswahl. Führe Basis-Demo aus...")
            run_basic_demo()

    except KeyboardInterrupt:
        print("\n\nDemo durch Benutzer unterbrochen.")
    except Exception as e:
        print(f"\n❌ Unerwarteter Fehler: {e}")
        print("Führe Basis-Demo als Fallback aus...")
        run_basic_demo()


if __name__ == "__main__":
    # Seed für Reproduzierbarkeit
    random.seed(42)
    np.random.seed(42)

    main()
