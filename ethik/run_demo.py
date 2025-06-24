import logging
import numpy as np # Wird in create_example_neural_society benötigt
import random # Wird in create_example_neural_society benötigt
import matplotlib.pyplot as plt # Hinzugefügt für manuellen Plot

# Importiere die modularisierten Klassen und Konstanten
from .neural_types import NeuralProcessingType
# CognitiveArchitecture wird indirekt über NeuralEthicalSociety -> NeuralEthicalAgent importiert
# from .cognitive_architecture import CognitiveArchitecture
from .beliefs import NeuralEthicalBelief
from .scenarios import EthicalScenario
from .agents import NeuralEthicalAgent
from .neural_society import NeuralEthicalSociety
from .analyzer import SimulationAnalyzer
from .visualizer import SimulationVisualizer
# Importiere Konstanten, falls sie direkt in run_demo oder create_example benötigt werden
from .constants import (SIMULATION_DEFAULT_SCENARIO_PROB,
                        SIMULATION_DEFAULT_SOCIAL_PROB,
                        SIMULATION_DEFAULT_REFLECTION_PROB)

# Logging-Konfiguration, falls noch nicht global gesetzt
# Es ist besser, das Logging im aufrufenden Skript zu konfigurieren,
# aber für eine Demo-Datei ist es hier OK.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_example_neural_society() -> NeuralEthicalSociety:
    """
    Erstellt eine einfache Beispielgesellschaft mit einigen Agenten und einem Szenario.
    Diese Funktion dient als Beispiel und für Testzwecke.
    (Extrahiert und angepasst aus se2.py)

    Returns:
        Eine initialisierte NeuralEthicalSociety mit Agenten und einem Szenario
    """
    society = NeuralEthicalSociety()

    # Einige Belief-Templates hinzufügen
    society.add_belief_template(
        name="Autonomie",
        category="Individualrechte",
        connections={"Solidarität": (0.3, -1), "Verantwortung": (0.4, 1)},
        associated_concepts={"Freiheit": 0.8, "Selbstbestimmung": 0.9},
        emotional_valence=0.6
    )

    society.add_belief_template(
        name="Solidarität",
        category="Gemeinschaftswerte",
        connections={"Autonomie": (0.3, -1), "Verantwortung": (0.5, 1)},
        associated_concepts={"Gemeinschaft": 0.7, "Zusammenhalt": 0.8},
        emotional_valence=0.5
    )

    society.add_belief_template(
        name="Verantwortung",
        category="Pflichten",
        connections={"Autonomie": (0.4, 1), "Solidarität": (0.5, 1)},
        associated_concepts={"Pflicht": 0.6, "Führsorge": 0.5},
        emotional_valence=0.4
    )
    society.add_belief_template(
        "individual_freedom", "Freiheit",
        connections={
            "government_control": (0.7, -1), "free_speech": (0.8, 1), "free_market": (0.6, 1)
        },
        associated_concepts={"liberty": 0.9, "independence": 0.8}, emotional_valence=0.7
    )
    society.add_belief_template(
        "government_control", "Ordnung",
        connections={
            "individual_freedom": (0.7, -1), "social_welfare": (0.6, 1)
        },
        associated_concepts={"security": 0.8, "stability": 0.7}, emotional_valence=-0.2
    )
    society.add_belief_template(
        "free_speech", "Freiheit",
        connections={"individual_freedom": (0.8, 1), "hate_speech_laws": (0.7, -1)},
        associated_concepts={"expression": 0.9, "democracy": 0.6}, emotional_valence=0.8
    )
    society.add_belief_template(
        "hate_speech_laws", "Gerechtigkeit",
        connections={"free_speech": (0.7, -1), "equality": (0.6, 1)},
        associated_concepts={"protection": 0.8, "respect": 0.7}, emotional_valence=0.3
    )
    society.add_belief_template(
        "equality", "Gerechtigkeit",
        connections={"social_welfare": (0.6, 1), "meritocracy": (0.5, -1)},
        associated_concepts={"fairness": 0.9, "rights": 0.8}, emotional_valence=0.7
    )
    society.add_belief_template(
        "social_welfare", "Gemeinschaft",
        connections={"government_control": (0.6, 1), "equality": (0.6, 1)},
        associated_concepts={"care": 0.9, "support": 0.8}, emotional_valence=0.6
    )
    society.add_belief_template(
        "meritocracy", "Leistung",
        connections={"equality": (0.5,-1)},
        associated_concepts={"achievement":0.8, "effort":0.7}, emotional_valence=0.5
    )

    # Agenten generieren
    society.generate_diverse_society(num_agents=20, num_archetypes=4, similarity_range=(0.5,0.9))

    covid_scenario = EthicalScenario(
        scenario_id="covid_dilemma_restrictions",
        description="Eine Pandemie erfordert möglicherweise erneute Einschränkungen persönlicher Freiheiten zum Schutz der Gemeinschaft.",
        relevant_beliefs={"Autonomie": 0.9, "Solidarität": 0.9, "Verantwortung": 0.8, "government_control": 0.7, "individual_freedom": 0.8},
        options={
            "strenge_einschraenkungen": {
                "Autonomie": -0.7, "individual_freedom": -0.6, "Solidarität": 0.8, "Verantwortung": 0.6, "government_control": 0.5
            },
            "empfehlungen_und_tests": {
                "Autonomie": 0.3, "individual_freedom": 0.4, "Solidarität": 0.2, "Verantwortung": 0.3, "government_control": -0.2
            },
            "keine_staatlichen_massnahmen": {
                "Autonomie": 0.8, "individual_freedom": 0.9, "Solidarität": -0.7, "Verantwortung": -0.5, "government_control": -0.8
            }
        },
        option_attributes={
            "strenge_einschraenkungen": {"risks": 0.3, "group_norms": {"Kollektivisten": 0.7, "Individualisten": -0.6}},
            "empfehlungen_und_tests": {"risks": 0.5, "group_norms": {"Kollektivisten": 0.1, "Individualisten": 0.3}},
            "keine_staatlichen_massnahmen": {"risks": 0.8, "group_norms": {"Kollektivisten": -0.8, "Individualisten": 0.7}}
        },
        outcome_feedback={
            "strenge_einschraenkungen": {"Solidarität": 0.1, "Autonomie": -0.1, "government_control": 0.05},
            "empfehlungen_und_tests": {"Verantwortung": 0.05, "Solidarität": 0.05},
            "keine_staatlichen_massnahmen": {"individual_freedom": 0.1, "Solidarität": -0.1}
        },
        moral_implications={
            "strenge_einschraenkungen": {"care": 0.6, "liberty": -0.5, "authority": 0.3},
            "empfehlungen_und_tests": {"care": 0.2, "liberty": 0.2},
            "keine_staatlichen_massnahmen": {"liberty": 0.7, "care": -0.4}
        },
        emotional_valence=-0.4,
        narrative_elements={
            "characters": ["Bürger", "Expertenrat", "Politiker"],
            "conflict": "Gesundheitsschutz vs. Freiheit und Wirtschaft",
            "context": "Anhaltende Pandemie-Situation mit neuen Varianten",
            "coherence": 0.75
        }
    )
    society.add_scenario(covid_scenario)

    agent_ids_list = list(society.agents.keys())
    if len(agent_ids_list) >= 2:
        half_point = len(agent_ids_list) // 2
        society.add_group("Gruppe_A", agent_ids_list[:half_point])
        society.add_group("Gruppe_B", agent_ids_list[half_point:])

    if not society.social_network.edges():
        society.generate_realistic_social_network()

    return society


def run_simulation_and_analysis():
    """
    Führt die Simulation aus, analysiert die Ergebnisse und visualisiert sie.
    """
    logging.info("Erstelle Beispielgesellschaft für die Demo...")
    society = create_example_neural_society()
    if not society.agents:
        logging.error("Gesellschaft konnte nicht mit Agenten initialisiert werden. Abbruch.")
        return None, None, None
    logging.info(f"Gesellschaft erstellt: {society}")

    logging.info("Starte robuste Simulation...")
    results = society.run_robust_simulation(
        num_steps=15,
        scenario_probability=SIMULATION_DEFAULT_SCENARIO_PROB,
        social_influence_probability=SIMULATION_DEFAULT_SOCIAL_PROB,
        reflection_probability=SIMULATION_DEFAULT_REFLECTION_PROB
    )
    logging.info("Simulation abgeschlossen.")

    logging.info("Analysiere Ergebnisse...")
    analyzer = SimulationAnalyzer(society, results)

    logging.info("Visualisiere Ergebnisse...")
    visualizer = SimulationVisualizer(society, analyzer)

    example_belief = next(iter(society.belief_templates.keys()), None)
    example_scenario_id = next(iter(society.scenarios.keys()), None)

    if example_belief:
        visualizer.plot_belief_evolution(example_belief, show_mean=True, show_styles=True)
        # visualizer.plot_polarization_trend(belief_name=example_belief, metric='variance') # Weniger informativ als Belief Evo
        # visualizer.plot_ensemble_variance(example_belief) # Nur sinnvoll wenn Ensemble > 1
        pass


    if example_scenario_id:
        visualizer.plot_decision_distribution(example_scenario_id)
        visualizer.plot_cognitive_style_decision_comparison(example_scenario_id)

    visualizer.visualize_social_network(color_by='cognitive_style', show_clusters=True, step=-1)
    # visualizer.visualize_social_network(color_by='group', step=-1) # Weniger interessant für Demo

    if society.agents:
        first_agent_id = list(society.agents.keys())[0]
        try:
            agent_to_plot = society.agents[first_agent_id]
            biases = agent_to_plot.cognitive_architecture.cognitive_biases
            if biases:
                plt.figure(figsize=(10, 4)) # Kleinerer Plot
                plt.barh(list(biases.keys()), list(biases.values()), color='salmon', height=0.5)
                plt.xlabel('Ausprägung')
                plt.title(f'Kognitive Biases für Agent {first_agent_id}')
                plt.tight_layout()
                plt.show()
        except Exception as e:
            logging.warning(f"Konnte Bias-Plot nicht erstellen: {e}")

    logging.info("Demo-Lauf und Analyse abgeschlossen.")
    return society, results, analyzer


if __name__ == "__main__":
    # Führe die Demo aus, wenn das Skript direkt gestartet wird.
    # Dies erlaubt es, die Simulation durch `python -m ethik.run_demo` zu starten.
    society_instance, sim_results, analysis_results = run_simulation_and_analysis()

    # Hier könnten weitere Interaktionen oder Auswertungen folgen,
    # z.B. Speichern der Ergebnisse, spezifische Abfragen etc.
    if society_instance:
        print(f"\nSimulation mit {len(society_instance.agents)} Agenten und {len(society_instance.scenarios)} Szenarien beendet.")
        if sim_results and sim_results.get("validation"):
             print(f"Validierungs-Log enthält {len(sim_results['validation'])} Einträge.")
             num_errors = sum(1 for entry in sim_results['validation'] if entry.get("errors"))
             num_warnings = sum(1 for entry in sim_results['validation'] if entry.get("warnings"))
             print(f"Davon {num_errors} Fehler und {num_warnings} Warnungen.")
