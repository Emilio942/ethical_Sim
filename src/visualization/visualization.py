"""
Visualisierung für die ethische Agenten-Simulation
==================================================

Dieses Modul stellt Funktionen zur grafischen Darstellung von
Simulationsergebnissen, Agent-Netzwerken und Entscheidungsverläufen bereit.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
import os
from datetime import datetime

# Set style for better looking plots
plt.style.use("seaborn-v0_8" if "seaborn-v0_8" in plt.style.available else "default")
sns.set_palette("husl")

# Configure matplotlib for non-interactive environments
import warnings

warnings.filterwarnings("ignore", "FigureCanvasAgg is non-interactive")


def safe_show_plot():
    """Safely show plot only if interactive backend is available."""
    try:
        if plt.get_backend() != "Agg":
            safe_show_plot()
        else:
            print("📊 Plot erstellt (non-interactive mode)")
    except:
        print("📊 Plot erstellt (Backend nicht verfügbar)")
    finally:
        plt.close()  # Speicher freigeben


class EthicalSimulationVisualizer:
    """Hauptklasse für die Visualisierung der ethischen Agenten-Simulation."""

    def __init__(self, output_dir: str = "output", figsize: Tuple[int, int] = (12, 8)):
        """
        Initialisiert den Visualizer.

        Args:
            output_dir: Verzeichnis für die Ausgabedateien
            figsize: Standard-Figurengröße für Plots
        """
        self.output_dir = output_dir
        self.figsize = figsize

        # Erstelle Output-Verzeichnis falls es nicht existiert
        os.makedirs(self.output_dir, exist_ok=True)

        # Farbschema für verschiedene Verarbeitungstypen
        self.processing_colors = {
            "systematic": "#2E86AB",  # Blau
            "intuitive": "#A23B72",  # Lila
            "associative": "#F18F01",  # Orange
            "analogical": "#C73E1D",  # Rot
            "emotional": "#8E44AD",  # Violett
            "narrative": "#16A085",  # Grün
        }

        # Farbschema für Persönlichkeitsmerkmale
        self.personality_colors = {
            "openness": "#3498DB",
            "conscientiousness": "#2ECC71",
            "extroversion": "#F39C12",
            "agreeableness": "#E74C3C",
            "neuroticism": "#9B59B6",
        }

    def plot_agent_personalities(self, agents: List[Any], save_path: Optional[str] = None) -> None:
        """
        Visualisiert die Persönlichkeitsprofile der Agenten.

        Args:
            agents: Liste der Agenten
            save_path: Pfad zum Speichern der Grafik
        """
        # Create figure with specific layout for radar chart
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle("Persönlichkeitsprofile der Agenten", fontsize=16, fontweight="bold")

        # Grid layout
        gs = fig.add_gridspec(2, 3)

        # Sammle Persönlichkeitsdaten
        personality_data: Dict[str, List[float]] = {
            trait: []
            for trait in [
                "openness",
                "conscientiousness",
                "extroversion",
                "agreeableness",
                "neuroticism",
            ]
        }
        agent_names = []

        for agent in agents:
            agent_names.append(agent.agent_id)
            for trait, value in agent.personality_traits.items():
                if trait in personality_data:
                    personality_data[trait].append(value)

        # Radar-Chart für durchschnittliche Persönlichkeit (Polar projection)
        ax_radar = fig.add_subplot(gs[0, 0], polar=True)
        traits = list(personality_data.keys())
        avg_values = [np.mean(personality_data[trait]) for trait in traits]

        angles = np.linspace(0, 2 * np.pi, len(traits), endpoint=False).tolist()
        avg_values += avg_values[:1]  # Schließe den Kreis
        angles += angles[:1]

        ax_radar.plot(angles, avg_values, "o-", linewidth=2, color="#3498DB")
        ax_radar.fill(angles, avg_values, alpha=0.25, color="#3498DB")
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels([t.capitalize() for t in traits])
        ax_radar.set_ylim(0, 1)
        ax_radar.set_title("Durchschnittspersönlichkeit", fontweight="bold", pad=20)
        ax_radar.grid(True)

        # Histogramme für einzelne Traits
        trait_positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

        for i, (trait, values) in enumerate(personality_data.items()):
            if i < len(trait_positions):
                row, col = trait_positions[i]
                ax = fig.add_subplot(gs[row, col])

                color = self.personality_colors.get(trait, "#95A5A6")
                ax.hist(values, bins=10, alpha=0.7, color=color, edgecolor="black")
                ax.set_title(f"{trait.capitalize()}", fontweight="bold")
                ax.set_xlabel("Wert")
                ax.set_ylabel("Anzahl Agenten")
                ax.set_xlim(0, 1)
                ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Persönlichkeitsplot gespeichert: {save_path}")

        safe_show_plot()

    def plot_social_network(self, society: Any, save_path: Optional[str] = None) -> None:
        """
        Visualisiert das soziale Netzwerk der Agenten.

        Args:
            society: Die Gesellschaft mit dem sozialen Netzwerk
            save_path: Pfad zum Speichern der Grafik
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle("Soziales Netzwerk der Agenten", fontsize=16, fontweight="bold")

        # Netzwerk-Layout berechnen
        G = society.social_network
        pos = nx.spring_layout(G, k=1, iterations=50)

        # Farben basierend auf Verarbeitungstyp
        node_colors = []
        for node in G.nodes():
            if node in society.agents:
                processing_type = str(
                    society.agents[node].cognitive_architecture.primary_processing
                ).lower()
                # Handle enum string representation if needed
                if "." in processing_type:
                    processing_type = processing_type.split(".")[-1]
                
                color = self.processing_colors.get(processing_type, "#95A5A6")
                node_colors.append(color)
            else:
                node_colors.append("#95A5A6")

        # Netzwerk-Darstellung
        nx.draw(
            G,
            pos,
            ax=ax1,
            node_color=node_colors,
            node_size=1000,
            with_labels=True,
            font_size=8,
            font_weight="bold",
            edge_color="gray",
            alpha=0.8,
        )

        ax1.set_title("Netzwerk-Struktur", fontweight="bold")

        # Legende für Verarbeitungstypen
        legend_elements = []
        for proc_type, color in self.processing_colors.items():
            legend_elements.append(patches.Patch(color=color, label=proc_type.capitalize()))

        ax1.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.15, 1))

        # Netzwerk-Statistiken
        ax2.axis("off")

        # Berechne Netzwerk-Metriken
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        density = nx.density(G) if num_nodes > 1 else 0

        # Zentralitätsmaße (nur wenn Kanten vorhanden)
        if num_edges > 0:
            centrality = nx.degree_centrality(G)
            avg_centrality = np.mean(list(centrality.values()))
            clustering = nx.average_clustering(G)
        else:
            avg_centrality = 0
            clustering = 0

        # Verarbeitungstyp-Verteilung
        processing_distribution: Dict[Any, int] = {}
        for agent in society.agents.values():
            proc_type = agent.cognitive_architecture.primary_processing
            processing_distribution[proc_type] = processing_distribution.get(proc_type, 0) + 1

        # Statistik-Text
        stats_text = f"""
Netzwerk-Statistiken:

Knoten: {num_nodes}
Kanten: {num_edges}
Dichte: {density:.3f}
Ø Zentralität: {avg_centrality:.3f}
Clustering: {clustering:.3f}

Verarbeitungstypen:
"""

        for proc_type, count in processing_distribution.items():
            percentage = (count / num_nodes) * 100
            stats_text += f"• {proc_type.capitalize()}: {count} ({percentage:.1f}%)\n"

        ax2.text(
            0.1,
            0.9,
            stats_text,
            transform=ax2.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Netzwerk-Plot gespeichert: {save_path}")

        safe_show_plot()

    def plot_scenario_decisions(
        self, scenario: Any, decisions: Dict[str, Dict], save_path: Optional[str] = None
    ) -> None:
        """
        Visualisiert Entscheidungen der Agenten für ein Szenario.

        Args:
            scenario: Das ethische Szenario
            decisions: Dictionary mit Agent-Entscheidungen {agent_id: {decision, confidence, ...}}
            save_path: Pfad zum Speichern der Grafik
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            f"Entscheidungsanalyse: {scenario.scenario_id}", fontsize=16, fontweight="bold"
        )

        # Sammle Entscheidungsdaten
        options = list(scenario.options.keys())
        option_counts = {option: 0 for option in options}
        confidences = []
        agent_types = []
        decision_list = []

        for agent_id, decision_data in decisions.items():
            decision = decision_data.get("decision", "unknown")
            confidence = decision_data.get("confidence", 0.5)
            agent_type = decision_data.get("processing_type", "unknown")

            if decision in option_counts:
                option_counts[decision] += 1

            confidences.append(confidence)
            agent_types.append(agent_type)
            decision_list.append(decision)

        # 1. Entscheidungsverteilung (Pie Chart)
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]
        wedges, texts, autotexts = ax1.pie(
            option_counts.values(),
            labels=[opt.replace("_", " ").title() for opt in option_counts.keys()],
            autopct="%1.1f%%",
            colors=colors[: len(options)],
            startangle=90,
        )

        ax1.set_title("Entscheidungsverteilung", fontweight="bold")

        # 2. Konfidenz nach Verarbeitungstyp
        unique_types = list(set(agent_types))
        type_confidences: Dict[Any, List[float]] = {ptype: [] for ptype in unique_types}

        for i, agent_type in enumerate(agent_types):
            type_confidences[agent_type].append(confidences[i])

        # Box-Plot für Konfidenz
        box_data = []
        box_labels = []
        for ptype, conf_list in type_confidences.items():
            if conf_list:  # Nur wenn Daten vorhanden
                box_data.append(conf_list)
                box_labels.append(ptype.capitalize())

        if box_data:
            bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)

            # Farben für Box-Plot
            for i, box in enumerate(bp["boxes"]):
                ptype = box_labels[i].lower()
                color = self.processing_colors.get(ptype, "#95A5A6")
                box.set_facecolor(color)
                box.set_alpha(0.7)

        ax2.set_title("Konfidenz nach Verarbeitungstyp", fontweight="bold")
        ax2.set_ylabel("Konfidenz")
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)

        # 3. Szenario-Details
        ax3.axis("off")

        scenario_text = f"""
Szenario-Details:

Beschreibung:
{scenario.description[:200]}...

Eigenschaften:
• Komplexität: {scenario.complexity:.2f}
• Zeitdruck: {scenario.time_pressure:.2f}
• Unsicherheit: {scenario.uncertainty:.2f}
• Emotionale Valenz: {scenario.emotional_valence:.2f}

Verfügbare Optionen:
"""

        for option in options:
            scenario_text += f"• {option.replace('_', ' ').title()}\n"

        ax3.text(
            0.05,
            0.95,
            scenario_text,
            transform=ax3.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3),
        )

        # 4. Entscheidung vs. Verarbeitungstyp (Heatmap)
        # Erstelle Kontingenztabelle
        decision_type_matrix = np.zeros((len(unique_types), len(options)))

        for i, ptype in enumerate(unique_types):
            for j, option in enumerate(options):
                count = sum(
                    1
                    for k, (at, dec) in enumerate(zip(agent_types, decision_list))
                    if at == ptype and dec == option
                )
                decision_type_matrix[i, j] = count

        if len(unique_types) > 0 and len(options) > 0:
            im = ax4.imshow(decision_type_matrix, cmap="YlOrRd", aspect="auto")
            ax4.set_xticks(range(len(options)))
            ax4.set_xticklabels([opt.replace("_", " ").title() for opt in options], rotation=45)
            ax4.set_yticks(range(len(unique_types)))
            ax4.set_yticklabels([ptype.capitalize() for ptype in unique_types])
            ax4.set_title("Entscheidungen nach Verarbeitungstyp", fontweight="bold")

            # Annotationen hinzufügen
            for i in range(len(unique_types)):
                for j in range(len(options)):
                    text = ax4.text(
                        j,
                        i,
                        int(decision_type_matrix[i, j]),
                        ha="center",
                        va="center",
                        color="black",
                        fontweight="bold",
                    )

            plt.colorbar(im, ax=ax4, label="Anzahl Entscheidungen")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Entscheidungsplot gespeichert: {save_path}")

        # Nur anzeigen wenn interaktives Backend verfügbar
        safe_show_plot()

    def plot_belief_evolution(
        self, agent: Any, belief_history: Dict[str, List[float]], save_path: Optional[str] = None
    ) -> None:
        """
        Visualisiert die Evolution von Überzeugungen über Zeit.

        Args:
            agent: Der Agent
            belief_history: Dictionary mit Überzeugungsverläufen {belief_name: [values]}
            save_path: Pfad zum Speichern der Grafik
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle(f"Überzeugungsentwicklung: {agent.agent_id}", fontsize=16, fontweight="bold")

        # Zeitachse
        max_length = max(len(values) for values in belief_history.values()) if belief_history else 1
        time_steps = list(range(max_length))

        # 1. Überzeugungsstärken über Zeit
        for belief_name, values in belief_history.items():
            # Fülle kürzere Listen mit dem letzten Wert auf
            padded_values = (
                values + [values[-1]] * (max_length - len(values)) if values else [0] * max_length
            )
            ax1.plot(
                time_steps, padded_values, marker="o", linewidth=2, label=belief_name, alpha=0.8
            )

        ax1.set_xlabel("Zeitschritt")
        ax1.set_ylabel("Überzeugungsstärke")
        ax1.set_title("Entwicklung der Überzeugungsstärken", fontweight="bold")
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        # 2. Aktuelle Überzeugungsverteilung
        if hasattr(agent, "beliefs") and agent.beliefs:
            current_beliefs = {name: belief.strength for name, belief in agent.beliefs.items()}
            belief_names = list(current_beliefs.keys())
            belief_values = list(current_beliefs.values())

            bars = ax2.bar(
                range(len(belief_names)),
                belief_values,
                color=plt.cm.get_cmap("viridis")(np.linspace(0, 1, len(belief_names))),
            )

            ax2.set_xlabel("Überzeugungen")
            ax2.set_ylabel("Aktuelle Stärke")
            ax2.set_title("Aktuelle Überzeugungsverteilung", fontweight="bold")
            ax2.set_xticks(range(len(belief_names)))
            ax2.set_xticklabels(belief_names, rotation=45, ha="right")
            ax2.set_ylim(0, 1)
            ax2.grid(True, alpha=0.3)

            # Werte über den Balken anzeigen
            for bar, value in zip(bars, belief_values):
                height = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )
        else:
            ax2.text(
                0.5,
                0.5,
                "Keine Überzeugungsdaten verfügbar",
                ha="center",
                va="center",
                transform=ax2.transAxes,
                fontsize=14,
                style="italic",
            )
            ax2.set_title("Aktuelle Überzeugungsverteilung", fontweight="bold")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Überzeugungsplot gespeichert: {save_path}")

        safe_show_plot()

    def create_simulation_dashboard(
        self,
        society: Any,
        scenario: Any,
        decisions: Dict[str, Dict],
        save_path: Optional[str] = None,
    ) -> None:
        """
        Erstellt ein umfassendes Dashboard der Simulation.

        Args:
            society: Die simulierte Gesellschaft
            scenario: Das aktuelle Szenario
            decisions: Entscheidungsdaten
            save_path: Pfad zum Speichern der Grafik
        """
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        fig.suptitle("Ethische Agenten Simulation - Dashboard", fontsize=20, fontweight="bold")

        # 1. Netzwerk (oben links)
        ax1 = fig.add_subplot(gs[0, :2])
        G = society.social_network
        if G.number_of_nodes() > 0:
            pos = nx.spring_layout(G, k=1, iterations=30)
            node_colors = []
            for node in G.nodes():
                if node in society.agents:
                    processing_type = society.agents[node].cognitive_architecture.primary_processing
                    color = self.processing_colors.get(processing_type, "#95A5A6")
                    node_colors.append(color)
                else:
                    node_colors.append("#95A5A6")

            nx.draw(
                G,
                pos,
                ax=ax1,
                node_color=node_colors,
                node_size=500,
                with_labels=True,
                font_size=8,
                font_weight="bold",
            )

        ax1.set_title("Soziales Netzwerk", fontweight="bold")

        # 2. Entscheidungsverteilung (oben rechts)
        ax2 = fig.add_subplot(gs[0, 2:])
        options = list(scenario.options.keys())
        option_counts = {option: 0 for option in options}

        for decision_data in decisions.values():
            decision = decision_data.get("decision", "unknown")
            if decision in option_counts:
                option_counts[decision] += 1

        if sum(option_counts.values()) > 0:
            colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]
            ax2.pie(
                list(option_counts.values()),
                labels=[opt.replace("_", " ").title() for opt in option_counts.keys()],
                autopct="%1.1f%%",
                colors=colors[: len(options)],
            )

        ax2.set_title("Entscheidungsverteilung", fontweight="bold")

        # 3. Persönlichkeitsverteilung (unten links)
        ax3 = fig.add_subplot(gs[1, 0])
        personality_data: Dict[str, List[float]] = {
            trait: []
            for trait in [
                "openness",
                "conscientiousness",
                "extroversion",
                "agreeableness",
                "neuroticism",
            ]
        }

        for agent in society.agents.values():
            for trait, value in agent.personality_traits.items():
                if trait in personality_data:
                    personality_data[trait].append(value)

        traits = list(personality_data.keys())
        avg_values = [
            float(np.mean(personality_data[trait])) if personality_data[trait] else 0.0 for trait in traits
        ]

        x_pos = np.arange(len(traits))
        bars = ax3.bar(
            x_pos,
            avg_values,
            color=[self.personality_colors.get(trait, "#95A5A6") for trait in traits],
        )
        ax3.set_xlabel("Persönlichkeitsmerkmale")
        ax3.set_ylabel("Durchschnittswert")
        ax3.set_title("Durchschnittliche Persönlichkeitsverteilung", fontweight="bold")
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([trait.capitalize() for trait in traits], rotation=45)
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3)

        # 4. Verarbeitungstypen (Mitte rechts)
        ax4 = fig.add_subplot(gs[1, 2:])
        processing_distribution: Dict[Any, int] = {}
        for agent in society.agents.values():
            proc_type = agent.cognitive_architecture.primary_processing
            processing_distribution[proc_type] = processing_distribution.get(proc_type, 0) + 1

        if processing_distribution:
            proc_types = list(processing_distribution.keys())
            counts = list(processing_distribution.values())
            colors = [self.processing_colors.get(ptype, "#95A5A6") for ptype in proc_types]

            ax4.bar(range(len(proc_types)), counts, color=colors)
            ax4.set_xlabel("Verarbeitungstyp")
            ax4.set_ylabel("Anzahl Agenten")
            ax4.set_title("Verteilung der Verarbeitungstypen", fontweight="bold")
            ax4.set_xticks(range(len(proc_types)))
            ax4.set_xticklabels([ptype.capitalize() for ptype in proc_types], rotation=45)
            ax4.grid(True, alpha=0.3)

        # 5. Szenario-Info (unten)
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis("off")

        info_text = f"""
Szenario: {scenario.scenario_id}
Beschreibung: {scenario.description[:150]}...

Eigenschaften: Komplexität: {scenario.complexity:.2f} | Zeitdruck: {scenario.time_pressure:.2f} | Unsicherheit: {scenario.uncertainty:.2f} | Emotionale Valenz: {scenario.emotional_valence:.2f}

Agenten: {len(society.agents)} | Netzwerk-Dichte: {nx.density(society.social_network):.3f} | Entscheidungen: {len(decisions)}
        """

        ax5.text(
            0.5,
            0.5,
            info_text,
            transform=ax5.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.5),
        )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Dashboard gespeichert: {save_path}")

        safe_show_plot()

    def save_all_plots(self, society: Any, scenario: Any, decisions: Dict[str, Dict]) -> str:
        """
        Speichert alle verfügbaren Plots für eine Simulation.

        Args:
            society: Die simulierte Gesellschaft
            scenario: Das aktuelle Szenario
            decisions: Entscheidungsdaten

        Returns:
            Pfad zum Ausgabeverzeichnis
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(self.output_dir, f"simulation_{timestamp}")
        os.makedirs(session_dir, exist_ok=True)

        # Dashboard
        dashboard_path = os.path.join(session_dir, "dashboard.png")
        self.create_simulation_dashboard(society, scenario, decisions, dashboard_path)

        # Persönlichkeiten
        personality_path = os.path.join(session_dir, "personalities.png")
        self.plot_agent_personalities(list(society.agents.values()), personality_path)

        # Soziales Netzwerk
        network_path = os.path.join(session_dir, "social_network.png")
        self.plot_social_network(society, network_path)

        # Entscheidungsanalyse
        decisions_path = os.path.join(session_dir, "decisions.png")
        self.plot_scenario_decisions(scenario, decisions, decisions_path)

        print(f"\n🎨 Alle Visualisierungen gespeichert in: {session_dir}")
        return session_dir


# Hilfsfunktionen für einfache Verwendung
def quick_plot_agents(agents: List[Any]) -> None:
    """Schnelle Visualisierung von Agent-Persönlichkeiten."""
    viz = EthicalSimulationVisualizer()
    viz.plot_agent_personalities(agents)


def quick_plot_decisions(scenario: Any, decisions: Dict[str, Dict]) -> None:
    """Schnelle Visualisierung von Entscheidungen."""
    viz = EthicalSimulationVisualizer()
    viz.plot_scenario_decisions(scenario, decisions)


def quick_dashboard(society: Any, scenario: Any, decisions: Dict[str, Dict]) -> None:
    """Schnelles Dashboard für eine Simulation."""
    viz = EthicalSimulationVisualizer()
    viz.create_simulation_dashboard(society, scenario, decisions)


if __name__ == "__main__":
    print("🎨 Ethische Agenten Visualisierung")
    print("=" * 50)
    print("Verfügbare Visualisierungen:")
    print("- Agent-Persönlichkeiten")
    print("- Soziales Netzwerk")
    print("- Entscheidungsanalyse")
    print("- Überzeugungsentwicklung")
    print("- Simulation Dashboard")
    print("\nVerwendung: Importieren Sie die Klasse EthicalSimulationVisualizer")
    print("oder nutzen Sie die quick_* Funktionen für einfache Plots.")
