import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional

# Importiere notwendige Klassen und Konstanten aus dem ethik Paket
from .neural_types import NeuralProcessingType
from .analyzer import SimulationAnalyzer # Abhängigkeit vom Analyzer für Daten
# NeuralEthicalSociety wird für Typ-Hinweise benötigt
from .neural_society import NeuralEthicalSociety
from .constants import (VIS_SOCIAL_NODE_DEGREE_SCALING, VIS_SOCIAL_EDGE_WEIGHT_SCALING,
                        VIS_STYLE_COMPARISON_BAR_WIDTH)


class SimulationVisualizer:
    """Klasse zur Visualisierung der Simulationsergebnisse."""

    def __init__(self, society: 'NeuralEthicalSociety', analyzer: SimulationAnalyzer):
        self.society = society
        self.analyzer = analyzer
        self.results = analyzer.results

    def plot_belief_evolution(self, belief_name: str, agent_ids: Optional[List[str]] = None,
                              show_mean: bool = True, show_styles: bool = True):
        """Plottet die Entwicklung der Stärke eines Beliefs über die Zeit."""
        belief_df = self.analyzer.get_belief_evolution(belief_name)
        if belief_df is None or belief_df.empty:
            print(f"Keine Daten für Belief '{belief_name}' gefunden.")
            return

        plt.figure(figsize=(12, 7))
        title = f"Entwicklung von Belief '{belief_name}'"

        if agent_ids:
            belief_df = belief_df[belief_df['agent_id'].isin(agent_ids)]
            title += f" (Agenten: {', '.join(agent_ids)})"

        unique_agents = belief_df['agent_id'].unique()

        if show_styles:
            styles = sorted(belief_df['primary_style'].unique())
            colors = plt.cm.get_cmap('tab10', len(styles)) # type: ignore
            style_colors = {style: colors(i) for i, style in enumerate(styles)}

            for agent_id_val in unique_agents:
                 agent_data = belief_df[belief_df['agent_id'] == agent_id_val]
                 if not agent_data.empty:
                     style = agent_data['primary_style'].iloc[0]
                     plt.plot(agent_data['step'], agent_data['strength'], marker='.', linestyle='-',
                              linewidth=0.5, alpha=0.6, color=style_colors.get(style, 'gray'),
                              label=f"{agent_id_val}" if (agent_ids and len(agent_ids) < 10) else None)

            handles = [plt.Line2D([0], [0], color=color, lw=2, label=str(style)) for style, color in style_colors.items()]
            legend_title = "Cognitive Style"
            if show_mean and not (agent_ids and len(agent_ids) < 10) : # Wenn keine individuellen Agenten gelabelt sind
                 pass # Mean wird separat gelabelt
            elif agent_ids and len(agent_ids) < 10: # Wenn individuelle Agenten gelabelt
                 legend_title = "Agent ID / Style" # Kombinierte Legende

            # plt.legend(handles=handles, title=legend_title, bbox_to_anchor=(1.05, 1), loc='upper left')

        else:
             for agent_id_val in unique_agents:
                 agent_data = belief_df[belief_df['agent_id'] == agent_id_val]
                 plt.plot(agent_data['step'], agent_data['strength'], marker='.', linestyle='-',
                          linewidth=0.5, alpha=0.7,
                          label=agent_id_val if len(unique_agents) < 10 else None)
            # if len(unique_agents) < 10 : plt.legend(title="Agent ID", bbox_to_anchor=(1.05, 1), loc='upper left')


        current_handles, current_labels = plt.gca().get_legend_handles_labels()
        if show_styles:
            style_legend = plt.legend(handles=handles, title="Cognitive Style", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
            plt.gca().add_artist(style_legend)


        if show_mean:
            mean_strength = belief_df.groupby('step')['strength'].mean()
            mean_line, = plt.plot(mean_strength.index, mean_strength.values, color='black', linewidth=2, linestyle='--', label='Mean Strength')
            current_handles.append(mean_line)
            current_labels.append('Mean Strength')

        if current_handles: # Nur Legende anzeigen, wenn es etwas zu labeln gibt
            if show_styles and show_mean: # Wenn beides da ist, Mean separat
                 plt.legend(handles=[mean_line], labels=['Mean Strength'], title="", bbox_to_anchor=(1.05, 0), loc='lower left', frameon=False)
            elif not show_styles and show_mean and len(unique_agents) >=10: # Nur Mean, viele Agenten
                 plt.legend(handles=[mean_line], labels=['Mean Strength'], title="", bbox_to_anchor=(1.05, 1), loc='upper left')
            elif not show_styles and len(unique_agents) < 10 : # Individuelle Agenten oder Mean
                 plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Agent ID / Mean")
            elif not show_styles and not show_mean:
                pass # Keine Legende
            # Fall für nur Styles wurde oben behandelt

        plt.title(title)
        plt.xlabel("Simulationsschritt")
        plt.ylabel("Überzeugungsstärke")
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.tight_layout(rect=[0, 0, 0.80, 1])
        plt.show()


    def plot_polarization_trend(self, belief_name: Optional[str] = None, metric: str = 'bimodality'):
        """Plottet die Entwicklung der Polarisierung (z.B. Bimodalität) über die Zeit."""
        pol_df = self.analyzer.get_polarization_trend(metric=metric)
        if pol_df is None or pol_df.empty:
            print(f"Keine Polarisierungsdaten für Metrik '{metric}' gefunden.")
            return

        plt.figure(figsize=(12, 7))
        title_text = f"Polarisierungstrend ({metric})"

        if belief_name:
             belief_data = pol_df[pol_df['belief'] == belief_name]
             if belief_data.empty:
                  print(f"Keine Daten für Belief '{belief_name}' und Metrik '{metric}'.")
                  return
             plt.plot(belief_data['step'], belief_data['value'], marker='o', linestyle='-', label=belief_name)
             title_text = f"Polarisierungstrend für '{belief_name}' ({metric})"
             plt.legend()
        else:
             mean_pol = pol_df.groupby('step')['value'].mean()
             std_pol = pol_df.groupby('step')['value'].std().fillna(0) # NaN füllen für fill_between
             plt.plot(mean_pol.index, mean_pol.values, marker='o', linestyle='-', color='blue', label=f'Mean {metric}')
             plt.fill_between(mean_pol.index, mean_pol - std_pol, mean_pol + std_pol, color='blue', alpha=0.2, label=f'Std Dev {metric}')
             title_text = f"Durchschnittlicher Polarisierungstrend ({metric}) über alle Beliefs"
             plt.legend()

        plt.title(title_text)
        plt.xlabel("Simulationsschritt")
        plt.ylabel(f"{metric.capitalize()}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_decision_distribution(self, scenario_id: str):
        """Zeigt die Verteilung der Entscheidungen für ein Szenario."""
        counts = self.analyzer.get_decision_summary(scenario_id=scenario_id)
        if not counts or scenario_id not in counts:
            print(f"Keine Entscheidungsdaten für Szenario '{scenario_id}' gefunden.")
            return
        scenario_counts = counts[scenario_id]
        options = list(scenario_counts.keys())
        values = list(scenario_counts.values())

        plt.figure(figsize=(10, 6)) # Etwas breiter für Lesbarkeit der Ticks
        bars = plt.bar(options, values, color=plt.cm.get_cmap('viridis', len(options))(np.linspace(0, 1, len(options)))) # type: ignore
        plt.title(f"Entscheidungsverteilung für Szenario '{scenario_id}'")
        plt.xlabel("Gewählte Option")
        plt.ylabel("Anzahl Gesamtentscheidungen")
        plt.xticks(rotation=45, ha='right')
        # Füge Werte über den Balken hinzu
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05 * max(values), # Offset für Text
                     int(yval), ha='center', va='bottom')
        plt.tight_layout()
        plt.show()

    def plot_cognitive_style_decision_comparison(self, scenario_id: str):
        """Vergleicht Entscheidungsmuster verschiedener kognitiver Stile für ein Szenario."""
        style_counts = self.analyzer.get_cognitive_style_decision_summary()
        if not style_counts:
            print("Keine Daten für kognitive Stil-Entscheidungen gefunden.")
            return

        scenario_style_data = {style: counts.get(scenario_id, {})
                               for style, counts in style_counts.items()}
        # Filter leere Dictionaries, falls ein Stil keine Entscheidungen in diesem Szenario hatte
        scenario_style_data = {s: c for s, c in scenario_style_data.items() if c}


        if not scenario_style_data:
            print(f"Keine Daten für Szenario '{scenario_id}' nach kognitivem Stil gefunden.")
            return

        all_options = set()
        for options_dict in scenario_style_data.values():
            all_options.update(options_dict.keys())
        if not all_options:
            print(f"Keine Optionen für Szenario '{scenario_id}' in den Stil-Daten gefunden.")
            return
        sorted_options = sorted(list(all_options))

        plot_data = []
        for style, options_dict in scenario_style_data.items():
             total_decisions_by_style = sum(options_dict.values())
             for option in sorted_options:
                  count = options_dict.get(option, 0)
                  proportion = count / total_decisions_by_style if total_decisions_by_style > 0 else 0
                  plot_data.append({"style": str(style), "option": option, "proportion": proportion, "count": count})

        if not plot_data:
            print(f"Keine Plot-Daten für Szenario '{scenario_id}' nach Stil generiert.")
            return

        df = pd.DataFrame(plot_data)

        plt.figure(figsize=(14, 8)) # Größer für bessere Lesbarkeit
        sns.barplot(x='option', y='proportion', hue='style', data=df, palette='tab10') # type: ignore
        plt.title(f"Entscheidungsanteile nach Kognitivem Stil für Szenario '{scenario_id}'")
        plt.xlabel("Gewählte Option")
        plt.ylabel("Anteil der Entscheidungen pro Stil")
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Cognitive Style', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.show()

    def plot_ensemble_variance(self, belief_name: str):
        """Plottet die Varianz der Belief-Stärke über die Ensemble-Läufe."""
        var_df = self.analyzer.get_ensemble_variance("belief_strength_variance")
        if var_df is None or var_df.empty:
            print("Keine Ensemble-Varianzdaten gefunden.")
            return

        belief_var_df = var_df[var_df['belief'] == belief_name]
        if belief_var_df.empty:
            print(f"Keine Varianzdaten für Belief '{belief_name}'.")
            return

        plt.figure(figsize=(12, 7))
        mean_variance = belief_var_df.groupby('step')['variance'].mean()
        plt.plot(mean_variance.index, mean_variance.values, marker='.', linestyle='-', label=f'Mean Ensemble Variance for {belief_name}')

        plt.title(f"Ensemble-Varianz für Belief '{belief_name}'")
        plt.xlabel("Simulationsschritt")
        plt.ylabel("Varianz der Stärke über Ensemble-Läufe")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def visualize_social_network(self, color_by: str = 'cognitive_style', show_clusters: bool = False, step: int = -1):
        """Visualisiert das soziale Netzwerk, optional mit Cluster-Markierung zu einem bestimmten Schritt."""
        G = self.society.social_network # Das ist das initiale Netzwerk
        if G.number_of_nodes() == 0:
            print("Soziales Netzwerk ist leer.")
            return

        plt.figure(figsize=(16, 14)) # Größer
        pos = nx.spring_layout(G, seed=42, k=0.5, iterations=50)

        node_colors_list = [] # Wird für die Farbgebung verwendet
        legend_elements = []

        # Agenten-Instanzen aus dem Graphen oder der Society holen
        # Es ist besser, die Agenten-Instanzen direkt aus der Society zu nehmen,
        # da diese den aktuellen Zustand (z.B. für Belief-basierte Färbung) halten könnten.
        # Für kognitiven Stil und Gruppe ist der Society-Zustand (initial) ausreichend.

        agent_instances_map = {agent_id: self.society.agents[agent_id] for agent_id in G.nodes() if agent_id in self.society.agents}


        if color_by == 'cognitive_style':
            style_colors_map = plt.cm.get_cmap('tab10', len(NeuralProcessingType.ALL_TYPES)) # type: ignore
            styles = sorted(list(NeuralProcessingType.ALL_TYPES))
            style_to_color = {style: style_colors_map(i) for i, style in enumerate(styles)}

            for node_id in G.nodes():
                agent = agent_instances_map.get(node_id)
                if agent:
                    node_colors_list.append(style_to_color.get(agent.cognitive_architecture.primary_processing, 'lightgrey'))
                else:
                    node_colors_list.append('lightgrey') # Fallback
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=str(style), markersize=10, markerfacecolor=color)
                               for style, color in style_to_color.items()]
        elif color_by == 'group':
            all_groups_in_society = sorted(list(self.society.groups.keys()))
            if not all_groups_in_society:
                print("Keine Gruppen in der Gesellschaft definiert für die Färbung.")
                node_colors_list = ['lightgrey'] * G.number_of_nodes()
            else:
                group_colors_map = plt.cm.get_cmap('Accent', len(all_groups_in_society)) # type: ignore
                group_to_color = {group: group_colors_map(i) for i, group in enumerate(all_groups_in_society)}

                for node_id in G.nodes():
                    agent = agent_instances_map.get(node_id)
                    if agent and agent.group_identities:
                        dominant_group = max(agent.group_identities, key=agent.group_identities.get, default=None)
                        node_colors_list.append(group_to_color.get(dominant_group, 'lightgrey') if dominant_group else 'lightgrey')
                    else:
                        node_colors_list.append('lightgrey')
                legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=group, markersize=10, markerfacecolor=color)
                                   for group, color in group_to_color.items()]
        else:
            node_colors_list = ['skyblue'] * G.number_of_nodes()


        node_shapes_map = {}
        default_marker_shape = 'o'
        if show_clusters:
            clusters_at_step = self.analyzer.identify_belief_clusters(step=step)
            available_markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X']
            for i, cluster_info in enumerate(clusters_at_step):
                marker_for_cluster = available_markers[i % len(available_markers)]
                for agent_id_in_cluster in cluster_info['agents']:
                    node_shapes_map[agent_id_in_cluster] = marker_for_cluster
            # Legende für Cluster-Marker (falls gewünscht)
            # cluster_legend_elements = [plt.Line2D([0], [0], marker=mk, color='w', label=f'Cluster {i+1}', markersize=10, markerfacecolor='black')
            # for i, mk in enumerate(set(node_shapes_map.values()))]
            # if legend_elements: legend_elements.extend(cluster_legend_elements)
            # else: legend_elements = cluster_legend_elements


        # Knoten zeichnen, iteriere über eindeutige Shapes für separate draw_networkx_nodes Aufrufe
        unique_shapes_present = sorted(list(set(node_shapes_map.values()) | {default_marker_shape}))

        for shape_val in unique_shapes_present:
            nodes_for_this_shape = [node for node in G.nodes() if node_shapes_map.get(node, default_marker_shape) == shape_val]
            if not nodes_for_this_shape: continue

            colors_for_this_shape_batch = []
            for node_id_in_batch in nodes_for_this_shape:
                try:
                    original_node_index = list(G.nodes()).index(node_id_in_batch)
                    colors_for_this_shape_batch.append(node_colors_list[original_node_index])
                except ValueError: #Sollte nicht passieren, wenn node_colors_list korrekt initialisiert wurde
                    colors_for_this_shape_batch.append('grey')


            nx.draw_networkx_nodes(G, pos, nodelist=nodes_for_this_shape,
                                   node_size=VIS_SOCIAL_NODE_DEGREE_SCALING, # Ggf. anpassen oder dynamisch machen
                                   node_color=colors_for_this_shape_batch, alpha=0.8, node_shape=shape_val)

        edge_widths_list = [VIS_SOCIAL_EDGE_WEIGHT_SCALING * G[u][v]['weight'] for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=edge_widths_list, alpha=0.3, edge_color='grey')
        nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')

        title_str = f"Soziales Netzwerk (Gefärbt nach: {color_by})"
        if show_clusters: title_str += f" (Cluster am {'finalen' if step == -1 else 'Step '+str(step)} markiert)"
        plt.title(title_str, fontsize=16)
        plt.axis('off')
        if legend_elements:
            plt.legend(handles=legend_elements, title=color_by.replace('_', ' ').title(), bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.show()
