import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from .neural_core import NeuralProcessingType
from .neural_agent import NeuralEthicalAgent # For type hints if used explicitly
from matplotlib.lines import Line2D
from typing import TYPE_CHECKING, List, Dict, Tuple, Set, Union, Callable, Optional

if TYPE_CHECKING:
    # This will be NeuralEthicalSociety from ethik.neural_society eventually
    # For now, using the main script's class name as a string hint.
    # This is a forward declaration and won't cause a runtime circular import.
    from ..Ethik_Simulation_mit_neurokognitivenErweiterungen import NeuralEthicalSociety


def visualize_neural_processing(society: 'NeuralEthicalSociety', agent_id: str):
    """Visualisiert die neuronale Verarbeitung eines Agenten."""
    if agent_id not in society.agents:
        print(f"Agent {agent_id} nicht gefunden.")
        return
        
    agent = society.agents[agent_id]
    
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
    # plt.show() # Commented out as per plan

def visualize_belief_network(society: 'NeuralEthicalSociety', agent_id: str, min_connection_strength: float = 0.2, 
                           show_activation: bool = False):
    """
    Visualisiert das Netzwerk von Überzeugungen eines Agenten.
    
    Args:
        society: The NeuralEthicalSociety instance.
        agent_id: ID des zu visualisierenden Agenten
        min_connection_strength: Minimale Verbindungsstärke für die Anzeige
        show_activation: Ob Aktivierungsniveaus angezeigt werden sollen
    """
    if agent_id not in society.agents:
        print(f"Agent {agent_id} nicht gefunden.")
        return
        
    agent = society.agents[agent_id]
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
    categories_set = set(nx.get_node_attributes(G, 'category').values()) # Renamed categories to categories_set
    # Ensure there's a fallback if no categories exist (e.g. empty graph)
    if not categories_set:
        # Handle empty or category-less graph if necessary, or let it pass
        pass

    colors_list = plt.cm.tab10(np.linspace(0, 1, len(categories_set))) if categories_set else [] # Renamed colors to colors_list
    category_colors_map = dict(zip(categories_set, colors_list)) # Renamed category_colors to category_colors_map
    
    for category, color_val in category_colors_map.items(): # Renamed color to color_val
        node_list = [node for node, data in G.nodes(data=True) if data.get('category') == category]
        
        node_colors_viz: List[Any] # Declare type for node_colors_viz
        if show_activation:
            # Größe basierend auf Aktivierung
            node_sizes = [300 + 700 * G.nodes[node].get('activation', 0) for node in node_list]
            
            # Farbtransparenz basierend auf Gewissheit
            alphas = [0.3 + 0.7 * G.nodes[node].get('certainty', 0.5) for node in node_list]
            node_colors_viz = [(*color_val[:3], alpha) for alpha in alphas] 
        else:
            # Größe basierend auf Stärke
            node_sizes = [300 + 700 * G.nodes[node].get('strength', 0) for node in node_list]
            node_colors_viz = [color_val] * len(node_list)
        
        nx.draw_networkx_nodes(
            G, pos, 
            nodelist=node_list, 
            node_color=node_colors_viz, 
            node_size=node_sizes,
            alpha=0.8,
            label=category
        )
    
    # Kanten zeichnen, rot für negative, grün für positive
    pos_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('polarity', 0) > 0]
    neg_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('polarity', 0) < 0]
    
    edge_weights_pos = [G[u][v].get('weight', 0.1) * 2 for u,v in pos_edges]
    nx.draw_networkx_edges(G, pos, edgelist=pos_edges, width=edge_weights_pos, 
                          edge_color='green', alpha=0.6, arrows=True)
    
    edge_weights_neg = [G[u][v].get('weight', 0.1) * 2 for u,v in neg_edges]
    nx.draw_networkx_edges(G, pos, edgelist=neg_edges, width=edge_weights_neg, 
                          edge_color='red', alpha=0.6, arrows=True, style='dashed')
    
    # Knotenbeschriftungen
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    
    title = f"Überzeugungsnetzwerk für Agent {agent_id}"
    if show_activation:
        title += " (mit Aktivierungsniveaus)"
        
    plt.title(title)
    if categories_set: # Only add legend if there are categories
        plt.legend()
    plt.axis('off')
    plt.tight_layout()
    # plt.show() # Commented out

def visualize_cognitive_style_comparison(society: 'NeuralEthicalSociety'):
    """Visualisiert einen Vergleich der verschiedenen kognitiven Verarbeitungsstile."""
    # Agenten nach kognitivem Stil gruppieren
    style_groups: Dict[str, List[str]] = {
        NeuralProcessingType.SYSTEMATIC: [],
        NeuralProcessingType.INTUITIVE: [],
        NeuralProcessingType.ASSOCIATIVE: [],
        NeuralProcessingType.EMOTIONAL: [],
        NeuralProcessingType.ANALOGICAL: [],
        NeuralProcessingType.NARRATIVE: []
    }
    
    # Nur Stile mit Agenten berücksichtigen
    for agent_id, agent_obj in society.agents.items(): 
        style = agent_obj.cognitive_architecture.primary_processing
        if style in style_groups: # Ensure style is valid
            style_groups[style].append(agent_id)
            
    active_styles = {s: agents for s, agents in style_groups.items() if agents}
    
    if not active_styles:
        print("Keine Agenten mit definierten kognitiven Stilen gefunden.")
        return
            
    # Häufige Überzeugungen identifizieren (für den Vergleich)
    all_beliefs_counts: Dict[str, int] = {} 
    for agent_obj in society.agents.values(): 
        for belief_name in agent_obj.beliefs:
            if belief_name not in all_beliefs_counts:
                all_beliefs_counts[belief_name] = 0
            all_beliefs_counts[belief_name] += 1
    
    # Top 5 häufigste Überzeugungen auswählen
    common_beliefs = sorted(all_beliefs_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    common_belief_names = [b[0] for b in common_beliefs]
    
    if not common_belief_names: # Handle case with no common beliefs
        print("Keine gemeinsamen Überzeugungen für den Vergleich gefunden.")
        return

    # Durchschnittliche Überzeugungsstärken pro Stil und Überzeugung
    style_belief_strengths: Dict[str, Dict[str, float]] = {}
    
    for style, agent_ids in active_styles.items():
        style_belief_strengths[style] = {}
        
        for belief_name in common_belief_names:
            strengths = []
            
            for agent_id in agent_ids:
                agent_obj = society.agents[agent_id] 
                if belief_name in agent_obj.beliefs:
                    strengths.append(agent_obj.beliefs[belief_name].strength)
            
            if strengths:
                style_belief_strengths[style][belief_name] = np.mean(strengths)
            else:
                style_belief_strengths[style][belief_name] = 0.0 
    
    # Visualisierung
    plt.figure(figsize=(12, 8))
    
    # Gruppierte Balkendiagramme erstellen
    x = np.arange(len(common_belief_names))
    width = 0.15  # Breite der Balken
    
    # Für jeden Stil einen Balken pro Überzeugung
    num_styles = len(style_belief_strengths)
    for i, (style, beliefs_map) in enumerate(style_belief_strengths.items()): 
        strengths = [beliefs_map.get(b, 0.0) for b in common_belief_names] 
        offset = width * (i - num_styles / 2 + 0.5)
        plt.bar(x + offset, strengths, width, label=str(style))
    
    plt.xlabel('Überzeugungen')
    plt.ylabel('Durchschnittliche Stärke')
    plt.title('Überzeugungsstärke nach kognitivem Verarbeitungsstil')
    plt.xticks(x, common_belief_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    # plt.show() # Commented out

def visualize_social_network(society: 'NeuralEthicalSociety', color_by: str = 'cognitive_style'):
    """
    Visualisiert das soziale Netzwerk der Agenten.
    
    Args:
        society: The NeuralEthicalSociety instance.
        color_by: Bestimmt, wie die Knoten gefärbt werden 
                ('group', 'belief', oder 'cognitive_style')
    """
    if not society.social_network or not society.social_network.nodes(): # Check if social network exists and has nodes
        print("Soziales Netzwerk ist leer oder nicht vorhanden.")
        return

    plt.figure(figsize=(12, 10))
    
    # Positionen berechnen
    pos = nx.spring_layout(society.social_network, seed=42)
    
    node_colors_val: Union[List[float], List[str], List[Tuple[float,float,float,float]]] # Type hint for node_colors_val
    legend_elements: List = [] # Initialize legend_elements
    color_map_plt = None # Initialize color_map_plt

    if color_by == 'group':
        colors_val_group: List[int] = [] # Renamed colors_val to colors_val_group
        for agent_id in society.social_network.nodes():
            agent = society.agents[agent_id]
            primary_group: Optional[str] = None
            max_id_val: float = 0.0
            for group_name, id_strength in agent.group_identities.items(): # Renamed group to group_name
                if id_strength > max_id_val:
                    max_id_val = id_strength
                    primary_group = group_name
            colors_val_group.append(hash(primary_group) % 10 if primary_group else 0)
        
        color_map_plt = plt.cm.tab10
        node_colors_val = [color_map_plt(c/10.0) for c in colors_val_group]
    
    elif color_by == 'belief':
        all_beliefs_set: Set[str] = set()
        for agent_obj in society.agents.values():
            all_beliefs_set.update(agent_obj.beliefs.keys())
            
        if all_beliefs_set:
            rep_belief = list(all_beliefs_set)[0]
            colors_val_belief: List[float] = []
            for agent_id in society.social_network.nodes():
                agent = society.agents[agent_id]
                if rep_belief in agent.beliefs:
                    colors_val_belief.append(agent.beliefs[rep_belief].strength)
                else:
                    colors_val_belief.append(0.0)
            color_map_plt = plt.cm.viridis
            node_colors_val = colors_val_belief
        else:
            color_map_plt = plt.cm.viridis 
            node_colors_val = [0.5] * len(society.social_network.nodes())
            
    elif color_by == 'cognitive_style':
        style_colors_map: Dict[str, str] = { 
            NeuralProcessingType.SYSTEMATIC: 'blue',
            NeuralProcessingType.INTUITIVE: 'red',
            NeuralProcessingType.ASSOCIATIVE: 'green',
            NeuralProcessingType.EMOTIONAL: 'purple',
            NeuralProcessingType.ANALOGICAL: 'orange',
            NeuralProcessingType.NARRATIVE: 'brown'
        }
        node_colors_list_style: List[str] = []
        for agent_id in society.social_network.nodes():
            agent = society.agents[agent_id]
            style = agent.cognitive_architecture.primary_processing
            node_colors_list_style.append(style_colors_map.get(style, 'gray'))
        node_colors_val = node_colors_list_style
        
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color_val, label=str(style_key), markersize=10) # Renamed style to style_key
            for style_key, color_val in style_colors_map.items()
            if any(society.agents[agent_id_check].cognitive_architecture.primary_processing == style_key # Renamed agent_id to agent_id_check
                  for agent_id_check in society.social_network.nodes())
        ]
    
    else:
        color_map_plt = plt.cm.viridis
        node_colors_val = [0.5] * len(society.social_network.nodes())
        
    node_size = [300 * (1 + society.social_network.degree(node)) for node in society.social_network.nodes()]
    edge_widths = [2 * society.social_network[u][v].get('weight', 0.1) for u, v in society.social_network.edges()]
    
    nodes_drawn = None
    if color_by in ['group', 'belief'] and color_map_plt is not None:
        nodes_drawn = nx.draw_networkx_nodes(
            society.social_network, pos, 
            node_size=node_size,
            node_color=node_colors_val, 
            cmap=color_map_plt, 
            alpha=0.8
        )
    else: 
        nodes_drawn = nx.draw_networkx_nodes(
            society.social_network, pos, 
            node_size=node_size,
            node_color=node_colors_val, 
            alpha=0.8
        )
    
    nx.draw_networkx_edges( 
        society.social_network, pos,
        width=edge_widths,
        alpha=0.5
    )
    
    nx.draw_networkx_labels(
        society.social_network, pos, 
        font_size=8, 
        font_family='sans-serif'
    )
    
    plt.title("Soziales Netzwerk der Agenten")
    plt.axis('off')
    
    if color_by in ['group', 'belief'] and nodes_drawn is not None and color_map_plt is not None:
        plt.colorbar(nodes_drawn)
    elif legend_elements:
        plt.legend(handles=legend_elements)
            
    plt.tight_layout()
    # plt.show() # Commented out
