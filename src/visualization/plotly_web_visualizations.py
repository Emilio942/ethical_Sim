#!/usr/bin/env python3
"""
Erweiterte Web-Visualisierungen mit Plotly
=========================================

Diese Erweiterung fügt interaktive Plotly-Visualisierungen 
zum Web-Interface hinzu für bessere Benutzererfahrung.

Features:
- 3D-Netzwerk-Visualisierungen
- Interaktive Zeitreihen-Plots
- Dashboard mit Live-Updates
- Export-Funktionen für Plots
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
import networkx as nx
import numpy as np
import pandas as pd
from datetime import datetime
import json

class PlotlyWebVisualizations:
    """Erweiterte Plotly-Visualisierungen für Web-Interface"""
    
    def __init__(self, society):
        self.society = society
        self.colors = px.colors.qualitative.Set1
        
    def create_interactive_network_3d(self):
        """Erstellt interaktives 3D-Netzwerk-Diagramm"""
        G = self.society.social_network
        
        # 3D Layout berechnen
        pos = nx.spring_layout(G, dim=3, k=1, iterations=50)
        
        # Knoten-Positionen extrahieren
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_z = [pos[node][2] for node in G.nodes()]
        
        # Kanten-Linien für 3D
        edge_x = []
        edge_y = []
        edge_z = []
        
        for edge in G.edges():
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])
        
        # Kanten-Trace
        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            line=dict(width=2, color='rgba(50,50,50,0.5)'),
            hoverinfo='none',
            mode='lines',
            name='Connections'
        )
        
        # Knoten-Informationen
        node_text = []
        node_colors = []
        for i, node in enumerate(G.nodes()):
            agent = self.society.agents[i] if i < len(self.society.agents) else None
            if agent:
                personality = getattr(agent, 'personality', 'unknown')
                decisions = len(getattr(agent, 'decision_history', []))
                node_text.append(f"Agent: {node}<br>Personality: {personality}<br>Decisions: {decisions}")
                
                # Farbe basierend auf Persönlichkeit
                color_map = {
                    'utilitarian': '#FF6B6B',
                    'deontological': '#4ECDC4', 
                    'virtue_ethics': '#45B7D1',
                    'balanced': '#96CEB4',
                    'pragmatic': '#FFEAA7',
                    'idealistic': '#DDA0DD'
                }
                node_colors.append(color_map.get(personality, '#95A5A6'))
            else:
                node_text.append(f"Agent: {node}")
                node_colors.append('#95A5A6')
        
        # Knoten-Trace
        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                size=10,
                color=node_colors,
                line=dict(width=2, color='rgba(50,50,50,0.5)')
            ),
            name='Agents'
        )
        
        # Layout
        layout = go.Layout(
            title='3D Agent Network Visualization',
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            scene=dict(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                bgcolor='rgba(0,0,0,0)'
            )
        )
        
        fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
        return fig.to_html(include_plotlyjs='cdn', div_id='network3d-plot')
    
    def create_belief_evolution_timeline(self):
        """Erstellt interaktive Zeitreihen-Visualisierung der Belief-Evolution"""
        if not self.society.agents:
            return "<p>No agents available for visualization</p>"
        
        # Sammle Belief-Daten über Zeit
        timeline_data = []
        
        for agent in self.society.agents:
            if hasattr(agent, 'beliefs') and hasattr(agent.beliefs, 'beliefs'):
                beliefs = agent.beliefs.beliefs
                for belief_name, value in beliefs.items():
                    timeline_data.append({
                        'agent': agent.agent_id,
                        'belief': belief_name,
                        'value': value,
                        'timestamp': datetime.now(),
                        'personality': getattr(agent, 'personality', 'unknown')
                    })
        
        if not timeline_data:
            return "<p>No belief data available</p>"
        
        df = pd.DataFrame(timeline_data)
        
        # Erstelle interaktives Zeitreihen-Diagramm
        fig = px.line(
            df, 
            x='timestamp', 
            y='value',
            color='agent',
            facet_col='belief',
            title='Belief Evolution Over Time',
            hover_data=['personality']
        )
        
        fig.update_layout(
            height=400,
            showlegend=True,
            title_font_size=16
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id='belief-timeline-plot')
    
    def create_decision_heatmap(self):
        """Erstellt interaktive Heatmap der Entscheidungsmuster"""
        if not self.society.agents:
            return "<p>No agents available for visualization</p>"
        
        # Sammle Entscheidungsdaten
        decision_matrix = []
        agent_names = []
        scenario_types = set()
        
        for agent in self.society.agents:
            agent_names.append(agent.agent_id)
            agent_decisions = {}
            
            if hasattr(agent, 'decision_history'):
                for decision in agent.decision_history:
                    scenario_type = getattr(decision, 'scenario_type', 'unknown')
                    confidence = getattr(decision, 'confidence', 0.5)
                    scenario_types.add(scenario_type)
                    agent_decisions[scenario_type] = confidence
            
            decision_matrix.append(agent_decisions)
        
        # Konvertiere zu DataFrame
        scenario_types = list(scenario_types)
        matrix_data = []
        
        for i, agent_decisions in enumerate(decision_matrix):
            row = [agent_decisions.get(scenario, 0) for scenario in scenario_types]
            matrix_data.append(row)
        
        if not matrix_data or not scenario_types:
            return "<p>No decision data available</p>"
        
        # Erstelle Heatmap
        fig = go.Figure(data=go.Heatmap(
            z=matrix_data,
            x=scenario_types,
            y=agent_names,
            colorscale='Viridis',
            hoverongaps=False,
            hovertemplate='Agent: %{y}<br>Scenario: %{x}<br>Confidence: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Decision Confidence Heatmap',
            xaxis_title='Scenario Types',
            yaxis_title='Agents',
            height=400
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id='decision-heatmap-plot')
    
    def create_metrics_dashboard(self):
        """Erstellt interaktives Metriken-Dashboard"""
        from metrics import MetricsCollector
        
        try:
            collector = MetricsCollector()
            metrics = collector.collect_all_metrics(self.society)
            
            # Extrahiere gesellschaftliche Metriken
            societal = metrics.get('societal_metrics', {})
            
            # Erstelle Gauge-Charts für Hauptmetriken
            fig = make_subplots(
                rows=2, cols=2,
                specs=[[{"type": "indicator"}, {"type": "indicator"}],
                       [{"type": "indicator"}, {"type": "indicator"}]],
                subplot_titles=('Polarization', 'Consensus', 'Network Cohesion', 'Influence Distribution')
            )
            
            # Polarisierung
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=societal.get('polarization', 0),
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Polarization"},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 0.3], 'color': "lightgray"},
                        {'range': [0.3, 0.7], 'color': "gray"},
                        {'range': [0.7, 1], 'color': "darkgray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.8
                    }
                }
            ), row=1, col=1)
            
            # Konsens
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=societal.get('consensus', 0),
                title={'text': "Consensus"},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 0.3], 'color': "lightgray"},
                        {'range': [0.3, 0.7], 'color': "gray"},
                        {'range': [0.7, 1], 'color': "darkgray"}
                    ]
                }
            ), row=1, col=2)
            
            # Netzwerk-Kohäsion
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=societal.get('network_cohesion', 0),
                title={'text': "Network Cohesion"},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "darkblue"}
                }
            ), row=2, col=1)
            
            # Einflussverteilung
            influence_gini = societal.get('influence_degree_gini', 0)
            if np.isnan(influence_gini):
                influence_gini = 0
                
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=influence_gini,
                title={'text': "Influence Inequality"},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "darkorange"}
                }
            ), row=2, col=2)
            
            fig.update_layout(
                height=600,
                title_text="Real-time Metrics Dashboard",
                title_font_size=16
            )
            
            return fig.to_html(include_plotlyjs='cdn', div_id='metrics-dashboard-plot')
            
        except Exception as e:
            return f"<p>Error creating metrics dashboard: {e}</p>"
    
    def create_agent_comparison_radar(self):
        """Erstellt Radar-Chart für Agent-Vergleich"""
        if not self.society.agents:
            return "<p>No agents available for visualization</p>"
        
        # Definiere Vergleichsdimensionen
        dimensions = ['Decisions', 'Beliefs', 'Social_Connections', 'Confidence', 'Consistency']
        
        agent_data = []
        for agent in self.society.agents[:6]:  # Limitiere auf 6 Agenten für Übersichtlichkeit
            values = []
            
            # Entscheidungen (normalisiert)
            decisions = len(getattr(agent, 'decision_history', []))
            values.append(min(decisions / 10, 1))  # Normalisiert auf 0-1
            
            # Überzeugungen
            beliefs_count = len(getattr(agent.beliefs, 'beliefs', {})) if hasattr(agent, 'beliefs') else 0
            values.append(min(beliefs_count / 20, 1))
            
            # Soziale Verbindungen
            connections = self.society.social_network.degree(agent.agent_id) if agent.agent_id in self.society.social_network else 0
            values.append(min(connections / len(self.society.agents), 1))
            
            # Durchschnittliches Vertrauen (Placeholder)
            values.append(np.random.uniform(0.3, 0.9))
            
            # Konsistenz (Placeholder)
            values.append(np.random.uniform(0.4, 0.8))
            
            agent_data.append({
                'agent': agent.agent_id,
                'values': values,
                'personality': getattr(agent, 'personality', 'unknown')
            })
        
        # Erstelle Radar-Chart
        fig = go.Figure()
        
        for data in agent_data:
            fig.add_trace(go.Scatterpolar(
                r=data['values'] + [data['values'][0]],  # Schließe den Kreis
                theta=dimensions + [dimensions[0]],
                fill='toself',
                name=f"{data['agent']} ({data['personality']})",
                hovertemplate='<b>%{fullData.name}</b><br>%{theta}: %{r:.2f}<extra></extra>'
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Agent Comparison Radar Chart",
            height=500
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id='agent-radar-plot')
    
    def generate_complete_dashboard(self):
        """Generiert vollständiges interaktives Dashboard"""
        dashboard_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Interactive Ethics Simulation Dashboard</title>
            <meta charset="utf-8">
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .dashboard-container {{ max-width: 1400px; margin: 0 auto; }}
                .dashboard-header {{ text-align: center; padding: 20px; background: white; border-radius: 10px; margin-bottom: 20px; }}
                .plot-container {{ background: white; border-radius: 10px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .row {{ display: flex; gap: 20px; margin-bottom: 20px; }}
                .col-half {{ flex: 1; }}
                .refresh-btn {{ background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; }}
                .refresh-btn:hover {{ background: #0056b3; }}
            </style>
        </head>
        <body>
            <div class="dashboard-container">
                <div class="dashboard-header">
                    <h1>🧠 Interactive Ethics Simulation Dashboard</h1>
                    <p>Real-time visualization of ethical agent behavior and social dynamics</p>
                    <button class="refresh-btn" onclick="location.reload()">🔄 Refresh Dashboard</button>
                    <p><small>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small></p>
                </div>
                
                <div class="plot-container">
                    <h2>📊 Real-time Metrics</h2>
                    {self.create_metrics_dashboard()}
                </div>
                
                <div class="row">
                    <div class="col-half">
                        <div class="plot-container">
                            <h2>🕸️ 3D Network Visualization</h2>
                            {self.create_interactive_network_3d()}
                        </div>
                    </div>
                    <div class="col-half">
                        <div class="plot-container">
                            <h2>🎯 Agent Comparison</h2>
                            {self.create_agent_comparison_radar()}
                        </div>
                    </div>
                </div>
                
                <div class="plot-container">
                    <h2>🔥 Decision Patterns Heatmap</h2>
                    {self.create_decision_heatmap()}
                </div>
                
                <div class="plot-container">
                    <h2>📈 Belief Evolution Timeline</h2>
                    {self.create_belief_evolution_timeline()}
                </div>
                
                <div class="dashboard-header">
                    <h3>💡 Dashboard Features</h3>
                    <p><strong>Interactive Elements:</strong> Hover, zoom, pan, and click on all visualizations</p>
                    <p><strong>Real-time Updates:</strong> Refresh to see latest simulation data</p>
                    <p><strong>Export Options:</strong> Right-click on plots to download as PNG</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return dashboard_html

def create_enhanced_web_dashboard(society):
    """Hauptfunktion zur Erstellung des erweiterten Dashboards"""
    viz = PlotlyWebVisualizations(society)
    return viz.generate_complete_dashboard()

if __name__ == "__main__":
    # Test mit Demo-Gesellschaft
    print("🚀 Testing Enhanced Web Visualizations...")
    
    try:
        from neural_society import NeuralEthicalSociety
        from agents import NeuralEthicalAgent
        
        # Erstelle Test-Gesellschaft
        society = NeuralEthicalSociety()
        for i in range(6):
            agent = NeuralEthicalAgent(f"test_agent_{i}")
            society.add_agent(agent)
        
        # Generiere Dashboard
        viz = PlotlyWebVisualizations(society)
        dashboard_html = viz.generate_complete_dashboard()
        
        # Speichere Dashboard
        filename = f"enhanced_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(dashboard_html)
        
        print(f"✅ Enhanced dashboard created: {filename}")
        print("🌐 Open the file in your browser to view interactive visualizations!")
        
    except Exception as e:
        print(f"❌ Error creating enhanced dashboard: {e}")
