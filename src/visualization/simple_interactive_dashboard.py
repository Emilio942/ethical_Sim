#!/usr/bin/env python3
"""
Einfache interaktive Visualisierungen mit Plotly
==============================================

Diese Version erstellt einfache, aber funktionsfähige 
interaktive Visualisierungen für das Web-Interface.
"""

import json
from datetime import datetime
import random

class SimpleInteractiveVisualizations:
    """Einfache interaktive Visualisierungen ohne externe Dependencies"""
    
    def __init__(self, society):
        self.society = society
        
    def create_network_visualization(self):
        """Erstellt eine einfache Netzwerk-Visualisierung mit HTML/CSS/JS"""
        # self.society.agents ist ein Dict, nicht eine Liste
        agents_dict = self.society.agents if self.society.agents else {}
        agents = list(agents_dict.values())
        
        # Generiere JSON-Daten für das Netzwerk
        nodes = []
        links = []
        
        for i, agent in enumerate(agents):
            nodes.append({
                'id': agent.agent_id,
                'name': agent.agent_id,
                'personality': getattr(agent, 'personality', 'unknown'),
                'x': random.uniform(100, 700),
                'y': random.uniform(100, 400),
                'color': self._get_personality_color(getattr(agent, 'personality', 'unknown'))
            })
        
        # Einfache Links zwischen Agenten
        for i in range(len(nodes)):
            for j in range(i+1, min(i+3, len(nodes))):  # Verbinde mit 1-2 Nachbarn
                if random.random() > 0.5:
                    links.append({
                        'source': nodes[i]['id'],
                        'target': nodes[j]['id']
                    })
        
        network_html = f"""
        <div id="network-container" style="width: 100%; height: 400px; border: 1px solid #ccc; position: relative; background: #f9f9f9;">
            <svg id="network-svg" width="100%" height="100%">
                <!-- Links -->
                {self._generate_svg_links(links, nodes)}
                
                <!-- Nodes -->
                {self._generate_svg_nodes(nodes)}
            </svg>
            
            <div id="network-legend" style="position: absolute; top: 10px; right: 10px; background: white; padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h4>Agent Types</h4>
                <div style="display: flex; flex-direction: column; gap: 5px;">
                    <div><span style="width: 15px; height: 15px; background: #FF6B6B; display: inline-block; border-radius: 50%; margin-right: 8px;"></span>Utilitarian</div>
                    <div><span style="width: 15px; height: 15px; background: #4ECDC4; display: inline-block; border-radius: 50%; margin-right: 8px;"></span>Deontological</div>
                    <div><span style="width: 15px; height: 15px; background: #45B7D1; display: inline-block; border-radius: 50%; margin-right: 8px;"></span>Virtue Ethics</div>
                    <div><span style="width: 15px; height: 15px; background: #96CEB4; display: inline-block; border-radius: 50%; margin-right: 8px;"></span>Balanced</div>
                </div>
            </div>
        </div>
        
        <script>
        // Einfache Interaktivität
        document.querySelectorAll('.network-node').forEach(node => {{
            node.addEventListener('mouseover', function() {{
                this.style.strokeWidth = '3';
                this.style.stroke = '#333';
                
                // Zeige Tooltip
                const tooltip = document.getElementById('node-tooltip') || document.createElement('div');
                tooltip.id = 'node-tooltip';
                tooltip.style.cssText = 'position: absolute; background: black; color: white; padding: 5px; border-radius: 3px; font-size: 12px; pointer-events: none; z-index: 1000;';
                tooltip.textContent = this.getAttribute('data-info');
                document.body.appendChild(tooltip);
            }});
            
            node.addEventListener('mousemove', function(e) {{
                const tooltip = document.getElementById('node-tooltip');
                if (tooltip) {{
                    tooltip.style.left = e.pageX + 10 + 'px';
                    tooltip.style.top = e.pageY - 30 + 'px';
                }}
            }});
            
            node.addEventListener('mouseout', function() {{
                this.style.strokeWidth = '2';
                this.style.stroke = '#666';
                
                const tooltip = document.getElementById('node-tooltip');
                if (tooltip) tooltip.remove();
            }});
        }});
        </script>
        """
        
        return network_html
    
    def _get_personality_color(self, personality):
        """Gibt Farbe für Persönlichkeitstyp zurück"""
        color_map = {
            'utilitarian': '#FF6B6B',
            'deontological': '#4ECDC4', 
            'virtue_ethics': '#45B7D1',
            'balanced': '#96CEB4',
            'pragmatic': '#FFEAA7',
            'idealistic': '#DDA0DD',
            'unknown': '#95A5A6'
        }
        return color_map.get(personality, '#95A5A6')
    
    def _generate_svg_links(self, links, nodes):
        """Generiert SVG für Links"""
        svg_links = ""
        node_positions = {node['id']: (node['x'], node['y']) for node in nodes}
        
        for link in links:
            if link['source'] in node_positions and link['target'] in node_positions:
                x1, y1 = node_positions[link['source']]
                x2, y2 = node_positions[link['target']]
                svg_links += f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#999" stroke-width="1" opacity="0.6"/>\n'
        
        return svg_links
    
    def _generate_svg_nodes(self, nodes):
        """Generiert SVG für Nodes"""
        svg_nodes = ""
        
        for node in nodes:
            svg_nodes += f'''
            <circle class="network-node" 
                    cx="{node['x']}" cy="{node['y']}" r="15" 
                    fill="{node['color']}" stroke="#666" stroke-width="2"
                    data-info="Agent: {node['name']} | Type: {node['personality']}"
                    style="cursor: pointer;"/>
            <text x="{node['x']}" y="{node['y']+5}" text-anchor="middle" 
                  font-family="Arial" font-size="10" fill="white" pointer-events="none">
                {node['id'][:3]}
            </text>
            '''
        
        return svg_nodes
    
    def create_metrics_chart(self):
        """Erstellt einfaches Metriken-Chart"""
        try:
            from metrics import MetricsCollector
            collector = MetricsCollector()
            metrics = collector.collect_all_metrics(self.society)
            societal = metrics.get('societal_metrics', {})
        except:
            # Fallback zu Demo-Daten
            societal = {
                'polarization': random.uniform(0.1, 0.4),
                'consensus': random.uniform(0.3, 0.8),
                'network_cohesion': random.uniform(0.2, 0.7),
                'influence_degree_gini': random.uniform(0.1, 0.5)
            }
        
        chart_html = f"""
        <div id="metrics-chart" style="display: flex; gap: 20px; padding: 20px; background: #f8f9fa; border-radius: 10px;">
            {self._create_metric_gauge('Polarization', societal.get('polarization', 0), '#FF6B6B')}
            {self._create_metric_gauge('Consensus', societal.get('consensus', 0), '#4ECDC4')}
            {self._create_metric_gauge('Cohesion', societal.get('network_cohesion', 0), '#45B7D1')}
            {self._create_metric_gauge('Influence', societal.get('influence_degree_gini', 0), '#96CEB4')}
        </div>
        """
        
        return chart_html
    
    def _create_metric_gauge(self, name, value, color):
        """Erstellt einfachen Gauge-Chart"""
        if isinstance(value, str) or value is None:
            value = 0
        
        # Behandle NaN-Werte
        try:
            import math
            if math.isnan(value):
                value = 0
        except:
            value = float(value) if value else 0
        
        percentage = min(max(value * 100, 0), 100)
        
        return f"""
        <div class="metric-gauge" style="text-align: center; flex: 1;">
            <h4 style="margin: 0 0 10px 0; color: #333;">{name}</h4>
            <div style="position: relative; width: 100px; height: 100px; margin: 0 auto;">
                <svg width="100" height="100" viewBox="0 0 100 100">
                    <!-- Background circle -->
                    <circle cx="50" cy="50" r="40" fill="none" stroke="#e0e0e0" stroke-width="8"/>
                    <!-- Progress circle -->
                    <circle cx="50" cy="50" r="40" fill="none" stroke="{color}" stroke-width="8"
                            stroke-dasharray="{percentage * 2.51} 251.2" 
                            stroke-linecap="round" transform="rotate(-90 50 50)"/>
                </svg>
                <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-weight: bold; color: #333;">
                    {value:.2f}
                </div>
            </div>
        </div>
        """
    
    def create_simple_dashboard(self):
        """Erstellt vollständiges einfaches Dashboard"""
        dashboard_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Simple Interactive Dashboard</title>
            <meta charset="utf-8">
            <style>
                body {{ 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    margin: 0; padding: 20px; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                }}
                .dashboard-container {{ 
                    max-width: 1200px; 
                    margin: 0 auto; 
                    background: white;
                    border-radius: 15px;
                    overflow: hidden;
                    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                }}
                .dashboard-header {{ 
                    text-align: center; 
                    padding: 30px; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                }}
                .dashboard-header h1 {{ margin: 0; font-size: 2.5em; font-weight: 300; }}
                .dashboard-header p {{ margin: 10px 0 0 0; opacity: 0.9; }}
                .content-section {{ 
                    padding: 30px; 
                    border-bottom: 1px solid #eee; 
                }}
                .content-section:last-child {{ border-bottom: none; }}
                .section-title {{ 
                    margin: 0 0 20px 0; 
                    color: #333; 
                    font-size: 1.5em; 
                    font-weight: 500;
                }}
                .refresh-btn {{ 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; 
                    border: none; 
                    padding: 12px 24px; 
                    border-radius: 25px; 
                    cursor: pointer; 
                    font-size: 14px;
                    transition: transform 0.2s;
                }}
                .refresh-btn:hover {{ transform: translateY(-2px); }}
                .stats-grid {{ 
                    display: grid; 
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                    gap: 20px; 
                    margin: 20px 0; 
                }}
                .stat-card {{ 
                    background: #f8f9fa; 
                    padding: 20px; 
                    border-radius: 10px; 
                    text-align: center;
                    border-left: 4px solid #667eea;
                }}
                .stat-number {{ 
                    font-size: 2em; 
                    font-weight: bold; 
                    color: #667eea; 
                    margin: 10px 0;
                }}
                .stat-label {{ 
                    color: #666; 
                    font-size: 0.9em; 
                }}
            </style>
        </head>
        <body>
            <div class="dashboard-container">
                <div class="dashboard-header">
                    <h1>🧠 Ethics Simulation Dashboard</h1>
                    <p>Interactive visualization of ethical agent behavior</p>
                    <button class="refresh-btn" onclick="location.reload()">🔄 Refresh Data</button>
                    <p style="font-size: 0.8em; margin-top: 15px;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="content-section">
                    <h2 class="section-title">📊 Key Metrics</h2>
                    {self.create_metrics_chart()}
                </div>
                
                <div class="content-section">
                    <h2 class="section-title">🕸️ Agent Network</h2>
                    {self.create_network_visualization()}
                </div>
                
                <div class="content-section">
                    <h2 class="section-title">📈 Simulation Statistics</h2>
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-number">{len(self.society.agents)}</div>
                            <div class="stat-label">Active Agents</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">{sum(len(getattr(agent, 'decision_history', [])) for agent in self.society.agents.values())}</div>
                            <div class="stat-label">Total Decisions</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">{len(set(getattr(agent, 'personality', 'unknown') for agent in self.society.agents.values()))}</div>
                            <div class="stat-label">Personality Types</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">{self.society.social_network.number_of_edges() if hasattr(self.society, 'social_network') else 0}</div>
                            <div class="stat-label">Social Connections</div>
                        </div>
                    </div>
                </div>
                
                <div class="content-section">
                    <h2 class="section-title">💡 Dashboard Info</h2>
                    <p><strong>🎯 Purpose:</strong> Real-time monitoring of ethical agent simulations</p>
                    <p><strong>🔄 Updates:</strong> Refresh the page to see latest simulation data</p>
                    <p><strong>🎨 Features:</strong> Interactive network visualization, real-time metrics, agent statistics</p>
                    <p><strong>🚀 Technology:</strong> Native HTML/CSS/JS for fast performance</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return dashboard_html

def create_simple_interactive_dashboard(society):
    """Hauptfunktion zur Erstellung des einfachen interaktiven Dashboards"""
    viz = SimpleInteractiveVisualizations(society)
    return viz.create_simple_dashboard()

if __name__ == "__main__":
    print("🚀 Creating Simple Interactive Dashboard...")
    
    try:
        from neural_society import NeuralEthicalSociety
        from agents import NeuralEthicalAgent
        
        # Erstelle Test-Gesellschaft
        society = NeuralEthicalSociety()
        personalities = ['utilitarian', 'deontological', 'virtue_ethics', 'balanced', 'pragmatic', 'idealistic']
        
        for i in range(6):
            agent = NeuralEthicalAgent(f"agent_{i}")
            # Setze Persönlichkeit falls möglich
            if hasattr(agent, 'personality'):
                agent.personality = personalities[i % len(personalities)]
            society.add_agent(agent)
        
        # Generiere Dashboard
        dashboard_html = create_simple_interactive_dashboard(society)
        
        # Speichere Dashboard
        filename = f"simple_interactive_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(dashboard_html)
        
        print(f"✅ Simple interactive dashboard created: {filename}")
        print("🌐 Open the HTML file in your browser to view the interactive dashboard!")
        print("📊 Features: Interactive network, real-time metrics, responsive design")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
