#!/usr/bin/env python3
"""
Web Interface für die Ethische Agenten-Simulation
==================================================

Ein Flask-basiertes Web-Interface für interaktive Simulationen,
Echtzeit-Visualisierungen und benutzerfreundliche Konfiguration.

Features:
- Interaktive Agenten-Konfiguration
- Echtzeit-Simulation mit Live-Updates
- Interaktive Visualisierungen
- Szenario-Builder
- Export-Downloads
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_jwt_extended import JWTManager, create_access_token, jwt_required
from ..visualization.plotly_web_visualizations import PlotlyWebVisualizations
from flask_socketio import SocketIO, emit
import json
import os
import threading
import time
from datetime import datetime
import tempfile

# Import der Simulation-Module
from ..society.neural_society import NeuralEthicalSociety
from ..agents.agents import NeuralEthicalAgent
from ..scenarios.scenarios import ScenarioGenerator
from ..analysis.metrics import MetricsCollector
from ..analysis.validation import ValidationSuite
from ..analysis.export_reporting import DataExporter, AutomatedReporter
from ..visualization.visualization import EthicalSimulationVisualizer

app = Flask(__name__)
app.secret_key = 'ethical_agents_simulation_2025'
jwt = JWTManager(app)

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state für die Simulation
simulation_state = {
    'society': None,
    'running': False,
    'progress': 0,
    'current_scenario': 0,
    'total_scenarios': 0,
    'metrics': {},
    'validation_results': {},
    'agents_data': []
}

class SimulationThread(threading.Thread):
    """Thread für die Hintergrund-Simulation"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.daemon = True
        
    def run(self):
        try:
            # Simulation initialisieren
            simulation_state['running'] = True
            socketio.emit('simulation_started', {'status': 'running'})
            simulation_state['progress'] = 0
            
            # Gesellschaft erstellen
            society = NeuralEthicalSociety()
            
            # Agenten hinzufügen
            agent_configs = self.config.get('agents', [])
            for i, agent_config in enumerate(agent_configs):
                agent = NeuralEthicalAgent(
                    agent_id=f"agent_{i}",
                    personality=agent_config.get('personality', 'balanced'),
                    ethical_framework=agent_config.get('framework', 'utilitarian')
                )
                society.add_agent(agent)
            
            simulation_state['society'] = society
            
            # Szenarien generieren
            scenario_gen = ScenarioGenerator()
            scenarios = []
            num_scenarios = self.config.get('num_scenarios', 10)
            
            for i in range(num_scenarios):
                scenario = scenario_gen.generate_random_scenario()
                scenarios.append(scenario)
            
            simulation_state['total_scenarios'] = len(scenarios)
            
            # Simulation durchführen
            for i, scenario in enumerate(scenarios):
                if not simulation_state['running']:
                    break
                    
                simulation_state['current_scenario'] = i + 1
                simulation_state['progress'] = int((i + 1) / len(scenarios) * 100)
                
                # Scenario durchführen
                decisions = society.run_scenario(scenario)
                
                # Kurze Pause für UI-Updates
                time.sleep(0.5)
            
            # Metriken sammeln
            metrics_collector = MetricsCollector()
            simulation_state['metrics'] = metrics_collector.collect_all_metrics(society)
            
            # Validierung durchführen
            validator = ValidationSuite()
            simulation_state['validation_results'] = validator.validate_society(society)
            
            # Agenten-Daten für Visualisierung
            simulation_state['agents_data'] = [
                {
                    'id': agent.agent_id,
                    'personality': agent.personality,
                    'framework': agent.ethical_framework,
                    'beliefs': dict(agent.beliefs.beliefs),
                    'decision_count': len(agent.decision_history)
                }
                for agent in society.agents
            ]
            
            simulation_state['progress'] = 100
            
        except Exception as e:
            print(f"Simulation error: {e}")
        finally:
            simulation_state['running'] = False
            socketio.emit('simulation_finished', {'status': 'stopped'})

@app.route('/')
def index():
    """Hauptseite"""
    return render_template('index.html')

@app.route('/api/start_simulation', methods=['POST'])
def start_simulation():
    """Startet eine neue Simulation"""
    if simulation_state['running']:
        return jsonify({'error': 'Simulation already running'}), 400
    
    config = request.json
    thread = SimulationThread(config)
    thread.start()
    
    return jsonify({'status': 'started'})

@app.route('/api/stop_simulation', methods=['POST'])
def stop_simulation():
    """Stoppt die laufende Simulation"""
    simulation_state['running'] = False
    return jsonify({'status': 'stopped'})

@app.route('/api/simulation_status')
def simulation_status():
    """Gibt den aktuellen Simulationsstatus zurück"""
    return jsonify({
        'running': simulation_state['running'],
        'progress': simulation_state['progress'],
        'current_scenario': simulation_state['current_scenario'],
        'total_scenarios': simulation_state['total_scenarios']
    })

@app.route('/api/metrics')
def get_metrics():
    """Gibt die aktuellen Metriken zurück"""
    return jsonify(simulation_state['metrics'])

@app.route('/api/validation')
def get_validation():
    """Gibt die Validierungsergebnisse zurück"""
    return jsonify(simulation_state['validation_results'])

@app.route('/api/agents')
def get_agents():
    """Gibt die Agenten-Daten zurück"""
    return jsonify(simulation_state['agents_data'])

@app.route('/api/export/<format>')
def export_data(format):
    """Exportiert Daten in verschiedenen Formaten"""
    if not simulation_state['society']:
        return jsonify({'error': 'No simulation data available'}), 400
    
    try:
        exporter = DataExporter(simulation_state['society'])
        
        # Temporäre Datei erstellen
        temp_dir = tempfile.mkdtemp()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == 'json':
            filename = f"simulation_export_{timestamp}.json"
            filepath = os.path.join(temp_dir, filename)
            exporter.export_json(filepath)
            
        elif format == 'csv':
            filename = f"simulation_export_{timestamp}.csv"
            filepath = os.path.join(temp_dir, filename)
            exporter.export_csv(filepath)
            
        elif format == 'html':
            filename = f"simulation_report_{timestamp}.html"
            filepath = os.path.join(temp_dir, filename)
            exporter.generate_html_report(filepath)
            
        else:
            return jsonify({'error': 'Unsupported format'}), 400
        
        return send_file(filepath, as_attachment=True, download_name=filename)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/visualizations')
def get_visualizations():
    """Generiert Visualisierungen für das Web-Interface"""
    if not simulation_state['society']:
        return jsonify({'error': 'No simulation data available'}), 400
    
    try:
        viz = EthicalSimulationVisualizer(simulation_state['society'])
        
        # Hier würden wir Plotly-Grafiken für das Web generieren
        # Für jetzt geben wir Mock-Daten zurück
        visualizations = {
            'network_graph': {'type': 'network', 'data': 'placeholder'},
            'belief_evolution': {'type': 'line', 'data': 'placeholder'},
            'decision_matrix': {'type': 'heatmap', 'data': 'placeholder'}
        }
        
        return jsonify(visualizations)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/dashboard')
def dashboard():
    """Zeigt das interaktive Dashboard"""
    if not simulation_state['society']:
        # Erstelle eine Demo-Society falls keine existiert
        from neural_society import NeuralEthicalSociety
        from agents import NeuralEthicalAgent
        
        demo_society = NeuralEthicalSociety()
        for i in range(3):
            agent = NeuralEthicalAgent(agent_id=f'demo_agent_{i}')
            agent.personality = ['utilitarian', 'deontological', 'virtue'][i % 3]
            demo_society.add_agent(agent)
        simulation_state['society'] = demo_society
    
    try:
        from ..visualization.simple_interactive_dashboard import create_simple_interactive_dashboard
        dashboard_html = create_simple_interactive_dashboard(simulation_state['society'])
        return dashboard_html
    except Exception as e:
        return f"""
        <html>
        <head><title>Dashboard Error</title></head>
        <body>
            <h1>Dashboard Error</h1>
            <p>Fehler beim Laden des Dashboards: {str(e)}</p>
            <a href="/">Zurück zur Hauptseite</a>
        </body>
        </html>
        """

# === API v2: Authentication ===
@app.route('/api/v2/login', methods=['POST'])
def login():
    username = request.json.get('username', None)
    password = request.json.get('password', None)
    # Simplified: Accept any credentials
    access_token = create_access_token(identity=username)
    return jsonify(access_token=access_token), 200

# === API v2: Start Simulation ===
@app.route('/api/v2/simulation/start', methods=['POST'])
@jwt_required()
def api_v2_start_simulation():
    config = request.json
    # Reuse existing simulation start logic
    if simulation_state['running']:
        return jsonify({'error': 'Simulation already running'}), 400
    
    thread = SimulationThread(config)
    thread.start()
    
    return jsonify({'status': 'simulation started', 'config': config}), 200

# === API v2: Build Custom Scenario ===
@app.route('/api/v2/scenarios/build', methods=['POST'])
@jwt_required()
def build_scenario_v2():
    params = request.json or {}
    try:
        generator = ScenarioGenerator()
        options = params.get('options', {})
        scenario = generator.create_custom_scenario(
            scenario_id=params.get('scenario_id'),
            description=params.get('description', ''),
            options=options,
            relevant_beliefs=params.get('relevant_beliefs', {}),
            option_attributes=params.get('option_attributes', {}),
            outcome_feedback=params.get('outcome_feedback', {}),
            moral_implications=params.get('moral_implications', {})
        )
        return jsonify({'scenario_id': scenario.scenario_id}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# === API v2: Plotly 3D Network Visualization ===
@app.route('/api/v2/visualizations/network3d')
@jwt_required()
def network3d_v2():
    if not simulation_state.get('society'):
        return jsonify(error='No society available'), 400
    viz = PlotlyWebVisualizations(simulation_state['society'])
    html = viz.create_interactive_network_3d()
    return html

# Template-Ordner erstellen falls nicht vorhanden
if not os.path.exists('templates'):
    os.makedirs('templates')

# Statische Dateien-Ordner erstellen
if not os.path.exists('static'):
    os.makedirs('static')
    os.makedirs('static/css')
    os.makedirs('static/js')

if __name__ == '__main__':
    print("🌐 Starting Ethical Agents Simulation Web Interface...")
    print("📱 Access the interface at: http://localhost:5000")
    print("🔧 Debug mode enabled for development")
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
