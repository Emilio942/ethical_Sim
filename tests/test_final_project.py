import pytest
import sys
import os
import threading
import time
import requests

# Add root directory and src directory to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "src"))

from society.neural_society import NeuralEthicalSociety
from agents.neural_agent import NeuralEthicalAgent
from scenarios.scenarios import get_trolley_problem
from analysis.metrics import MetricsCollector
from analysis.validation import ValidationSuite
from analysis.export_reporting import DataExporter, ReportGenerator, ExportConfig, ExportFormat
from visualization.simple_interactive_dashboard import SimpleInteractiveVisualizations, create_simple_interactive_dashboard
from web.web_interface import app

def test_core_functionality():
    """Test core functionality."""
    society = NeuralEthicalSociety()
    agents = []
    personalities = ["utilitarian", "deontological", "virtue", "balanced"]
    
    for i, personality in enumerate(personalities):
        agent = NeuralEthicalAgent(agent_id=f"agent_{i}")
        # Assuming personality is a property or attribute we can set, 
        # though NeuralEthicalAgent constructor takes personality_traits dict.
        # The original test did agent.personality = personality.
        # I'll stick to what the original test did, assuming it works or the agent allows dynamic attributes.
        agent.personality = personality 
        society.add_agent(agent)
        agents.append(agent)

    assert len(society.agents) == 4

    scenario = get_trolley_problem()
    decisions = []
    for agent in agents:
        decision = agent.make_decision(scenario)
        decisions.append(decision)

    assert len(decisions) == 4

def test_metrics_and_validation():
    """Test metrics and validation."""
    society = NeuralEthicalSociety()
    for i in range(3):
        agent = NeuralEthicalAgent(agent_id=f"test_agent_{i}")
        society.add_agent(agent)

    collector = MetricsCollector()
    metrics = collector.collect_all_metrics(society)
    assert len(metrics) > 0

    validator = ValidationSuite()
    validation_results = validator.validate_all(society)
    assert len(validation_results) > 0

def test_export_and_reporting():
    """Test export and reporting."""
    society = NeuralEthicalSociety()
    for i in range(2):
        agent = NeuralEthicalAgent(agent_id=f"export_agent_{i}")
        society.add_agent(agent)

    exporter = DataExporter()
    report_gen = ReportGenerator()

    formats_tested = 0

    # JSON Export
    try:
        config = ExportConfig(format=ExportFormat.JSON, filepath="test_export.json")
        json_data = exporter.export_society_data(society, config)
        if json_data:
            formats_tested += 1
    except Exception as e:
        print(f"JSON Export failed: {e}")

    # CSV Export
    try:
        config = ExportConfig(format=ExportFormat.CSV, filepath="test_export.csv")
        csv_data = exporter.export_society_data(society, config)
        if csv_data:
            formats_tested += 1
    except Exception as e:
        print(f"CSV Export failed: {e}")

    # Report
    try:
        report = report_gen.generate_simulation_summary(society)
        if report:
            formats_tested += 1
    except Exception as e:
        print(f"Report failed: {e}")

    # Cleanup
    if os.path.exists("test_export.json"):
        os.remove("test_export.json")
    if os.path.exists("test_export.csv"):
        os.remove("test_export.csv")

    assert formats_tested == 3

    assert formats_tested == 3

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_web_interface(client):
    """Test web interface routes."""
    response = client.get("/")
    assert response.status_code == 200
    
    response = client.get("/dashboard")
    assert response.status_code == 200
    
    response = client.get("/api/simulation_status")
    assert response.status_code == 200

def test_interactive_dashboard():
    """Test interactive dashboard generation."""
    society = NeuralEthicalSociety()
    for i in range(4):
        agent = NeuralEthicalAgent(agent_id=f"viz_agent_{i}")
        # agent.personality = ... (original test did this, but it might not be needed for dashboard)
        society.add_agent(agent)

    dashboard = SimpleInteractiveVisualizations(society)
    html1 = dashboard.create_simple_dashboard()

    html2 = create_simple_interactive_dashboard(society)

    assert len(html1) > 1000
    assert len(html2) > 1000
