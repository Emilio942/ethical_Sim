import pytest
import os
import sys
import json

# Add root directory and src directory to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "src"))

from web.web_interface import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.config['JWT_SECRET_KEY'] = 'test-secret'
    with app.test_client() as client:
        yield client

def test_api_v2_flow(client):
    # 1. Login
    response = client.post("/api/v2/login", json={"username": "user", "password": "pass"})
    assert response.status_code == 200
    token = response.json.get("access_token")
    assert token is not None
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Start Simulation
    response = client.post(
        "/api/v2/simulation/start",
        json={"numAgents": 3, "numScenarios": 5},
        headers=headers,
    )
    assert response.status_code in [200, 201]

    import time
    time.sleep(2) # Wait for thread to initialize society

    # 3. Build Scenario
    response = client.post(
        "/api/v2/scenarios/build",
        json={
            "scenario_id": "test_scenario",
            "description": "TestScenario",
            "options": {"optA": {"Gerechtigkeit": 0.5, "Fairness": 0.3}},
        },
        headers=headers,
    )
    assert response.status_code in [200, 201]

    # 4. Network3D Visualization
    response = client.get("/api/v2/visualizations/network3d", headers=headers)
    # It might return 404 if no simulation data exists yet, or 200 with empty data
    assert response.status_code in [200, 404]
