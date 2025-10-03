import requests
import time

# Wait a moment for the server to start
for i in range(3):
    try:
        requests.get("http://127.0.0.1:5000/")
        break
    except requests.exceptions.ConnectionError:
        time.sleep(1)

# 1. Login
print("Testing /api/v2/login")
r = requests.post(
    "http://127.0.0.1:5000/api/v2/login", json={"username": "user", "password": "pass"}
)
print("Status:", r.status_code, "Response:", r.json())
if r.status_code != 200:
    exit(1)

token = r.json().get("access_token")
headers = {"Authorization": f"Bearer {token}"}

# 2. Start Simulation
print("\nTesting /api/v2/simulation/start")
r2 = requests.post(
    "http://127.0.0.1:5000/api/v2/simulation/start",
    json={"numAgents": 3, "numScenarios": 5},
    headers=headers,
)
print("Status:", r2.status_code, "Response:", r2.json())

# 3. Build Scenario
print("\nTesting /api/v2/scenarios/build")
r3 = requests.post(
    "http://127.0.0.1:5000/api/v2/scenarios/build",
    json={
        "scenario_id": "test_scenario",
        "description": "TestScenario",
        "options": {"optA": {"Gerechtigkeit": 0.5, "Fairness": 0.3}},
    },
    headers=headers,
)
print("Status:", r3.status_code, "Response:", r3.json())

# 4. Network3D Visualization
print("\nTesting /api/v2/visualizations/network3d")
r4 = requests.get("http://127.0.0.1:5000/api/v2/visualizations/network3d", headers=headers)
print("Status:", r4.status_code)
if r4.status_code == 200:
    print("HTML snippet:", r4.text[:200].replace("\n", " "))
else:
    print("Error payload:", r4.text)

print("\n✅ API v2 Tests abgeschlossen")
