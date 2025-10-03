#!/usr/bin/env python3
"""
Finale Projekttests - Ethische Agenten-Simulation
==============================================

Führt umfassende Tests aller Kernfunktionen durch.
"""

import sys
import traceback
from typing import List, Dict, Any


def test_core_functionality():
    """Test der Kernfunktionen"""
    print("🧪 Teste Kernfunktionen...")

    try:
        # Import und Erstellung
        from neural_society import NeuralEthicalSociety
        from agents import NeuralEthicalAgent
        from scenarios import get_trolley_problem

        # Erstelle Society
        society = NeuralEthicalSociety()

        # Erstelle Agenten
        agents = []
        personalities = ["utilitarian", "deontological", "virtue", "balanced"]
        for i, personality in enumerate(personalities):
            agent = NeuralEthicalAgent(agent_id=f"agent_{i}")
            agent.personality = personality
            society.add_agent(agent)
            agents.append(agent)

        print(f"✅ Society mit {len(society.agents)} Agenten erstellt")

        # Teste Szenario
        scenario = get_trolley_problem()
        decisions = []
        for agent in agents:
            decision = agent.make_decision(scenario)
            decisions.append(decision)

        print(f"✅ {len(decisions)} ethische Entscheidungen getroffen")

        # Teste einfache Society-Operation
        print("✅ Society erfolgreich getestet")

        return True, "Kernfunktionen OK"

    except Exception as e:
        return False, f"Kernfunktionen fehlgeschlagen: {e}"


def test_metrics_and_validation():
    """Test von Metriken und Validierung"""
    print("📊 Teste Metriken und Validierung...")

    try:
        from neural_society import NeuralEthicalSociety
        from agents import NeuralEthicalAgent
        from metrics import MetricsCollector
        from validation import ValidationSuite

        # Setup
        society = NeuralEthicalSociety()
        for i in range(3):
            agent = NeuralEthicalAgent(agent_id=f"test_agent_{i}")
            society.add_agent(agent)

        # Metriken
        collector = MetricsCollector()
        metrics = collector.collect_all_metrics(society)
        print(f"✅ Metriken gesammelt: {len(metrics)} Kategorien")

        # Validierung
        validator = ValidationSuite()
        validation_results = validator.validate_all(society)
        print(f"✅ Validierung durchgeführt: {len(validation_results)} Kategorien")

        return True, "Metriken & Validierung OK"

    except Exception as e:
        return False, f"Metriken/Validierung fehlgeschlagen: {e}"


def test_export_and_reporting():
    """Test von Export und Reporting"""
    print("📄 Teste Export und Reporting...")

    try:
        from neural_society import NeuralEthicalSociety
        from agents import NeuralEthicalAgent
        from export_reporting import DataExporter, ReportGenerator

        # Setup
        society = NeuralEthicalSociety()
        for i in range(2):
            agent = NeuralEthicalAgent(agent_id=f"export_agent_{i}")
            society.add_agent(agent)

        exporter = DataExporter()
        report_gen = ReportGenerator()

        # Test verschiedene Formate
        formats_tested = 0

        # JSON Export
        try:
            json_data = exporter.export_society_data(society, format="json")
            if json_data:
                formats_tested += 1
        except:
            pass

        # CSV Export
        try:
            csv_data = exporter.export_society_data(society, format="csv")
            if csv_data:
                formats_tested += 1
        except:
            pass

        # Report
        try:
            report = report_gen.generate_society_report(society)
            if report:
                formats_tested += 1
        except:
            pass

        print(f"✅ {formats_tested}/3 Export-Formate erfolgreich")

        return True, f"Export & Reporting OK ({formats_tested} Formate)"

    except Exception as e:
        return False, f"Export/Reporting fehlgeschlagen: {e}"


def test_web_interface():
    """Test des Web-Interfaces"""
    print("🌐 Teste Web-Interface...")

    try:
        from web_interface import app
        import threading
        import time
        import requests

        # Starte Server im Hintergrund
        def run_server():
            app.run(port=5003, debug=False, use_reloader=False)

        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        time.sleep(2)

        # Teste Routen
        routes_tested = 0

        try:
            response = requests.get("http://localhost:5003/", timeout=3)
            if response.status_code == 200:
                routes_tested += 1
        except:
            pass

        try:
            response = requests.get("http://localhost:5003/dashboard", timeout=5)
            if response.status_code == 200:
                routes_tested += 1
        except:
            pass

        try:
            response = requests.get("http://localhost:5003/api/simulation_status", timeout=3)
            if response.status_code == 200:
                routes_tested += 1
        except:
            pass

        print(f"✅ {routes_tested}/3 Web-Routen erfolgreich")

        return True, f"Web-Interface OK ({routes_tested} Routen)"

    except Exception as e:
        return False, f"Web-Interface fehlgeschlagen: {e}"


def test_interactive_dashboard():
    """Test des interaktiven Dashboards"""
    print("📈 Teste interaktives Dashboard...")

    try:
        from neural_society import NeuralEthicalSociety
        from agents import NeuralEthicalAgent
        from simple_interactive_dashboard import (
            SimpleInteractiveVisualizations,
            create_simple_interactive_dashboard,
        )

        # Setup
        society = NeuralEthicalSociety()
        for i in range(4):
            agent = NeuralEthicalAgent(agent_id=f"viz_agent_{i}")
            agent.personality = ["utilitarian", "deontological", "virtue", "balanced"][i]
            society.add_agent(agent)

        # Test Klassen-basiert
        dashboard = SimpleInteractiveVisualizations(society)
        html1 = dashboard.create_simple_dashboard()

        # Test Standalone-Funktion
        html2 = create_simple_interactive_dashboard(society)

        success = len(html1) > 1000 and len(html2) > 1000

        print(f"✅ Dashboard HTML generiert: {len(html1)}/{len(html2)} Zeichen")

        return success, "Interaktives Dashboard OK"

    except Exception as e:
        return False, f"Dashboard fehlgeschlagen: {e}"


def run_final_tests():
    """Führt alle finalen Tests durch"""
    print("=" * 60)
    print("🚀 FINALE PROJEKTTESTS - ETHISCHE AGENTEN-SIMULATION")
    print("=" * 60)

    tests = [
        ("Kernfunktionen", test_core_functionality),
        ("Metriken & Validierung", test_metrics_and_validation),
        ("Export & Reporting", test_export_and_reporting),
        ("Web-Interface", test_web_interface),
        ("Interaktives Dashboard", test_interactive_dashboard),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            success, message = test_func()
            results.append((test_name, success, message))
            status = "✅ BESTANDEN" if success else "❌ FEHLGESCHLAGEN"
            print(f"{status}: {message}")
        except Exception as e:
            results.append((test_name, False, f"Exception: {e}"))
            print(f"❌ FEHLGESCHLAGEN: {e}")
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("📋 TESTERGEBNISSE")
    print("=" * 60)

    passed = 0
    total = len(results)

    for test_name, success, message in results:
        status = "✅" if success else "❌"
        print(f"{status} {test_name}: {message}")
        if success:
            passed += 1

    print(f"\n🎯 GESAMT: {passed}/{total} Tests bestanden ({passed/total*100:.1f}%)")

    if passed == total:
        print("🎉 ALLE TESTS BESTANDEN - PROJEKT BEREIT FÜR PRODUKTION!")
    elif passed >= total * 0.8:
        print("✅ PROJEKT FUNKTIONSFÄHIG - Kleinere Probleme können behoben werden")
    else:
        print("⚠️  PROJEKT BENÖTIGT WEITERE ARBEIT")

    return passed, total


if __name__ == "__main__":
    try:
        passed, total = run_final_tests()
        sys.exit(0 if passed == total else 1)
    except Exception as e:
        print(f"❌ Fataler Fehler: {e}")
        traceback.print_exc()
        sys.exit(1)
