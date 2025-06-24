#!/usr/bin/env python3
"""
Einfache Demo für die reorganisierte Ethische Agenten-Simulation
================================================================

Diese Demo zeigt die grundlegende Funktionalität in der neuen Ordnerstruktur.
"""

import sys
import os

# Füge src-Verzeichnis zum Python-Pfad hinzu
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', 'src')
sys.path.insert(0, src_path)

def main():
    print("🧠 Ethische Agenten-Simulation - Einfache Demo")
    print("=" * 60)
    
    try:
        # Teste Imports
        print("📦 Teste Module-Imports...")
        
        # Core Module
        from core.neural_types import NeuralProcessingType
        from core.cognitive_architecture import CognitiveArchitecture
        print("✅ Core-Module importiert")
        
        # Agent Module
        from agents.agents import NeuralEthicalAgent
        print("✅ Agent-Module importiert")
        
        # Society Module  
        from society.neural_society import NeuralEthicalSociety
        print("✅ Society-Module importiert")
        
        # Scenarios Module
        from scenarios.scenarios import get_trolley_problem
        print("✅ Scenarios-Module importiert")
        
        print("\n🎯 Erstelle einfache Simulation...")
        
        # Erstelle eine kleine Simulation
        society = NeuralEthicalSociety()
        
        # Erstelle Agenten
        agent1 = NeuralEthicalAgent('demo_agent_1')
        agent2 = NeuralEthicalAgent('demo_agent_2')
        
        # Füge Agenten zur Gesellschaft hinzu
        society.add_agent(agent1)
        society.add_agent(agent2)
        
        print(f"✅ Gesellschaft erstellt mit {len(society.agents)} Agenten")
        
        # Teste Szenario
        scenario = get_trolley_problem()
        print(f"✅ Szenario geladen: {scenario.name}")
        
        # Lass Agenten entscheiden
        decision1 = agent1.make_decision(scenario)
        decision2 = agent2.make_decision(scenario)
        
        print(f"\n📊 Entscheidungsergebnisse:")
        print(f"Agent 1: {decision1.get('decision', 'N/A')} (Konfidenz: {decision1.get('confidence', 0):.2f})")
        print(f"Agent 2: {decision2.get('decision', 'N/A')} (Konfidenz: {decision2.get('confidence', 0):.2f})")
        
        print("\n🎉 Demo erfolgreich abgeschlossen!")
        print("📁 Die neue Ordnerstruktur funktioniert korrekt.")
        
    except ImportError as e:
        print(f"❌ Import-Fehler: {e}")
        print("Die Module müssen möglicherweise angepasst werden für die neue Struktur.")
        return False
        
    except Exception as e:
        print(f"❌ Allgemeiner Fehler: {e}")
        return False
        
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
