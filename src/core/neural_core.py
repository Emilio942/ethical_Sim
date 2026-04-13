"""
Kernmodul für neuronale ethische Agenten
========================================

Dieses Modul konsolidiert alle Importe und stellt eine einheitliche
Schnittstelle für die ethische Agenten-Simulation bereit.
"""

# Importiere alle notwendigen Komponenten aus den separaten Modulen
from core.neural_types import NeuralProcessingType
from core.cognitive_architecture import CognitiveArchitecture
from core.beliefs import NeuralEthicalBelief
from scenarios.scenarios import EthicalScenario, ScenarioGenerator

# Re-exportiere für einfache Verwendung
__all__ = [
    "NeuralProcessingType",
    "CognitiveArchitecture",
    "NeuralEthicalBelief",
    "EthicalScenario",
    "ScenarioGenerator",
]

# Versionsinfo
__version__ = "1.0.0"
__author__ = "Ethische Agenten Projekt"

# Versionsinfo
__version__ = "1.0.0"
__author__ = "Ethische Agenten Projekt"


def get_version():
    """Gibt die Version des Moduls zurück."""
    return __version__


def list_components():
    """Listet alle verfügbaren Komponenten auf."""
    return __all__


# --- MATHEMATISCHE ANFRAGE: EICHFELDTHEORIE DER ETHIK ---
# Frage: Könnten soziale Normen als ein Eichfeld auf dem sozialen Manifold 
# modelliert werden? Wenn wir eine Lagrange-Dichte für den ethischen Konsens 
# definieren, würde die moralische Kraft zwischen Agenten durch 
# Informations-Überträger vermittelt werden?

# --- MATHEMATISCHE ANFRAGE: LICHTKEGEL-THEORIE ---
# Frage: Wenn wir soziale Ereignisse in einem sozialen Minkowsk-Raum darstellen, 
# könnten moralische Urteile als Richtungs-Vektoren auf dem Lichtkegel 
# beschrieben werden?

# --- MATHEMATISCHE ANFRAGE: ANALYTISCHE ZAHLENTHEORIE ---
# Frage: Können wir eine ethische Zeta-Funktion definieren, deren Nullstellen 
# die Verteilung fundamentaler moralischer Axiome kodieren?

# --- MATHEMATISCHE ANFRAGE: STOCHASTISCHE ANALYSIS ---
# Frage: Kann das Ito-Lemma genutzt werden, um die moralische Volatilität 
# einer Bevölkerung unter externen Schocks abzuleiten?

# --- MATHEMATICAL INQUIRY 35: CHERN-SIMONS THEORY ---
# Question: Can the 'Topological Phases' of a society be described 
# by a Chern-Simons Lagrangian? Would the Linking Number of social 
# events provide a topological invariant for historical justice?


# Test-Funktionen für das Modul
if __name__ == "__main__":
    print(f"🧠 Neural Core Module v{get_version()}")
    print("Verfügbare Komponenten:")
    for component in list_components():
        print(f"  - {component}")

    # Teste die Imports
    try:
        # Teste Verarbeitungstypen
        processing_type = NeuralProcessingType.SYSTEMATIC
        print(f"✅ NeuralProcessingType funktioniert: {processing_type}")

        # Teste kognitive Architektur
        arch = CognitiveArchitecture()
        print(f"✅ CognitiveArchitecture funktioniert: {arch.primary_processing}")

        # Teste Überzeugungen
        belief = NeuralEthicalBelief("Test", "Kategorie")
        print(f"✅ NeuralEthicalBelief funktioniert: {belief.name}")

        # Teste Szenarien
        generator = ScenarioGenerator()
        print(
            f"✅ ScenarioGenerator funktioniert: {len(generator.get_available_templates())} Templates"
        )

        print("🎉 Alle Komponenten erfolgreich geladen!")

    except Exception as e:
        print(f"❌ Fehler beim Testen der Komponenten: {e}")
        print(f"❌ Fehler beim Testen der Komponenten: {e}")
