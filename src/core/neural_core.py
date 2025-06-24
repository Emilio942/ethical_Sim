"""
Kernmodul für neuronale ethische Agenten
========================================

Dieses Modul konsolidiert alle Importe und stellt eine einheitliche
Schnittstelle für die ethische Agenten-Simulation bereit.
"""

# Importiere alle notwendigen Komponenten aus den separaten Modulen
from neural_types import NeuralProcessingType
from cognitive_architecture import CognitiveArchitecture  
from beliefs import NeuralEthicalBelief
from scenarios import EthicalScenario, ScenarioGenerator

# Re-exportiere für einfache Verwendung
__all__ = [
    'NeuralProcessingType',
    'CognitiveArchitecture', 
    'NeuralEthicalBelief',
    'EthicalScenario',
    'ScenarioGenerator'
]

# Versionsinfo
__version__ = "1.0.0"
__author__ = "Ethische Agenten Projekt"

"""
Kernmodul für neuronale ethische Agenten
========================================

Dieses Modul konsolidiert alle Importe und stellt eine einheitliche
Schnittstelle für die ethische Agenten-Simulation bereit.
"""

# Importiere alle notwendigen Komponenten aus den separaten Modulen
from neural_types import NeuralProcessingType
from cognitive_architecture import CognitiveArchitecture  
from beliefs import NeuralEthicalBelief
from scenarios import EthicalScenario, ScenarioGenerator

# Re-exportiere für einfache Verwendung
__all__ = [
    'NeuralProcessingType',
    'CognitiveArchitecture', 
    'NeuralEthicalBelief',
    'EthicalScenario',
    'ScenarioGenerator'
]

# Versionsinfo
__version__ = "1.0.0"
__author__ = "Ethische Agenten Projekt"

def get_version():
    """Gibt die Version des Moduls zurück."""
    return __version__

def list_components():
    """Listet alle verfügbaren Komponenten auf."""
    return __all__

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
        print(f"✅ ScenarioGenerator funktioniert: {len(generator.get_available_templates())} Templates")
        
        print("🎉 Alle Komponenten erfolgreich geladen!")
        
    except Exception as e:
        print(f"❌ Fehler beim Testen der Komponenten: {e}")
        print(f"❌ Fehler beim Testen der Komponenten: {e}")