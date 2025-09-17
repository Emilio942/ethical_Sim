"""
Kernmodul für neuronale ethische Agenten
========================================

Dieses Modul konsolidiert alle Importe und stellt eine einheitliche
Schnittstelle für die ethische Agenten-Simulation bereit.
"""

# Importiere nur die Kern-Komponenten (ohne scenarios um Zirkularabhängigkeiten zu vermeiden)
try:
    # Try relative imports first (when used as a package)
    from .neural_types import NeuralProcessingType
    from .cognitive_architecture import CognitiveArchitecture  
    from .beliefs import NeuralEthicalBelief
except ImportError:
    try:
        # Try absolute imports from the src package
        from core.neural_types import NeuralProcessingType
        from core.cognitive_architecture import CognitiveArchitecture  
        from core.beliefs import NeuralEthicalBelief
    except ImportError:
        # Fallback to direct imports
        from neural_types import NeuralProcessingType
        from cognitive_architecture import CognitiveArchitecture  
        from beliefs import NeuralEthicalBelief

# Re-exportiere für einfache Verwendung
__all__ = [
    'NeuralProcessingType',
    'CognitiveArchitecture', 
    'NeuralEthicalBelief'
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
        
        print("🎉 Kern-Komponenten erfolgreich geladen!")
        
    except Exception as e:
        print(f"❌ Fehler beim Testen der Komponenten: {e}")