# Ethische Agenten-Simulation - Hauptmodul
"""
Ethische Agenten-Simulation
===========================

Ein Framework zur Simulation ethischer Entscheidungsfindung durch neuronale Agenten
mit verschiedenen kognitiven Architekturen und Persönlichkeitsmerkmalen.

Hauptmodule:
- core: Kernfunktionalitäten (neural_types, cognitive_architecture, beliefs, etc.)
- agents: Agent-Klassen und Verhalten
- society: Gesellschafts-Simulationen
- scenarios: Ethische Szenarien und Dilemmas
- analysis: Metriken, Validierung und Export
- visualization: Plotting und Dashboard-Funktionen
- web: Web-Interface und API
"""

__version__ = "1.0.0"
__author__ = "Ethische Agenten Simulation Team"

# Haupt-Imports für einfache Nutzung
from .core.neural_types import *
from .core.cognitive_architecture import *
from .core.beliefs import *
from .agents.agents import *
from .society.neural_society import *
from .scenarios.scenarios import *
