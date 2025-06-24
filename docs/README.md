# 🧠 Ethische Agenten Simulation

Eine Python-Simulation zur Untersuchung ethischer Entscheidungsfindung durch KI-Agenten mit verschiedenen kognitiven Architekturen und Persönlichkeitsmerkmalen.

## 🎯 Überblick

Dieses Projekt simuliert ethische Entscheidungsprozesse durch neuronale Agenten, die verschiedene Denkstile (systematisch, intuitiv, emotional, etc.) und Persönlichkeitsmerkmale haben. Die Agenten treffen Entscheidungen in klassischen ethischen Dilemmata wie dem Trolley-Problem.

## ✨ Features

- **🤖 Diverse Agent-Persönlichkeiten**: Big Five Persönlichkeitsmodell
- **🧠 Kognitive Verarbeitung**: 6 verschiedene neuronale Verarbeitungstypen
- **🎭 Ethische Szenarien**: Trolley-Problem, Autonome Fahrzeuge, Privatsphäre vs. Sicherheit, etc.
- **🌐 Soziale Netzwerke**: Agent-Interaktionen und Gruppenbildung
- **📊 Umfassende Visualisierung**: Interaktive Plots und Dashboards
- **🔄 Flexible Simulation**: Batch-Läufe, interaktive Modi, Anpassbare Parameter
- **🧪 Erweiterte Lernmechanismen**: 
  - Reinforcement Learning mit Exploration/Exploitation
  - Multi-Criteria Decision Making mit Trade-off Analysis
  - Uncertainty und Ambiguity Handling
  - Advanced Social Learning mit Trust Networks
  - Temporal Belief Dynamics mit Momentum-Effekten
  - Metacognitive Monitoring und Strategy Switching

## 🚀 Schnellstart

### Installation

```bash
# Repository klonen
git clone <repository-url>
cd ethik

# Dependencies installieren
pip install -r requirements.txt
```

### Demo ausführen

```bash
# Interaktive Demo starten
python run_demo.py

# Erweiterte Lernmechanismen Demo
python demo_enhanced_learning.py

# Visualisierungs-Test
python test_visualization.py
```

## 📁 Projektstruktur

```
ethik/
├── 📋 aufgabenliste.md          # Projekt-Roadmap und Status
├── 🤖 agents.py                 # Agent-Implementierung mit Persönlichkeiten
├── 🧠 beliefs.py                # Überzeugungssystem mit neuronaler Modellierung
├── ⚙️ cognitive_architecture.py  # Kognitive Verarbeitungsmodelle
├── 🎭 scenarios.py              # Ethische Dilemmata und Szenario-Generator
├── 🏛️ neural_society.py         # Gesellschafts-Simulation und Netzwerke
├── 🧩 neural_types.py           # Definitionen für Verarbeitungstypen
├── 🌐 neural_core.py            # Zentrale Import-Sammlung
├── 🎮 run_demo.py               # Hauptdemo mit verschiedenen Modi
├── 🔧 utils.py                  # Utility-Funktionen und Validierung
├── 📊 visualization.py          # Umfassende Visualisierungssuite
├── 🧪 test_visualization.py     # Tests für Visualisierung
├── 📦 requirements.txt          # Python-Dependencies
└── 📁 output/                   # Generierte Plots und Berichte
```

## 🎮 Demo-Modi

### 1. Basis-Demo
```bash
python run_demo.py
# Wähle Option 1
```
- Automatische Erstellung von 5 Agenten mit verschiedenen Persönlichkeiten
- Verarbeitung klassischer ethischer Dilemmata
- Anzeige von Entscheidungsmustern

### 2. Interaktive Demo
```bash
python run_demo.py
# Wähle Option 2
```
- Auswahl spezifischer Szenarien
- Anpassbare Agent-Anzahl
- Detaillierte Analyse einzelner Entscheidungen

### 3. Batch-Analyse
```bash
python run_demo.py
# Wähle Option 3
```
- Verarbeitung aller verfügbaren Szenarien
- Umfassende Gesellschaftsanalyse
- Statistische Auswertungen

### 4. Visualisierungs-Demo
```bash
python run_demo.py
# Wähle Option 4
```
- **Persönlichkeits-Plots**: Radar-Charts und Histogramme
- **Netzwerk-Visualisierung**: Soziale Verbindungen und Cluster
- **Entscheidungsanalyse**: Konfidenz, Verarbeitungstyp-Korrelationen
- **Simulation Dashboard**: Übersichtliche Gesamtdarstellung

## 📊 Visualisierungen

Das System erstellt automatisch verschiedene Grafiken:

### 🎭 Agent-Persönlichkeiten
- **Radar-Chart**: Durchschnittliche Persönlichkeitsverteilung
- **Histogramme**: Verteilung einzelner Traits (Big Five)
- **Farb-kodiert**: Intuitive Unterscheidung der Eigenschaften

### 🌐 Soziales Netzwerk
- **Graph-Darstellung**: Agent-Verbindungen und Gruppen
- **Farb-kodierung**: Nach kognitiven Verarbeitungstypen
- **Statistiken**: Dichte, Zentralität, Clustering-Koeffizienten

### 🎯 Entscheidungsanalyse
- **Pie-Charts**: Verteilung der gewählten Optionen
- **Box-Plots**: Konfidenz nach Verarbeitungstyp
- **Heatmaps**: Entscheidungsmatrix nach Agent-Typ
- **Szenario-Details**: Komplexität, Zeitdruck, emotionale Valenz

### 📈 Dashboard
- **Multi-Panel-Ansicht**: Alle wichtigen Metriken auf einen Blick
- **Echtzeit-Updates**: Dynamische Anpassung an Simulationsdaten
- **Export-Funktionen**: PNG-Format für Berichte und Präsentationen

## 🧠 Kognitive Verarbeitungstypen

| Typ | Beschreibung | Charakteristika |
|-----|--------------|----------------|
| **Systematisch** | Analytisch, schrittweise | Logik-orientiert, reduzierte Verzerrungen |
| **Intuitiv** | Schnell, ganzheitlich | Bauchgefühl, Verfügbarkeitsheuristiken |
| **Assoziativ** | Netzwerkartig | Spreading Activation, Konzept-Verbindungen |
| **Analogisch** | Metaphern-basiert | Strukturelle Ähnlichkeiten, Pattern-Matching |
| **Emotional** | Gefühls-gesteuert | Somatische Marker, Negativitäts-Bias |
| **Narrativ** | Story-orientiert | Kohärenz-fokussiert, Geschichten-Logik |

## 🎭 Verfügbare Szenarien

### Klassische Dilemmata
- **🚂 Trolley-Problem**: Utilitarismus vs. Deontologie
- **🚗 Autonome Fahrzeuge**: Programmierte Ethik in Unfallsituationen
- **🔒 Privatsphäre vs. Sicherheit**: Überwachung zur Terrorbekämpfung
- **🌍 Umwelt-Dilemma**: Wirtschaft vs. Klimaschutz

### Szenario-Generator
- **Automatische Erstellung**: Zufällige ethische Konflikte
- **Anpassbare Parameter**: Komplexität, Zeitdruck, Unsicherheit
- **Template-System**: Einfache Erweiterung um neue Dilemmata

## 🔧 Technische Details

### Dependencies
- **numpy**: Numerische Berechnungen und Statistiken
- **matplotlib**: Basis-Visualisierungen und Plots
- **seaborn**: Erweiterte statistische Grafiken
- **networkx**: Soziale Netzwerk-Analyse
- **scipy**: Wissenschaftliche Berechnungen

### Architektur
- **Modularer Aufbau**: Klare Trennung der Komponenten
- **Erweiterbare Basis**: Einfache Integration neuer Agenten-Typen
- **Datenvalidierung**: Robuste Eingabe-Überprüfung
- **Export-Funktionen**: JSON/CSV für weitere Analyse

## 📈 Aktuelle Metriken

Das Projekt hat erfolgreich alle kritischen Meilensteine erreicht:

✅ **P0 (Kritisch)**: Alle Kern-Module implementiert und funktionsfähig  
✅ **P1 (Hoch)**: Vollständige Visualisierung und Demo-System  
🔄 **P2 (Mittel)**: Erweiterte Agent-Logik in Entwicklung  

## 🚧 Roadmap

### Kurzfristig (nächste Wochen)
- **Erweiterte Entscheidungslogik**: Echte ethische Reasoning-Algorithmen
- **Lernmechanismen**: Agent-Anpassung durch Erfahrung
- **Mehr Szenarien**: Medizinethik, KI-Ethik, Klimawandel

### Mittelfristig (nächste Monate)
- **Web-Interface**: Browser-basierte Simulation
- **Machine Learning**: Echte neuronale Netzwerke für Entscheidungen
- **Kulturelle Dimension**: Cross-kulturelle ethische Unterschiede

### Langfristig (Forschungsziele)
- **Publikationsreife Ergebnisse**: Wissenschaftliche Auswertungen
- **Open-Source Community**: Beiträge und Erweiterungen
- **Reale Anwendungen**: Integration in ethische KI-Systeme

## 🤝 Beitragen

Dieses Projekt ist offen für Beiträge! Bereiche für Verbesserungen:

- **Neue ethische Szenarien**: Erweitere `scenarios.py`
- **Verbesserte Visualisierungen**: Zusätzliche Plot-Typen
- **Performance-Optimierung**: Parallelisierung großer Simulationen
- **Dokumentation**: Tutorials und Anwendungsbeispiele

## 📝 Lizenz

Dieses Projekt steht unter einer offenen Lizenz zur Förderung ethischer KI-Forschung.

---

**Erstellt am**: 19. Juni 2025  
**Status**: Voll funktionsfähig ✅  
**Letzte Aktualisierung**: Visualisierungssystem implementiert  

*Für Fragen oder Anregungen, siehe die Aufgabenliste (`aufgabenliste.md`) für Details zu geplanten Verbesserungen.*
