# 🚀 Installation & Setup Guide
# Ethische Agenten-Simulation

## 📋 Systemanforderungen

### Unterstützte Betriebssysteme:
- Linux (Ubuntu, Debian, CentOS, etc.)
- macOS 10.14+
- Windows 10/11

### Python-Version:
- Python 3.8 oder höher
- Empfohlen: Python 3.9-3.11

## 🛠️ Installationsschritte

### 1. Repository klonen oder herunterladen
```bash
# Option A: Git klonen (empfohlen)
git clone <repository-url>
cd ethik

# Option B: ZIP herunterladen und entpacken
# Laden Sie die ZIP-Datei herunter und entpacken Sie sie
```

### 2. Python Virtual Environment erstellen (empfohlen)
```bash
# Virtual Environment erstellen
python -m venv venv

# Aktivieren
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### 3. Dependencies installieren
```bash
# Alle benötigten Pakete installieren
pip install -r requirements.txt

# Bei Problemen: pip upgrade
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Installation verifizieren
```bash
# Basis-Demo ausführen
python run_demo.py

# Erweiterte Demo testen
python demo_enhanced_learning.py

# Umfassende Feature-Demo
python demo_metrics_validation_export.py
```

## 🌐 Web-Interface starten

### Flask-Server starten:
```bash
# Web-Interface starten
python web_interface.py

# Browser öffnen und navigieren zu:
# http://localhost:5000
```

### Features im Web-Interface:
- ✅ Interaktive Agent-Konfiguration
- ✅ Real-time Simulation mit Fortschrittsanzeige
- ✅ Live-Metriken Dashboard
- ✅ Multi-Format Export (JSON, CSV, HTML)
- ✅ Responsive Bootstrap UI

## 📚 Tutorial verwenden

### Jupyter Notebook starten:
```bash
# Jupyter installieren (falls nicht vorhanden)
pip install jupyter

# Notebook starten
jupyter notebook tutorial_ethical_agents.ipynb
```

### Tutorial-Inhalte:
1. 🤖 Agenten erstellen und konfigurieren
2. 🏛️ Gesellschaftssimulation durchführen
3. 📊 Metriken sammeln und analysieren
4. 🔍 Validierung und Qualitätssicherung
5. 📈 Interaktive Visualisierungen
6. 💾 Datenexport und Reporting

## 🔧 Konfiguration

### Agent-Konfiguration anpassen:
```python
# Beispiel: Eigene Agenten-Typen
from agents import NeuralEthicalAgent

agent = NeuralEthicalAgent(
    agent_id="my_agent",
    personality="utilitarian",  # utilitarian, deontological, virtue_ethics, balanced, pragmatic, idealistic
    ethical_framework="utilitarian"
)
```

### Szenario-Konfiguration:
```python
# Beispiel: Eigene Szenarien erstellen
from scenarios import EthicalScenario

scenario = EthicalScenario(
    title="Mein Szenario",
    description="Beschreibung des ethischen Dilemmas",
    context={"setting": "workplace", "stakeholders": ["employees", "customers"]},
    options=["Option A", "Option B", "Option C"],
    ethical_dimensions=["autonomy", "beneficence", "justice"]
)
```

## 📊 Ausgabeformate

### Unterstützte Export-Formate:
- **JSON**: Strukturierte Daten für weitere Programmierung
- **CSV**: Tabellendaten für Excel/Pandas-Analyse
- **SQLite**: Relationale Datenbank für komplexe Abfragen
- **HTML**: Interaktive Berichte für Präsentationen
- **Markdown**: Dokumentation und Reports

### Export-Beispiel:
```python
from export_reporting import ExportReporter

exporter = ExportReporter(society)
exporter.export_json("simulation_data.json")
exporter.export_csv("metrics_data.csv")
exporter.generate_html_report("simulation_report.html")
```

## 🐛 Troubleshooting

### Häufige Probleme und Lösungen:

#### ImportError: Module nicht gefunden
```bash
# Lösung: Virtual Environment aktivieren und Dependencies neu installieren
source venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

#### Matplotlib/Visualisierung-Probleme
```bash
# Linux: Fehlende GUI-Bibliotheken installieren
sudo apt-get install python3-tk

# macOS: XQuartz installieren
brew install --cask xquartz

# Windows: Normalerweise keine zusätzlichen Schritte nötig
```

#### Web-Interface startet nicht
```bash
# Port 5000 bereits belegt? Anderen Port verwenden:
python web_interface.py --port 8080

# Oder im Code ändern:
# app.run(debug=True, host='0.0.0.0', port=8080)
```

#### Jupyter Notebook Probleme
```bash
# Jupyter in Virtual Environment installieren
pip install jupyter ipykernel
python -m ipykernel install --user --name=venv

# Notebook starten
jupyter notebook
```

## 🔍 Performance-Optimierung

### Für große Simulationen (>50 Agenten):
```python
# Reduzierte Logging-Ausgabe
import logging
logging.getLogger().setLevel(logging.WARNING)

# Weniger Visualisierungen während Simulation
# Metriken-Sammlung am Ende statt kontinuierlich
```

### Speicher-Optimierung:
```python
# Periodisches Cleanup der Decision History
for agent in society.agents:
    if len(agent.decision_history) > 1000:
        agent.decision_history = agent.decision_history[-500:]
```

## 📞 Support

### Bei Problemen:
1. Überprüfen Sie die Systemanforderungen
2. Stellen Sie sicher, dass alle Dependencies installiert sind
3. Testen Sie die Demo-Skripte
4. Konsultieren Sie das Tutorial-Notebook
5. Prüfen Sie die Logs auf Fehlermeldungen

### Weiterführende Dokumentation:
- `README.md` - Projekt-Übersicht
- `aufgabenliste.md` - Detaillierte Implementierungsliste
- `tutorial_ethical_agents.ipynb` - Vollständiges Tutorial
- Demo-Skripte für spezifische Anwendungsfälle

## ✅ Erfolgreiche Installation verifizieren

Nach erfolgreicher Installation sollten Sie folgende Schritte durchführen können:
1. ✅ Demo-Skript ausführen (`python run_demo.py`)
2. ✅ Web-Interface starten (`python web_interface.py`)
3. ✅ Tutorial-Notebook öffnen (`jupyter notebook tutorial_ethical_agents.ipynb`)
4. ✅ Visualisierungen anzeigen (Dashboard mit Plots)
5. ✅ Export-Funktionen verwenden (JSON/CSV/HTML-Berichte)

**Herzlichen Glückwunsch! Sie können jetzt ethische Agenten-Simulationen durchführen! 🎉**
