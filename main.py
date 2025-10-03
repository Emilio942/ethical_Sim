#!/usr/bin/env python3
"""
Ethische Agenten-Simulation - Haupt-Demo
=========================================

Einfacher Einstiegspunkt für die Ethische Agenten-Simulation.
Führt eine grundlegende Demo mit verschiedenen Agenten und ethischen Szenarien durch.

Verwendung:
    python main.py                  # Basis-Demo
    python main.py --interactive    # Interaktive Demo
    python main.py --web           # Web-Interface starten
"""

import sys
import os
import argparse

# Füge src-Verzeichnis zum Python-Pfad hinzu
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def run_basic_demo():
    """Führt die Basis-Demo aus"""
    print("🚀 Lade Basis-Demo...")
    from demos.run_demo import main as demo_main

    demo_main()


def run_interactive_demo():
    """Führt die interaktive Demo aus"""
    print("🎮 Lade Interaktive Demo...")
    from demos.demo_enhanced_learning import main as interactive_main

    interactive_main()


def run_web_interface():
    """Startet das Web-Interface"""
    print("🌐 Starte Web-Interface...")
    from src.web.web_interface import app

    app.run(host="0.0.0.0", port=5000, debug=True)


def main():
    parser = argparse.ArgumentParser(description="Ethische Agenten-Simulation")
    parser.add_argument("--interactive", action="store_true", help="Starte interaktive Demo")
    parser.add_argument("--web", action="store_true", help="Starte Web-Interface")
    parser.add_argument("--version", action="version", version="1.0.0")

    args = parser.parse_args()

    print("🧠 Ethische Agenten-Simulation v1.0.0")
    print("=" * 50)

    if args.web:
        run_web_interface()
    elif args.interactive:
        run_interactive_demo()
    else:
        run_basic_demo()


if __name__ == "__main__":
    main()
