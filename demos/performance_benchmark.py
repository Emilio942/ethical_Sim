#!/usr/bin/env python3
"""
Performance Benchmark für Ethische Agenten-Simulation
====================================================

Misst die Performance verschiedener Komponenten des Systems
um Optimierungspotentiale zu identifizieren.
"""

import time
import psutil
import sys
import gc
from datetime import datetime

# Import der Simulation-Module
sys.path.append('.')
from neural_society import NeuralEthicalSociety
from agents import NeuralEthicalAgent
from scenarios import ScenarioGenerator
from metrics import MetricsCollector
from validation import ValidationSuite
from export_reporting import DataExporter

class PerformanceBenchmark:
    """Benchmark-Suite für Performance-Messungen"""
    
    def __init__(self):
        self.results = {}
        self.start_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
        
    def time_function(self, func, name, *args, **kwargs):
        """Misst Ausführungszeit einer Funktion"""
        print(f"🔄 Teste {name}...")
        
        # Memory vor Ausführung
        mem_before = psutil.virtual_memory().used / 1024 / 1024
        
        # Zeit messen
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Memory nach Ausführung
        mem_after = psutil.virtual_memory().used / 1024 / 1024
        memory_used = mem_after - mem_before
        
        execution_time = end_time - start_time
        
        self.results[name] = {
            'time': execution_time,
            'memory_mb': memory_used,
            'success': result is not None
        }
        
        print(f"  ⏱️  Zeit: {execution_time:.3f}s")
        print(f"  💾 Memory: {memory_used:.1f}MB")
        
        return result
        
    def benchmark_agent_creation(self):
        """Benchmark: Agent-Erstellung"""
        def create_agents():
            agents = []
            for i in range(10):
                agent = NeuralEthicalAgent(f"benchmark_agent_{i}")
                agents.append(agent)
            return agents
            
        return self.time_function(create_agents, "Agent Creation (10 agents)")
    
    def benchmark_society_creation(self):
        """Benchmark: Gesellschafts-Erstellung"""
        def create_society():
            society = NeuralEthicalSociety()
            for i in range(20):
                agent = NeuralEthicalAgent(f"society_agent_{i}")
                society.add_agent(agent)
            return society
            
        return self.time_function(create_society, "Society Creation (20 agents)")
    
    def benchmark_scenario_generation(self):
        """Benchmark: Szenario-Generierung"""
        def generate_scenarios():
            gen = ScenarioGenerator()
            scenarios = []
            for i in range(50):
                scenario = gen.generate_random_scenario()
                scenarios.append(scenario)
            return scenarios
            
        return self.time_function(generate_scenarios, "Scenario Generation (50 scenarios)")
    
    def benchmark_simulation(self, society, scenarios):
        """Benchmark: Simulation durchführen"""
        def run_simulation():
            # Nutze die verfügbare run_full_simulation Methode
            return society.run_full_simulation(num_steps=10)
            
        return self.time_function(run_simulation, "Simulation Execution (10 steps)")
    
    def benchmark_metrics_collection(self, society):
        """Benchmark: Metriken sammeln"""
        def collect_metrics():
            collector = MetricsCollector()
            return collector.collect_all_metrics(society)
            
        return self.time_function(collect_metrics, "Metrics Collection")
    
    def benchmark_validation(self, society):
        """Benchmark: Validierung durchführen"""
        def run_validation():
            validator = ValidationSuite()
            return validator.validate_society(society)
            
        return self.time_function(run_validation, "Validation Suite")
    
    def benchmark_export(self, society):
        """Benchmark: Datenexport"""
        def export_data():
            exporter = DataExporter(society)
            filename = f"benchmark_export_{int(time.time())}.json"
            exporter.export_json(filename)
            return filename
            
        return self.time_function(export_data, "Data Export (JSON)")
    
    def run_full_benchmark(self):
        """Führt vollständigen Benchmark durch"""
        print("🚀 PERFORMANCE BENCHMARK GESTARTET")
        print("=" * 50)
        print(f"📅 Zeitpunkt: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💻 System: {psutil.cpu_count()} CPUs, {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f}GB RAM")
        print()
        
        # Einzelne Benchmarks
        agents = self.benchmark_agent_creation()
        society = self.benchmark_society_creation()
        scenarios = self.benchmark_scenario_generation()
        
        print()
        print("🔄 Führe komplexere Tests durch...")
        
        self.benchmark_simulation(society, scenarios)
        self.benchmark_metrics_collection(society)
        self.benchmark_validation(society)
        self.benchmark_export(society)
        
        # Speicher-Cleanup
        gc.collect()
        
        print()
        self.print_summary()
        
    def print_summary(self):
        """Druckt Benchmark-Zusammenfassung"""
        print("📊 PERFORMANCE SUMMARY")
        print("=" * 50)
        
        total_time = sum(r['time'] for r in self.results.values())
        total_memory = sum(r['memory_mb'] for r in self.results.values())
        
        print(f"⏱️  Gesamtzeit: {total_time:.3f}s")
        print(f"💾 Gesamter Memory-Verbrauch: {total_memory:.1f}MB")
        print()
        
        print("📋 Detaillierte Ergebnisse:")
        for name, data in self.results.items():
            status = "✅" if data['success'] else "❌"
            print(f"  {status} {name}")
            print(f"     ⏱️  {data['time']:.3f}s | 💾 {data['memory_mb']:.1f}MB")
        
        print()
        print("🎯 Performance-Bewertung:")
        
        # Performance-Kategorien
        if total_time < 5:
            performance = "🚀 Excellent"
        elif total_time < 15:
            performance = "✅ Good"
        elif total_time < 30:
            performance = "⚠️ Acceptable"
        else:
            performance = "🐌 Needs Optimization"
            
        print(f"  Geschwindigkeit: {performance}")
        
        if total_memory < 100:
            memory_rating = "🚀 Excellent"
        elif total_memory < 500:
            memory_rating = "✅ Good"
        elif total_memory < 1000:
            memory_rating = "⚠️ Acceptable"
        else:
            memory_rating = "💾 Memory-Intensive"
            
        print(f"  Memory-Effizienz: {memory_rating}")
        
        print()
        print("💡 Optimierungsvorschläge:")
        
        # Spezifische Vorschläge basierend auf Ergebnissen
        slow_operations = [name for name, data in self.results.items() 
                          if data['time'] > 2.0]
        
        if slow_operations:
            print(f"  🔧 Langsame Operationen optimieren: {', '.join(slow_operations)}")
        
        memory_intensive = [name for name, data in self.results.items() 
                           if data['memory_mb'] > 50]
        
        if memory_intensive:
            print(f"  💾 Memory-intensive Operationen optimieren: {', '.join(memory_intensive)}")
        
        if not slow_operations and not memory_intensive:
            print("  🎉 System läuft bereits optimal!")

def main():
    """Hauptfunktion für Benchmark-Ausführung"""
    benchmark = PerformanceBenchmark()
    
    try:
        benchmark.run_full_benchmark()
        
        print()
        print("✅ Benchmark erfolgreich abgeschlossen!")
        print("📋 Ergebnisse können für Performance-Optimierung verwendet werden.")
        
    except Exception as e:
        print(f"❌ Benchmark-Fehler: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())
