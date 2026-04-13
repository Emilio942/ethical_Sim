
import numpy as np
import matplotlib.pyplot as plt
from core.cognitive_architecture import QuantumZenoArchitecture

def run_zeno_experiment():
    print("🧠 Starte Quantum Zeno Effekt Validierung...")
    
    # Szenario: 3 moralische Dimensionen (8 Zustände)
    # Wir verbieten Zustand 5 (Index 5), den System 1 intuitiv bevorzugt
    context = {"forbidden_indices": [5]}
    inputs = {f"option_{i}": 0.1 for i in range(8)}
    
    frequencies = [0.0, 0.2, 0.5, 0.9] # nu-Werte (Messfrequenz)
    results_for_forbidden = []

    for nu in frequencies:
        # Initialisiere Architektur mit spezifischer Frequenz
        arch = QuantumZenoArchitecture(num_moral_dimensions=3, measurement_frequency=nu)
        
        # Simuliere 50 Entscheidungsschritte
        forbidden_prob_sum = 0
        for _ in range(50):
            processed = arch.process_information(inputs, context)
            forbidden_prob_sum += processed["option_5"]
            
        avg_forbidden_prob = forbidden_prob_sum / 50
        results_for_forbidden.append(avg_forbidden_prob)
        print(f"  - Frequenz \u03bd={nu:.1f}: Durchschnittliche Wahrscheinlichkeit für verbotene Option: {avg_forbidden_prob:.4f}")

    # Visualisierung
    plt.figure(figsize=(10, 6))
    plt.bar([str(f) for f in frequencies], results_for_forbidden, color='teal')
    plt.xlabel('Messfrequenz \u03bd (System 2 Intensität)')
    plt.ylabel('Wahrscheinlichkeit der verbotenen Option')
    plt.title('Quantum Zeno Effekt: Unterdrückung von System 1 durch System 2')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    output_path = "archive/quantum_zeno_results.png"
    plt.savefig(output_path)
    print(f"✅ Validierung abgeschlossen. Grafik gespeichert unter: {output_path}")

if __name__ == "__main__":
    run_zeno_experiment()
