import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from core.audio_processor import AudioLoader, AuditoryFeatures
from core.auditory_cortex import AuditoryCortex, AuditoryExpertModule, STRFGenerator
from core.logger import logger

def run_demo():
    print("=== Auditory Cortex System Identification Demo ===")
    
    # 1. Setup Paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/emodb'))
    anger_file = os.path.join(base_dir, 'anger', '03a01Wa.wav')
    sadness_file = os.path.join(base_dir, 'sadness', '03a02Ta.wav')
    
    if not os.path.exists(anger_file) or not os.path.exists(sadness_file):
        print("Error: EMO-DB files not found. Please run scripts/download_datasets.py first.")
        return

    # 2. Initialize Pipeline
    print("Initializing Auditory System...")
    loader = AudioLoader(target_sr=16000)
    features = AuditoryFeatures(sr=16000, num_channels=64)
    cortex = AuditoryCortex()
    strf_gen = STRFGenerator(num_channels=64, time_steps=20) # 200ms window

    # 3. Create Expert Modules (Hypothesis Testing)
    # Module A: "Anger Detector" (High Arousal)
    # Hypothesis: Anger has high energy, fast modulation, high frequencies.
    mod_anger = AuditoryExpertModule(module_id="Fast_High_Energy", best_freq_range=(32, 64))
    
    # Add filters to Module A (Fast temporal modulation, High spectral scale)
    for freq in [40, 50, 60]:
        # Fast rate (10-20Hz), Upward/Downward sweeps
        kernel = strf_gen.generate_gabor_strf(
            best_freq_idx=freq,
            spectral_width=5.0,
            temporal_mod_rate=15.0, # Fast
            spectral_mod_scale=0.5,
            direction=1
        )
        mod_anger.add_filter(kernel)
        
    # Manually add to cortex modules list
    cortex.modules.append(mod_anger)

    # Module B: "Sadness Detector" (Low Arousal)
    # Hypothesis: Sadness has low energy, slow modulation, low frequencies.
    mod_sadness = AuditoryExpertModule(module_id="Slow_Low_Energy", best_freq_range=(0, 32))
    
    # Add filters to Module B (Slow temporal modulation)
    for freq in [10, 20, 30]:
        # Slow rate (2-5Hz)
        kernel = strf_gen.generate_gabor_strf(
            best_freq_idx=freq,
            spectral_width=8.0,
            temporal_mod_rate=3.0, # Slow
            spectral_mod_scale=0.2,
            direction=1
        )
        mod_sadness.add_filter(kernel)
        
    cortex.modules.append(mod_sadness)

    # 4. Process Audio
    print("\nProcessing 'Anger' Sample...")
    audio_anger, _ = loader.load_audio(anger_file)
    coch_anger = features.compute_cochleogram(audio_anger)
    # Use process_input which iterates over modules
    resp_anger = cortex.process_input(coch_anger) 
    
    print("\nProcessing 'Sadness' Sample...")
    audio_sadness, _ = loader.load_audio(sadness_file)
    coch_sadness = features.compute_cochleogram(audio_sadness)
    resp_sadness = cortex.process_input(coch_sadness)

    # 5. Analyze Results
    # resp is a dict: {module_id: activations}
    
    act_anger_on_anger = np.mean(np.abs(resp_anger["Fast_High_Energy"]))
    act_sad_on_anger = np.mean(np.abs(resp_anger["Slow_Low_Energy"]))
    
    act_anger_on_sad = np.mean(np.abs(resp_sadness["Fast_High_Energy"]))
    act_sad_on_sad = np.mean(np.abs(resp_sadness["Slow_Low_Energy"]))
    
    print("\n=== Results (Mean Activation) ===")
    print(f"Stimulus: ANGER")
    print(f"  -> Fast/High Module (Anger): {act_anger_on_anger:.4f}")
    print(f"  -> Slow/Low Module (Sadness): {act_sad_on_anger:.4f}")
    
    print(f"\nStimulus: SADNESS")
    print(f"  -> Fast/High Module (Anger): {act_anger_on_sad:.4f}")
    print(f"  -> Slow/Low Module (Sadness): {act_sad_on_sad:.4f}")
    
    # 6. Visualization
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot Cochleograms
    axs[0, 0].imshow(coch_anger, aspect='auto', origin='lower', cmap='inferno')
    axs[0, 0].set_title("Cochleogram: Anger")
    axs[0, 1].imshow(coch_sadness, aspect='auto', origin='lower', cmap='inferno')
    axs[0, 1].set_title("Cochleogram: Sadness")
    
    # Plot Module Activations (Just taking the first filter of each module for simplicity)
    # Reshape flattened output back to time? 
    # The current implementation returns flattened arrays per filter.
    # Let's just plot the bar chart of mean activations for clarity.
    
    modules = ['Fast/High (Anger)', 'Slow/Low (Sadness)']
    x = np.arange(len(modules))
    width = 0.35
    
    vals_anger = [act_anger_on_anger, act_sad_on_anger]
    vals_sad = [act_anger_on_sad, act_sad_on_sad]
    
    axs[1, 0].bar(x - width/2, vals_anger, width, label='Anger Stimulus')
    axs[1, 0].bar(x + width/2, vals_sad, width, label='Sadness Stimulus')
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(modules)
    axs[1, 0].set_ylabel('Mean Activation')
    axs[1, 0].set_title('Module Response Comparison')
    axs[1, 0].legend()
    
    axs[1, 1].axis('off')
    axs[1, 1].text(0.1, 0.5, "See console for detailed stats.\n\nHypothesis:\nFast/High module should react stronger to Anger.\nSlow/Low module should react stronger to Sadness.", fontsize=12)
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), '../output/demo_auditory_response.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"\nPlot saved to {output_path}")

if __name__ == "__main__":
    run_demo()
