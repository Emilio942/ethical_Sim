import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from core.audio_processor import AuditoryFeatures, AudioLoader
from core.auditory_cortex import AuditoryCortex, STRFGenerator

def visualize_agent_hearing():
    print("=== Visualizing Agent Hearing ===")
    
    # 1. Setup
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/emodb'))
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../output/visualizations'))
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Models
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))
    clf = joblib.load(os.path.join(model_dir, 'emotion_mlp_classifier.pkl'))
    scaler = joblib.load(os.path.join(model_dir, 'emotion_scaler.pkl'))
    labels = joblib.load(os.path.join(model_dir, 'emotion_labels.pkl'))
    
    # Initialize Cortex
    features = AuditoryFeatures(sr=16000, num_channels=64)
    loader = AudioLoader(target_sr=16000)
    cortex = AuditoryCortex(num_modules=16, num_channels=64)
    strf_gen = STRFGenerator(num_channels=64, time_steps=20)
    cortex.initialize_filters(strf_gen)
    
    # Pick an Anger file (usually high energy)
    anger_dir = os.path.join(base_dir, 'anger')
    if os.path.exists(anger_dir):
        file_path = os.path.join(anger_dir, random.choice(os.listdir(anger_dir)))
    else:
        # Fallback
        all_wavs = []
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith('.wav'):
                    all_wavs.append(os.path.join(root, file))
        file_path = random.choice(all_wavs)
        
    print(f"Visualizing: {os.path.basename(file_path)}")
    
    # 2. Process
    audio, sr = loader.load_audio(file_path)
    cochleogram = features.compute_cochleogram(audio)
    module_outputs = cortex.process_input(cochleogram)
    
    # Feature Extraction for Prediction
    feature_list = []
    min_time = float('inf')
    for mod_id in sorted(module_outputs.keys()):
        out = module_outputs[mod_id]
        feature_list.append(out)
        if out.shape[1] < min_time:
            min_time = out.shape[1]
    stacked = np.vstack([f[:, :min_time] for f in feature_list])
    mean_pool = np.mean(stacked, axis=1)
    std_pool = np.std(stacked, axis=1)
    max_pool = np.max(stacked, axis=1)
    feats = np.concatenate([mean_pool, std_pool, max_pool])
    
    # Predict
    feats_scaled = scaler.transform([feats])
    probs = clf.predict_proba(feats_scaled)[0]
    pred_idx = np.argmax(probs)
    prediction = labels[pred_idx]
    
    # 3. Plot
    fig = plt.figure(figsize=(15, 10))
    plt.suptitle(f"Agent Auditory Perception\nFile: {os.path.basename(file_path)} | Prediction: {prediction.upper()} ({probs[pred_idx]:.2f})", fontsize=16)
    
    # A. Waveform
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(np.linspace(0, len(audio)/sr, len(audio)), audio, color='black', alpha=0.7)
    ax1.set_title("1. Raw Audio Signal (The Ear)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    
    # B. Cochleogram
    ax2 = plt.subplot(2, 2, 2)
    im2 = ax2.imshow(cochleogram, aspect='auto', origin='lower', cmap='inferno')
    ax2.set_title("2. Cochleogram (Frequency Analysis)")
    ax2.set_ylabel("Frequency Channel (ERB)")
    plt.colorbar(im2, ax=ax2)
    
    # C. Neural Activity (Subset of Modules)
    ax3 = plt.subplot(2, 2, 3)
    # Stack first 4 modules for visualization
    # Keys are strings: "A1_Module_0", "A1_Module_1", etc.
    vis_stack_list = []
    for i in range(4):
        key = f"A1_Module_{i}"
        if key in module_outputs:
            vis_stack_list.append(module_outputs[key])
            
    if vis_stack_list:
        vis_stack = np.vstack(vis_stack_list)
        im3 = ax3.imshow(vis_stack, aspect='auto', origin='lower', cmap='viridis')
        ax3.set_title("3. Neural Activity (Auditory Cortex - First 4 Modules)")
        ax3.set_ylabel("Neuron ID")
        ax3.set_xlabel("Time Frame")
        plt.colorbar(im3, ax=ax3)
    else:
        ax3.text(0.5, 0.5, "No Module Data", ha='center')
    
    # D. Decision Probabilities
    ax4 = plt.subplot(2, 2, 4)
    bars = ax4.bar(labels, probs, color='skyblue')
    # Highlight winner
    bars[pred_idx].set_color('crimson')
    ax4.set_title("4. Agent Decision (Probabilities)")
    ax4.set_ylim(0, 1)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'agent_hearing_vis.png')
    plt.savefig(save_path)
    print(f"Visualization saved to: {save_path}")

if __name__ == "__main__":
    visualize_agent_hearing()
