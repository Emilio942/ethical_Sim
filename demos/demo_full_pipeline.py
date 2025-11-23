import os
import sys
import numpy as np
import random
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from core.dataset_pipeline import DatasetPipeline
from core.audio_processor import AuditoryFeatures
from core.auditory_cortex import AuditoryCortex, STRFGenerator
from core.logger import logger

def extract_features(cortex, features, loader, file_path):
    """Helper to extract fixed-size features from variable length audio."""
    try:
        # A. Perception (Ear -> Cortex)
        audio, _ = loader.load_audio(file_path)
        cochleogram = features.compute_cochleogram(audio)
        
        # B. Processing (Cortex)
        module_outputs = cortex.process_input(cochleogram)
        
        feature_list = []
        min_time = float('inf')
        
        for mod_id in sorted(module_outputs.keys()):
            out = module_outputs[mod_id]
            feature_list.append(out)
            if out.shape[1] < min_time:
                min_time = out.shape[1]
        
        # Crop and Stack: (total_filters, min_time)
        stacked = np.vstack([f[:, :min_time] for f in feature_list])
        
        # Global Pooling (Mean + Std) to get fixed size vector
        # Shape: (total_filters * 2,)
        mean_pool = np.mean(stacked, axis=1)
        std_pool = np.std(stacked, axis=1)
        max_pool = np.max(stacked, axis=1)
        
        return np.concatenate([mean_pool, std_pool, max_pool])
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return None

def run_full_pipeline_demo():
    print("=== Final Integration Demo: Full Auditory Pipeline (with MLP Readout) ===")
    
    # 1. Setup Data
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/emodb'))
    if not os.path.exists(base_dir):
        print("Error: EMO-DB dataset not found. Run scripts/download_datasets.py first.")
        return

    print("Loading Dataset...")
    pipeline = DatasetPipeline(dataset_path=base_dir, target_sr=16000)
    pipeline.scan_dataset()
    
    # Map string labels to indices
    unique_labels = sorted(list(set(pipeline.labels.values())))
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    
    print(f"Classes: {unique_labels}")
    
    # Split Train/Test
    all_files = pipeline.file_list
    random.shuffle(all_files)
    split_idx = int(len(all_files) * 0.8)
    train_files = all_files[:split_idx]
    test_files = all_files[split_idx:]
    
    print(f"Train samples: {len(train_files)}, Test samples: {len(test_files)}")
    
    # 2. Initialize System
    print("Initializing Auditory Cortex...")
    features = AuditoryFeatures(sr=16000, num_channels=64)
    
    # Cortex with fixed STRFs (Biologically Plausible Reservoir)
    cortex = AuditoryCortex(num_modules=16, num_channels=64) # Increased modules for more diversity
    strf_gen = STRFGenerator(num_channels=64, time_steps=20)
    cortex.initialize_filters(strf_gen)
    
    # 3. Feature Extraction Loop
    print("\n=== Extracting Features from Auditory Cortex ===")
    
    X_train = []
    y_train = []
    
    for file_path in tqdm(train_files, desc="Extracting Train"):
        feat = extract_features(cortex, features, pipeline.loader, file_path)
        if feat is not None:
            X_train.append(feat)
            y_train.append(label_to_idx[pipeline.labels[file_path]])
            
    X_test = []
    y_test = []
    
    for file_path in tqdm(test_files, desc="Extracting Test"):
        feat = extract_features(cortex, features, pipeline.loader, file_path)
        if feat is not None:
            X_test.append(feat)
            y_test.append(label_to_idx[pipeline.labels[file_path]])
            
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    print(f"Feature Shape: {X_train.shape}")
    
    # 4. Training Readout (MLP)
    print("\n=== Training Readout Layer (MLP) ===")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # MLP Classifier (Simulates the 'Concept' layer learning from Cortex features)
    clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42, learning_rate_init=0.001)
    clf.fit(X_train_scaled, y_train)
    
    # 5. Evaluation
    print("\n=== Evaluation Phase ===")
    y_pred = clf.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nFinal Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=unique_labels, zero_division=0))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 6. Save Model for Agent Integration
    import joblib
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"\nSaving models to {model_dir}...")
    joblib.dump(clf, os.path.join(model_dir, 'emotion_mlp_classifier.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'emotion_scaler.pkl'))
    joblib.dump(unique_labels, os.path.join(model_dir, 'emotion_labels.pkl'))
    print("Models saved successfully.")

if __name__ == "__main__":
    run_full_pipeline_demo()
