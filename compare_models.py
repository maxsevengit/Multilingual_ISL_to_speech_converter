import os
import time
import json
import numpy as np
import pandas as pd
import config
from src.dataset import load_dataset, normalize_labels, filter_labels, load_dataset_npz, save_dataset_npz
from src.feature_engineer import build_feature_vector, normalize_hands_sequence
from src.model import build_model, train_model
from sklearn.model_selection import train_test_split as tts

def main():
    # 1. SETUP
    print("\n" + "="*60)
    print("  ISL 5-Model Accuracy Comparison Pipeline")
    print("="*60)
    
    comparisons_dir = os.path.join(config.BASE_DIR, "models", "comparison")
    os.makedirs(comparisons_dir, exist_ok=True)
    
    # 2. LOAD DATA (One-time load for all models)
    dataset_npz_path = os.path.join(config.DATA_DIR, "dataset.npz")
    print(f"\n[STEP 1] Loading and preprocessing data...")
    X, y, label_names = load_dataset(config.DATA_DIR)
    X, y, label_names = normalize_labels(X, y, label_names)
    X, y, label_names = filter_labels(X, y, label_names, config.MAIN_WORDS)
    
    # Normalize and Pre-split
    X = np.array([normalize_hands_sequence(seq) for seq in X], dtype=np.float32)
    X_train_orig, X_val, y_train_orig, y_val = tts(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Build feature vectors (including velocity)
    X_train = np.array([build_feature_vector(seq) for seq in X_train_orig], dtype=np.float32)
    X_val = np.array([build_feature_vector(seq) for seq in X_val], dtype=np.float32)
    
    num_features = X_train.shape[2]
    num_classes = len(label_names)
    seq_length = X_train.shape[1]
    
    results = []
    model_types = ["lstm", "gru", "tcn", "transformer", "mlp"]
    
    from sklearn.metrics import f1_score
    
    # 3. TRAINING LOOP
    for m_type in model_types:
        print(f"\n\n>>> BEGIN TRAINING: {m_type.upper()} <<<")
        print("-" * 40)
        
        # Override config.MODEL_PATH for this model's best weights
        config.MODEL_PATH = os.path.join(comparisons_dir, f"comp_{m_type}.keras")
        config.EPOCHS = 100 # Deep Dive
        
        start_time = time.time()
        
        model = build_model(num_features, num_classes, seq_length=seq_length, model_type=m_type)
        model, history = train_model(X_train, y_train_orig, num_classes, model, 
                                     X_val_override=X_val, y_val_override=y_val)
        
        train_duration = time.time() - start_time
        best_acc = max(history.history['val_accuracy'])
        params = model.count_params()
        
        # Measure Inference Latency
        print(f"[INFO] Measuring inference latency for {m_type.upper()}...")
        inf_start = time.time()
        # Run inference on entire validation set 10 times for stable average
        for _ in range(10):
            preds = model.predict(X_val, verbose=0)
        inf_duration = (time.time() - inf_start) / (len(X_val) * 10) # Seconds per sample
        
        # Calculate F1 Score on Validation set
        y_pred = np.argmax(preds, axis=1)
        f1 = f1_score(y_val, y_pred, average='weighted')
        
        results.append({
            "Model": m_type.upper(),
            "Accuracy": f"{best_acc*100:.2f}%",
            "F1-Score": f"{f1*100:.2f}%",
            "Inf. Latency": f"{inf_duration*1000:.2f}ms",
            "Params": f"{params:,}",
            "Train Time": f"{train_duration:.1f}s"
        })
        
        print(f"\n[DONE] {m_type.upper()} | Acc: {best_acc*100:.1f}% | Inf: {inf_duration*1000:.2f}ms")

    
    # 4. FINAL REPORT
    print("\n" + "="*60)
    print("  FINAL COMPARISON RESULTS")
    print("="*60)
    df = pd.DataFrame(results)
    print(df.to_markdown(index=False))
    
    report_path = os.path.join(config.BASE_DIR, "models", "comparison_report.md")
    with open(report_path, "w") as f:
        f.write("# ISL Architecture Comparison\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n*Note: All models trained for 25 epochs on 11 core ISL words.*")
    
    print(f"\n[REPORT] Saved to {report_path}")

if __name__ == "__main__":
    main()
