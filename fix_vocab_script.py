import os
import json
import numpy as np
import config
from src.dataset import load_dataset, normalize_labels, filter_labels
from src.utils import save_vocabulary

def main():
    # 1. Load the exact same data as train.py
    print(f"Loading data from {config.DATA_DIR}...")
    X, y, label_names = load_dataset(config.DATA_DIR)
    
    # 2. Apply same normalization and filtering
    print("Applying normalization and filtering (config.MAIN_WORDS)...")
    X, y, label_names = normalize_labels(X, y, label_names)
    X, y, label_names = filter_labels(X, y, label_names, config.MAIN_WORDS)
    
    num_classes = len(label_names)
    print(f"Final 11 classes: {label_names}")
    
    # 3. Create the vocab dictionary exactly like train.py Step 9
    vocab = {
        "words": label_names,
        "word_to_index": {w: i for i, w in enumerate(label_names)},
        "index_to_word": {str(i): w for i, w in enumerate(label_names)},
        "use_velocity": True,  # Same as default in train.py
        "sequence_length": config.SEQUENCE_LENGTH,
        "num_features": config.NUM_FEATURES + 162, # hands (126) + pose (36) -- wait, 162 total
        "model_type": config.MODEL_TYPE,
    }
    
    # Actually, train.py calculates num_features after velocity concatenation
    # Hands-only: 126+36=162. Velocity adds another 162. Total 324.
    vocab["num_features"] = 324
    
    print(f"Saving vocabulary to {config.VOCAB_PATH}")
    save_vocabulary(vocab)
    print("Done!")

if __name__ == '__main__':
    main()
