import os
import numpy as np
import config

# Typical relaxed upper body pose (normalized coordinates)
IDLE_POSE = np.array([
    # Example values relative to shoulder center
    [-0.5, 0.0, 0.0],  # left_shoulder (11)
    [ 0.5, 0.0, 0.0],  # right_shoulder (12)
    [-0.6, 1.5, 0.0],  # left_elbow (13)
    [ 0.6, 1.5, 0.0],  # right_elbow (14)
    [-0.6, 2.5, 0.0],  # left_wrist (15)
    [ 0.6, 2.5, 0.0],  # right_wrist (16)
    [-0.3, 3.5, 0.0],  # left_hip (23)
    [ 0.3, 3.5, 0.0],  # right_hip (24)
    [-0.3, 5.0, 0.0],  # left_knee (25)
    [ 0.3, 5.0, 0.0],  # right_knee (26)
    [-0.3, 6.5, 0.0],  # left_ankle (27)
    [ 0.3, 6.5, 0.0],  # right_ankle (28)
]).flatten()

# Empty hands (shape: 63)
EMPTY_HAND = np.zeros(config.SINGLE_HAND_FEATURES)

import argparse

def generate_idle_frames(num_samples=100, seq_length=30, data_dir=config.DATA_DIR):
    idle_dir = os.path.join(data_dir, "IDLE")
    os.makedirs(idle_dir, exist_ok=True)
    
    print(f"Generating {num_samples} IDLE sequences in {idle_dir}...")
    
    for i in range(num_samples):
        sequence = []
        # Add a tiny bit of random noise to the base pose to simulate natural body sway
        pose_noise = np.random.normal(0, 0.02, size=config.POSE_FEATURES)
        base_pose = IDLE_POSE + pose_noise
        
        for _ in range(seq_length):
            # 50% chance the hands are completely "missing" (out of frame), 50% chance they are resting at the sides
            if np.random.rand() > 0.5:
                lh = EMPTY_HAND
                rh = EMPTY_HAND
            else:
                # Random noise near the hips for hands
                lh = np.random.normal(-0.6, 0.1, size=config.SINGLE_HAND_FEATURES)
                rh = np.random.normal(0.6, 0.1, size=config.SINGLE_HAND_FEATURES)
            
            # Add slight frame-to-frame noise to pose
            frame_pose = base_pose + np.random.normal(0, 0.005, size=config.POSE_FEATURES)
            
            frame_features = np.concatenate([lh, rh, frame_pose])
            sequence.append(frame_features)
            
        sequence_np = np.array(sequence, dtype=np.float32)
        save_path = os.path.join(idle_dir, f"idle_seq_{i:04d}.npy")
        np.save(save_path, sequence_np)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic IDLE class data.")
    parser.add_argument('--data-dir', type=str, default=config.DATA_DIR, help='Directory to save IDLE sequences')
    parser.add_argument('--samples', type=int, default=150, help='Number of samples to generate')
    args = parser.parse_args()

    # Generate IDLE samples (enough to match the larger categories)
    generate_idle_frames(args.samples, config.SEQUENCE_LENGTH, args.data_dir)
    print("Done generating IDLE class.")
