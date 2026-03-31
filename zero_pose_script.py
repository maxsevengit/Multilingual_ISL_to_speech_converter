import os
import numpy as np
from glob import glob

def main():
    files = glob('data/raw/**/*.npy', recursive=True)
    print(f"Processing {len(files)} files...")
    
    count = 0
    for f in files:
        try:
            data = np.load(f)
            # Each frame is (162,) where last 36 are pose
            # Set them to zero to match hands-only inference
            data[:, -36:] = 0
            np.save(f, data)
            count += 1
            if count % 100 == 0:
                print(f"  Processed {count}/{len(files)}...")
        except Exception as e:
            print(f"  Error processing {f}: {e}")
            
    print(f"Done. {count} files updated.")

if __name__ == '__main__':
    main()
