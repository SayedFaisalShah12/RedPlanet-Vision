import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from src.config import RAW_DATA_DIR, IMAGE_SIZE

def generate_mock_data(rover_name="curiosity", n_samples=20):
    print(f"Generating {n_samples} mock samples for {rover_name}...")
    
    rover_dir = RAW_DATA_DIR / rover_name
    rover_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Generate Metadata
    data = {
        'id': range(1000, 1000 + n_samples),
        'sol': np.random.randint(1000, 1005, n_samples),
        'camera_name': np.random.choice(["FHAZ", "RHAZ", "MAST", "NAVCAM"], n_samples),
        'img_src': [f"http://mock-url/img_{i}.jpg" for i in range(1000, 1000 + n_samples)]
    }
    df = pd.DataFrame(data)
    metadata_path = rover_dir / f"{rover_name}_metadata.csv"
    df.to_csv(metadata_path, index=False)
    print(f"Saved mock metadata to {metadata_path}")
    
    # 2. Generate Mock Images
    for img_id in data['id']:
        # Create a random noisy image to simulate "terrain"
        # We add some structure so clustering isn't completely random noise
        img_array = np.random.randint(0, 255, (IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.uint8)
        
        # Add a "color tint" based on ID to simulate different "clusters"
        if img_id % 4 == 0:
            img_array[:, :, 0] = 200 # Reddish
        elif img_id % 4 == 1:
            img_array[:, :, 1] = 200 # Greenish (Algae on Mars?!)
        elif img_id % 4 == 2:
            img_array[:, :, 2] = 200 # Blueish
        
        img = Image.fromarray(img_array)
        img.save(rover_dir / f"{img_id}.jpg")
        
    print(f"Saved {n_samples} mock images to {rover_dir}")
    print("You can now run 'python src/preprocess.py'")

if __name__ == "__main__":
    generate_mock_data()
