import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import random
from pathlib import Path
from src.config import PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR
from src.explainability import run_explainability

def run_demo():
    # 1. Load Model
    model_path = list(MODELS_DIR.glob("*_best.pth"))[0]
    print(f"Loading model: {model_path}")
    model = torch.load(model_path, weights_only=False)
    model.eval()

    # 2. Pick Random Image
    # Recursively find all jpgs in processed dir
    all_images = list(PROCESSED_DATA_DIR.rglob("*.jpg"))
    if not all_images:
        print("No images found in processed data.")
        return
    
    target_img = random.choice(all_images)
    print(f"Analyzing Image: {target_img}")

    # 3. Determine Target Layer (ResNet specific)
    # For ResNet18, layer4[-1] is the last conv block
    target_layer = model.layer4[-1]

    # 4. Run Grad-CAM
    run_explainability(model, str(target_img), target_layer)
    
    print(f"\nDone! Check the result in: {RESULTS_DIR}")

if __name__ == "__main__":
    run_demo()
