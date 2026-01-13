import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import PROCESSED_DATA_DIR, MODELS_DIR, IMAGE_SIZE, RESULTS_DIR

def evaluate_model(rover_name="curiosity", model_path=None):
    if model_path is None:
        # Find the latest best model
        model_path = list(MODELS_DIR.glob(f"{rover_name}_*_best.pth"))
        if not model_path:
            print("No model found.")
            return
        model_path = model_path[0]

    print(f"Loading model from {model_path}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, weights_only=False)
    model = model.to(device)
    model.eval()

    # Data
    data_dir = PROCESSED_DATA_DIR / rover_name
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    class_names = dataset.classes
    
    all_preds = []
    all_labels = []
    
    print("Evaluating...")
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # Metrics
    print("\nClassification Report:")
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(report)
    
    # Save Report
    with open(RESULTS_DIR / "classification_report.txt", "w") as f:
        f.write(report)
        
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(RESULTS_DIR / "confusion_matrix.png")
    print(f"Results saved to {RESULTS_DIR}")

if __name__ == "__main__":
    evaluate_model()
