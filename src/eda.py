import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import random
from PIL import Image

from src.config import RAW_DATA_DIR, RESULTS_DIR

class MarsEDA:
    def __init__(self, rover_name="curiosity"):
        self.rover_name = rover_name
        self.data_dir = RAW_DATA_DIR / rover_name
        self.metadata_file = self.data_dir / f"{rover_name}_metadata.csv"
        self.plots_dir = RESULTS_DIR / "eda"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        if not self.metadata_file.exists():
            print(f"Metadata file {self.metadata_file} not found. Run data_collection.py first.")
            return None
        return pd.read_csv(self.metadata_file)

    def generate_stats_plots(self, df):
        """Generates distribution plots."""
        # 1. Camera Distribution
        plt.figure(figsize=(10, 6))
        sns.countplot(x='camera_name', data=df, palette='viridis')
        plt.title(f'Image Distribution by Camera ({self.rover_name})')
        plt.xlabel('Camera')
        plt.ylabel('Count')
        plt.savefig(self.plots_dir / 'camera_distribution.png')
        plt.close()

        # 2. Sol Distribution
        plt.figure(figsize=(12, 6))
        sns.histplot(df['sol'], bins=20, kde=True, color='orange')
        plt.title(f'Image Distribution by Sol ({self.rover_name})')
        plt.xlabel('Sol')
        plt.ylabel('Count')
        plt.savefig(self.plots_dir / 'sol_distribution.png')
        plt.close()

    def visualize_samples(self, df, n=5):
        """Visualizes random sample images."""
        if df.empty:
            return

        sample_df = df.sample(n=min(n, len(df)), random_state=42)
        
        fig, axes = plt.subplots(1, n, figsize=(15, 5))
        if n == 1: axes = [axes]
        
        for i, (_, row) in enumerate(sample_df.iterrows()):
            img_path = self.data_dir / f"{row['id']}.jpg"
            if img_path.exists():
                img = Image.open(img_path)
                axes[i].imshow(img)
                axes[i].axis('off')
                axes[i].set_title(f"{row['camera_name']}\nSol: {row['sol']}")
            else:
                axes[i].text(0.5, 0.5, "Image not found", ha='center')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'sample_images.png')
        plt.close()

    def run(self):
        print("Starting EDA...")
        df = self.load_data()
        if df is not None:
            print(f"Loaded metadata with {len(df)} records.")
            self.generate_stats_plots(df)
            self.visualize_samples(df)
            print(f"EDA Complete. Plots saved to {self.plots_dir}")

if __name__ == "__main__":
    eda = MarsEDA()
    eda.run()
