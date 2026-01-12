import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
import pandas as pd
import numpy as np
from tqdm import tqdm
import shutil
from pathlib import Path

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, IMAGE_SIZE, RANDOM_SEED

# Set seed
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class MarsRawDataset(Dataset):
    """Dataset for raw images before labeled organization."""
    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = str(row['id'])
        img_path = self.root_dir / f"{img_id}.jpg"
        
        try:
            image = Image.open(img_path).convert('RGB')
        except (OSError, FileNotFoundError):
             # Return a black image if corrupted/missing, handled in collation loop ideally
             # But here we just return a tensor of zeros to avoid crashing
            image = Image.new('RGB', IMAGE_SIZE)
            
        if self.transform:
            image = self.transform(image)
            
        return image, img_id

def extract_features(dataset, device='cpu', batch_size=32):
    """Extracts ResNet features for clustering."""
    model = models.resnet18(pretrained=True)
    model.fc = nn.Identity() # Remove classification layer
    model = model.to(device)
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    features = []
    ids = []
    
    print("Extracting features for auto-labeling...")
    with torch.no_grad():
        for imgs, img_ids in tqdm(loader):
            imgs = imgs.to(device)
            feats = model(imgs)
            features.append(feats.cpu().numpy())
            ids.extend(img_ids)
            
    return np.vstack(features), ids

def auto_label_dataset(n_clusters=4, rover_name="curiosity"):
    """
    Autonomous pipeline to cluster images into terrain types.
    """
    rover_dir = RAW_DATA_DIR / rover_name
    metadata_path = rover_dir / f"{rover_name}_metadata.csv"
    
    if not metadata_path.exists():
        print("Metadata not found.")
        return

    df = pd.read_csv(metadata_path)
    # Filter only existing files
    df['exists'] = df['id'].apply(lambda x: (rover_dir / f"{x}.jpg").exists())
    df = df[df['exists']].reset_index(drop=True)
    
    if df.empty:
        print("No images found to process.")
        return

    # Transform
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = MarsRawDataset(df, rover_dir, transform=transform)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Extract Features
    features, img_ids = extract_features(dataset, device)
    
    # Clustering
    print(f"Clustering into {n_clusters} terrain types...")
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, batch_size=256)
    labels = kmeans.fit_predict(features)
    
    # Save Labels
    results_df = pd.DataFrame({'id': img_ids, 'cluster_label': labels})
    
    # Merge with full metadata
    # Ensure ID types match (int vs str)
    results_df['id'] = results_df['id'].astype(str)
    df['id'] = df['id'].astype(str)
    
    final_df = pd.merge(df, results_df, on='id')
    
    # Organize into Processed Directory
    processed_rover_dir = PROCESSED_DATA_DIR / rover_name
    if processed_rover_dir.exists():
        shutil.rmtree(processed_rover_dir)
    processed_rover_dir.mkdir(parents=True)
    
    print("Organizing data into labeled folders...")
    for label in range(n_clusters):
        label_dir = processed_rover_dir / f"terrain_type_{label}"
        label_dir.mkdir(exist_ok=True)
        
    for _, row in tqdm(final_df.iterrows(), total=len(final_df)):
        src = rover_dir / f"{row['id']}.jpg"
        dst = processed_rover_dir / f"terrain_type_{row['cluster_label']}" / f"{row['id']}.jpg"
        try:
            shutil.copy2(src, dst)
        except Exception:
            pass

    print(f"Data organized at {processed_rover_dir}")
    print("IMPORTANT: Inspect the folders and rename 'terrain_type_X' to semantic names (e.g., sandy, rocky) if desired.")

if __name__ == "__main__":
    auto_label_dataset()
