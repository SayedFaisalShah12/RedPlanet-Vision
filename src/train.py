import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import copy
import time
from pathlib import Path
import matplotlib.pyplot as plt

from src.config import PROCESSED_DATA_DIR, MODELS_DIR, IMAGE_SIZE, BATCH_SIZE, RANDOM_SEED
from src.model import MarsCNN, get_resnet_model

def train_model(rover_name="curiosity", model_type="resnet", num_epochs=10):
    data_dir = PROCESSED_DATA_DIR / rover_name
    if not data_dir.exists():
        print(f"Data directory {data_dir} not found. Run preprocess.py first.")
        return

    # Transforms
    train_transforms = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Dataset
    full_dataset = ImageFolder(root=data_dir) # Uses the transforms later wrapping or split
    # Note: ImageFolder applies transforms at init. To separate train/val transforms properly with random_split is tricky.
    # We will apply transforms to the subset wrappers or simpler: Apply train transforms to all for now (simplicity) 
    # OR better: implementation correct approach.
    
    # Correct approach: Two datasets referencing same path, split indices.
    train_dataset_full = ImageFolder(root=data_dir, transform=train_transforms)
    val_dataset_full = ImageFolder(root=data_dir, transform=val_transforms)
    
    dataset_size = len(train_dataset_full)
    class_names = train_dataset_full.classes
    num_classes = len(class_names)
    
    print(f"Classes found: {class_names}")
    
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    
    # We need fixed indices for reproducibility
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_subset, val_subset = random_split(train_dataset_full, [train_size, val_size], generator=generator)
    
    # HACK: Fix the transform for validation subset. 
    # random_split returns Subset, which relies on the underlying dataset.
    # To use different transforms, we must perform the split on indices, then create Subsets with DIFFERENT underlying datasets.
    indices = torch.randperm(dataset_size, generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_subset = torch.utils.data.Subset(train_dataset_full, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset_full, val_indices)

    dataloaders = {
        'train': DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2),
        'val': DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    }
    dataset_sizes = {'train': train_size, 'val': val_size}
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Model Initialization
    if model_type == "resnet":
        model = get_resnet_model(num_classes=num_classes)
    else:
        model = MarsCNN(num_classes=num_classes)
        
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training Loop
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Save Model
    model.load_state_dict(best_model_wts)
    model_path = MODELS_DIR / f"{rover_name}_{model_type}_best.pth"
    torch.save(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Plot History
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.legend()
    plt.title('Accuracy')
    
    plt.savefig(MODELS_DIR / "training_history.png")
    
if __name__ == "__main__":
    train_model()
