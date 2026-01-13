import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from src.config import IMAGE_SIZE, RESULTS_DIR

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        # Forward pass
        self.model.zero_grad()
        output = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(output)
            
        # Backward pass
        output[0, class_idx].backward()
        
        # Pool the gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight activations
        activations = self.activations[0]
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]
            
        # Average the channels of the activations
        heatmap = torch.mean(activations, dim=0).cpu().detach().numpy()
        
        # Normalize heatmap
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) if np.max(heatmap) > 0 else 1
        
        return heatmap

def visualize_cam(heatmap, original_image_path, save_path):
    print(f"Reading image from: {original_image_path}")
    img = cv2.imread(str(original_image_path))
    print(f"Image object type: {type(img)}")
    
    if img is None:
        raise ValueError(f"Could not read image at {original_image_path} with cv2. Check path and permissions.")
        
    print(f"Resizing image of shape {img.shape} to {IMAGE_SIZE}")
    img = cv2.resize(img, IMAGE_SIZE)
    
    print(f"Heatmap type: {type(heatmap)}, shape: {heatmap.shape}")
    print(f"Heatmap dtype: {heatmap.dtype}")
    sys.stdout.flush()
    
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    superimposed_img = heatmap * 0.4 + img
    cv2.imwrite(str(save_path), superimposed_img)

def run_explainability(model, image_path, target_layer):
    device = next(model.parameters()).device
    
    # Preprocess
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)
    
    grad_cam = GradCAM(model, target_layer)
    heatmap = grad_cam(x)
    
    save_path = RESULTS_DIR / f"gradcam_{Path(image_path).stem}.jpg"
    visualize_cam(heatmap, image_path, save_path)
    print(f"Saved Grad-CAM visualization to {save_path}")

# Example usage commented out to avoid runtime errors on import
# if __name__ == "__main__":
#     model = torch.load("path/to/model.pth", weights_only=False)
#     # For ResNet, target_layer is usually model.layer4[-1]
#     run_explainability(model, "path/to/image.jpg", model.layer4[-1])
