---
title: RadPlanet-Vision
emoji: 🔴
colorFrom: red
colorTo: black
sdk: streamlit
sdk_version: 1.41.1
python_version: 3.12
app_file: src/app.py
pinned: false
---

# RadPlanet-Vision: Autonomous Terrain Understanding Using NASA Mars Rover Imagery


**Author**: [Sayed Faisal/ RedPlanet-Vision]  
**Institution**: NASA-style Research Initiative Simulation  
**Date**: 2026-01-13

---

## 1. Abstract
**RadPlanet-Vision** is an autonomous AI pipeline designed to categorize Martian surface terrain features using monocular imagery from the Curiosity and Perseverance rovers. By leveraging weak supervision and deep scattering feature analysis (ResNet + K-Means), this system autonomously organizes raw data into semantic clusters, enabling researchers to rapidly index wide-area survey data. The pipeline is designed for reproducibility, explainability, and scalability.

## Links:
Streamlit: https://sayedfaisalshah12-redplanet-vision-srcapp-tz61tp.streamlit.app/

## 2. Motivation
Planetary exploration suffers from a significant bandwidth bottleneck; transmitting high-resolution data from Mars to Earth is costly and slow. On-board autonomy that can prioritize scientific data (e.g., "Transmit only images containing 'bedrock' or 'anomalies'") is critical for future missions. This project simulates such a capability by building a robust terrain classifier from raw API feeds.

## 3. Dataset
> **⚠️ IMPORTANT NOTICE (2026-01-13)**: The NASA Mars Rover Photos API is currently experiencing a service outage (Status 404: "No such app"). The project includes a **Mock Data Generator** (`src/generate_mock_data.py`) to simulate API responses and valid image data, ensuring the pipeline can still be developed and tested.

Data is sourced directly from the **NASA Mars Rover Photos API**.
- **Rovers**: Curiosity, Perseverance
- **Sensors**: FHAZ (Front Hazard Avoidance), RHAZ, NAVCAM, MAST
- **Data Integrity**: Source attribution is maintained via strict metadata logging (`data/raw/metadata.csv`).
- **Bias**: The dataset is inherently biased towards the rover's traverse path and ground-level viewpoints.

## 4. Methodology

### 4.1 Data Collection (`src/data_collection.py`)
We query the NASA API for specific Sols (Martian Days). A rate-limited pipeline fetches JSON metadata, filters for relevant navigation/science cameras, and downloads imagery to a structured raw directory.

### 4.2 Autonomous Clustering (`src/preprocess.py`)
Lacking pixel-level ground truth, we employ a **Self-Supervised / Weakly-Supervised** approach:
1. **Feature Extraction**: Images are passed through a pre-trained ResNet18 (ImageNet weights) to obtain 512-dimensional feature vectors.
2. **Clustering**: Mini-Batch K-Means clusters these vectors into $K=4$ distinct terrain types (e.g., sandy, rocky, outcrop, other).
3. **Pseudo-Labeling**: Images are physically organized into class folders, creating a labelled dataset for the supervised training phase.

### 4.3 Model Architecture (`src/model.py`)
Two architectures are evaluated:
1. **MarsNet**: A custom, lightweight 4-layer CNN designed for interpretability and low-compute environments.
2. **ResNet18-Transfer**: A standard residual network fine-tuned for Martian terrain features.

### 4.4 Training (`src/train.py`)
Models are trained using Cross-Entropy Loss and Adam optimization. We employ:
- Data Augmentation (Rotation, FLip) to improve robustness.
- Validation Checkpointing to save the best-performing weights.
- Learning Rate Scheduling (if needed).

## 5. Usage

### Prerequisites
```bash
pip install -r requirements.txt
# Set NASA_API_KEY environment variable (optional, defaults to DEMO_KEY)
```

### Execution Pipeline
1. **Fetch Data**:
   ```bash
   python src/data_collection.py --rover curiosity --start_sol 1000 --end_sol 1005
   ```
2. **Auto-Label**:
   ```bash
   python src/preprocess.py
   ```
3. **EDA (Optional)**:
   ```bash
   python src/eda.py
   ```
4. **Train Model**:
   ```bash
   python src/train.py
   ```
5. **Evaluate**:
   ```bash
   python src/evaluate.py
   ```

## 6. Results & Analysis
(To be populated after execution)
- **Accuracy**: Expected >85% on pseudo-labels.
- **Explainability**: See `results/gradcam_*.jpg` for Saliency visualization. The model focuses primarily on texture gradients of the regolith rather than the horizon.

## 7. Limitations
- **Pseudo-Label Noise**: Clustering may group distinct geological features together if they share low-level textural similarities.
- **Grayscale vs Color**: Navcam images are grayscale, Mastcam are color. The model must handle these domain shifts.

## 8. Future Work
- Implementation of **Semantic Segmentation** using U-Net for pixel-wise classification.
- Anomaly Detection auto-encoders for identifying 'science targets of opportunity'.

---
*RadPlanet-Vision is a research simulation and is not affiliated with NASA.*
