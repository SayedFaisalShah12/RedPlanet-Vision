import os
from pathlib import Path

# Project Context
PROJECT_NAME = "RadPlanet-Vision"
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Ensure directories exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# NASA API
NASA_API_KEY = os.getenv("NASA_API_KEY", "DEMO_KEY")
NASA_API_URL = "https://api.nasa.gov/mars-photos/api/v1/rovers"

# Data Collection Settings
ROVERS = ["curiosity", "perseverance"] 
# Cameras focused on terrain: FHAZ (Front Hazard), RHAZ (Rear Hazard), MAST (Mast Camera), NAVCAM (Navigation)
TARGET_CAMERAS = ["FHAZ", "RHAZ", "MAST", "NAVCAM", "MCZ_RIGHT", "MCZ_LEFT"]

# Image Settings
IMAGE_SIZE = (224, 224) # Standard for ResNet/VGG
BATCH_SIZE = 32
RANDOM_SEED = 42
