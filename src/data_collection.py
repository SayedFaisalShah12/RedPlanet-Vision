import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import time
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor

from src.config import (
    NASA_API_KEY, 
    NASA_API_URL, 
    RAW_DATA_DIR, 
    TARGET_CAMERAS, 
    NASA_API_URL
)

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_collection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MarsRoverDataLoader:
    def __init__(self, rover_name="curiosity"):
        self.rover_name = rover_name.lower()
        self.base_url = f"{NASA_API_URL}/{self.rover_name}/photos"
        self.rover_dir = RAW_DATA_DIR / self.rover_name
        self.rover_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.rover_dir / f"{self.rover_name}_metadata.csv"

    def fetch_photos_metadata(self, sol_start, sol_end):
        """Fetches metadata for photos within a Sol range."""
        all_photos = []
        
        logger.info(f"Fetching metadata for {self.rover_name} from Sol {sol_start} to {sol_end}...")
        
        for sol in tqdm(range(sol_start, sol_end + 1), desc="Fetching Metadata"):
            params = {
                "sol": sol,
                "api_key": NASA_API_KEY
            }
            try:
                response = requests.get(self.base_url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    photos = data.get("photos", [])
                    # Filter for target cameras
                    relevant_photos = [
                        p for p in photos 
                        if p["camera"]["name"] in TARGET_CAMERAS
                    ]
                    all_photos.extend(relevant_photos)
                    time.sleep(0.5) # Be nice to the API
                elif response.status_code == 429:
                    logger.warning(f"Rate limited at Sol {sol}. Sleeping for 60s...")
                    time.sleep(60)
                else:
                    logger.error(f"Error {response.status_code} at Sol {sol}: {response.text}")
            except Exception as e:
                logger.error(f"Failed to fetch Sol {sol}: {e}")

        logger.info(f"Found {len(all_photos)} photos matching criteria.")
        return pd.DataFrame(all_photos)

    def download_image(self, url, img_id):
        """Downloads a single image."""
        try:
            # Create filename structure
            filename = f"{img_id}.jpg"
            filepath = self.rover_dir / filename
            
            if filepath.exists():
                return str(filepath), True # Already exists

            response = requests.get(url, stream=True, timeout=15)
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                return str(filepath), True
            else:
                return None, False
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return None, False

    def run_pipeline(self, sol_start, sol_end, max_workers=5):
        """Main execution pipeline."""
        # 1. Fetch Metadata
        df = self.fetch_photos_metadata(sol_start, sol_end)
        
        if df.empty:
            logger.warning("No photos found.")
            return

        # 2. Save detailed metadata
        # Flatten the camera and rover dicts for cleaner CSV
        if 'camera' in df.columns:
            df['camera_name'] = df['camera'].apply(lambda x: x.get('name'))
            df['camera_full_name'] = df['camera'].apply(lambda x: x.get('full_name'))
        
        # Save raw metadata
        df.to_csv(self.metadata_file, index=False)
        logger.info(f"Metadata saved to {self.metadata_file}")

        # 3. Download Images
        logger.info(f"Starting download of {len(df)} images with {max_workers} workers...")
        
        download_status = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for _, row in df.iterrows():
                futures.append(
                    executor.submit(self.download_image, row['img_src'], row['id'])
                )
            
            for future in tqdm(futures, desc="Downloading Images"):
                path, success = future.result()
                download_status.append(success)

        success_count = sum(download_status)
        logger.info(f"Successfully downloaded {success_count}/{len(df)} images.")

if __name__ == "__main__":
    # Example usage for testing
    import argparse
    parser = argparse.ArgumentParser(description="Fetch Mars Rover Data")
    parser.add_argument("--rover", type=str, default="curiosity", choices=["curiosity", "perseverance"])
    parser.add_argument("--start_sol", type=int, default=1000)
    parser.add_argument("--end_sol", type=int, default=1002)
    
    args = parser.parse_args()
    
    loader = MarsRoverDataLoader(rover_name=args.rover)
    loader.run_pipeline(args.start_sol, args.end_sol)
