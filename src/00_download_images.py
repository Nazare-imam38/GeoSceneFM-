import os
import sys
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def main():
    print("Generating 60 synthetic dummy images for the prototype...")
    
    categories = ['Residential', 'Commercial', 'Downtown', 'Park']
    base_colors = {
        'Residential': [200, 100, 100],  # reddish
        'Commercial': [100, 200, 100],   # greenish
        'Downtown': [100, 100, 200],     # blueish
        'Park': [200, 200, 100]          # yellowish
    }
    
    generated_count = 0
    np.random.seed(config.RANDOM_SEED)
    
    for cat in categories:
        for i in range(15):
            # Create a 224x224 image
            color = base_colors[cat]
            # Add some noise
            noise = np.random.randint(-30, 30, (224, 224, 3))
            
            img_data = np.zeros((224, 224, 3), dtype=np.uint8)
            img_data[:, :] = color
            
            # Apply noise and clip
            img_data = np.clip(img_data + noise, 0, 255).astype(np.uint8)
            
            img = Image.fromarray(img_data)
            
            filename = f"{cat.lower()}_{i+1:02d}.jpg"
            dest_path = os.path.join(config.RAW_DIR, filename)
            
            img.save(dest_path)
            generated_count += 1
            
    print(f"Successfully generated {generated_count} synthetic images in {config.RAW_DIR}")
    print("NOTE: These are dummy images to test the pipeline because Kaggle SDK / Web downloads were blocked.")

if __name__ == "__main__":
    main()
