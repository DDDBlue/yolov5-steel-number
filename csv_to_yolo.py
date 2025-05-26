import os
import pandas as pd
from pathlib import Path
import cv2

# Config
CSV_PATH = "steel_data/train_labels.csv"
IMAGE_DIR = "steel_data/images/train"
LABEL_DIR = "steel_data/labels/train"
os.makedirs(LABEL_DIR, exist_ok=True)

# Read CSV
df = pd.read_csv(CSV_PATH)

# Process each image
for img_name, group in df.groupby('image'):
    img_path = os.path.join(IMAGE_DIR, img_name)
    if not os.path.exists(img_path):
        print(f"⚠️ Missing image: {img_path}")
        continue
    
    # Get image dimensions
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Corrupt image: {img_path}")
        continue
    h, w = img.shape[:2]
    
    # Create YOLO label file
    label_path = os.path.join(LABEL_DIR, Path(img_name).with_suffix('.txt'))
    with open(label_path, 'w') as f:
        for _, row in group.iterrows():
            # Convert CSV bbox (x1,y1,x2,y2) to YOLO format (class x_center y_center width height)
            x1, y1, x2, y2 = map(int, row['bbox'].split())
            
            # Calculate normalized values
            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            width = (x2 - x1) / w
            height = (y2 - y1) / h
            
            # Write to file (class 0 for single-class)
            f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

print(f"✅ Converted {len(df.groupby('image'))} images with {len(df)} total boxes")