import os
import pandas as pd
import cv2
import shutil

# Configuration
TRAIN_CSV = "train_labels.csv"
TEST_CSV = "submit_example.csv"
TRAIN_IMAGE_DIR = "images/train_dataset"  # Where your training images are
TEST_IMAGE_DIR = "images/test_dataset"    # Where your test images are
OUTPUT_DIR = "steel_data"

def prepare_dataset():
    # Create output directories
    for split in ['train', 'val', 'test']:
        os.makedirs(f"{OUTPUT_DIR}/images/{split}", exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/labels/{split}", exist_ok=True)

    # Process training data (from train_dataset folder)
    train_df = pd.read_csv(TRAIN_CSV)
    _process_images(train_df, TRAIN_IMAGE_DIR, 'train')

    # Process test data (from test_dataset folder)
    test_df = pd.read_csv(TEST_CSV)
    _process_images(test_df, TEST_IMAGE_DIR, 'test')

    print("✅ Dataset preparation complete!")
    print(f"Training images: {len(os.listdir(f'{OUTPUT_DIR}/images/train'))}")
    print(f"Test images: {len(os.listdir(f'{OUTPUT_DIR}/images/test'))}")

def _process_images(df, src_img_dir, split):
    for _, row in df.iterrows():
        img_name = row['image']
        bbox = row['bbox']
        
        src_path = os.path.join(src_img_dir, img_name)
        if not os.path.exists(src_path):
            print(f"⚠️ Missing image: {src_path}")
            continue
            
        try:
            # Read image to get dimensions
            img = cv2.imread(src_path)
            if img is None:
                print(f"⚠️ Corrupt image: {src_path}")
                continue
                
            h, w = img.shape[:2]
            
            # Convert bbox to YOLO format
            x1, y1, x2, y2 = map(int, bbox.split())
            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            width = (x2 - x1) / w
            height = (y2 - y1) / h
            
            # Copy image to output directory (not move, to keep original)
            dst_img_path = f"{OUTPUT_DIR}/images/{split}/{img_name}"
            shutil.copy2(src_path, dst_img_path)
            
            # Save label
            label_path = f"{OUTPUT_DIR}/labels/{split}/{img_name.replace('.jpg', '.txt')}"
            with open(label_path, 'w') as f:
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                
        except Exception as e:
            print(f"❌ Error processing {img_name}: {str(e)}")

if __name__ == "__main__":
    prepare_dataset()