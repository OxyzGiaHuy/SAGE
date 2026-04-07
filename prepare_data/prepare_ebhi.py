import os
import glob
import shutil
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split

BASE_SRC = "/path/to/EBHI-SEG/Adenocarcinoma"
SRC_IMG_DIR = os.path.join(BASE_SRC, "image")
SRC_MASK_DIR = os.path.join(BASE_SRC, "label")

DEST_DIR = "/path/to/dataset/processed_ebhi"
TEST_SIZE = 0.15
VAL_SIZE = 0.15

SEED = 42
NEW_MASK_SUFFIX = "_mask" 

def create_dirs(base_path):
    for sub in ['images', 'masks']:
        os.makedirs(os.path.join(base_path, sub), exist_ok=True)

def copy_and_rename_pair(filename, split_name):
    src_img_path = os.path.join(SRC_IMG_DIR, filename)
    src_mask_path = os.path.join(SRC_MASK_DIR, filename)

    if not os.path.exists(src_mask_path):
        print(f"[Warning] Mask not found for {filename}, skipping.")
        return False

    dst_img_dir = os.path.join(DEST_DIR, split_name, "images")
    dst_mask_dir = os.path.join(DEST_DIR, split_name, "masks")

    name_no_ext = os.path.splitext(filename)[0]
    ext = os.path.splitext(filename)[1] # likely .png

    dst_img_path = os.path.join(dst_img_dir, filename)
    new_mask_name = f"{name_no_ext}{NEW_MASK_SUFFIX}{ext}"
    dst_mask_path = os.path.join(dst_mask_dir, new_mask_name)

    shutil.copy2(src_img_path, dst_img_path)
    shutil.copy2(src_mask_path, dst_mask_path)
    return True

def main():
    random.seed(SEED)

    if not os.path.exists(SRC_IMG_DIR):
        raise FileNotFoundError(f"Source Images not found at:\n{SRC_IMG_DIR}")
    if not os.path.exists(SRC_MASK_DIR):
        raise FileNotFoundError(f"Source Labels not found at:\n{SRC_MASK_DIR}")

    print(f"Creating EBHI dataset structure at: {DEST_DIR}")
    if os.path.exists(DEST_DIR):
        shutil.rmtree(DEST_DIR)
    
    create_dirs(os.path.join(DEST_DIR, "train"))
    create_dirs(os.path.join(DEST_DIR, "val"))
    create_dirs(os.path.join(DEST_DIR, "test"))

    print("Scanning source directory...")
    all_files = sorted(os.listdir(SRC_IMG_DIR))
    all_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    if len(all_files) == 0:
        raise ValueError("Found 0 images. Check your path!")
    
    print(f"Found {len(all_files)} total images.")

    train_val_files, test_files = train_test_split(all_files, test_size=TEST_SIZE, random_state=SEED)
    relative_val_size = VAL_SIZE / (1 - TEST_SIZE)
    
    train_files, val_files = train_test_split(train_val_files, test_size=relative_val_size, random_state=SEED)

    print("-" * 30)
    print(f"Split Summary:")
    print(f"Train: {len(train_files)} ({len(train_files)/len(all_files):.1%})")
    print(f"Val:   {len(val_files)} ({len(val_files)/len(all_files):.1%})")
    print(f"Test:  {len(test_files)} ({len(test_files)/len(all_files):.1%})")
    print("-" * 30)

    print("Copying and renaming...")
    for f in tqdm(train_files, desc="Train"): copy_and_rename_pair(f, "train")
    for f in tqdm(val_files, desc="Val"):   copy_and_rename_pair(f, "val")
    for f in tqdm(test_files, desc="Test"):  copy_and_rename_pair(f, "test")

    print("\nDONE. Update your config.yaml for EBHI:")
    print("-" * 30)
    print(f"root_dir: {DEST_DIR}")
    print(f"mask_suffix: \"{NEW_MASK_SUFFIX}\"")
    print("train: { images: train/images, masks: train/masks }")
    print("val:   { images: val/images,   masks: val/masks }")
    print("test:  { images: test/images,  masks: test/masks }")
    print("-" * 30)

if __name__ == "__main__":
    main()