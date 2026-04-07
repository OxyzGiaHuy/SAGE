import os
import glob
import shutil
import re
import random
import warnings
from collections import defaultdict
from tqdm import tqdm
from sklearn.model_selection import train_test_split

SOURCE_DIR = "/path/to/dataset/new_patched_colon"
DEST_DIR = "/path/to/dataset/processed_colon"
NEW_MASK_SUFFIX = "_mask" 

TEST_SIZE_1 = 0.3  
TEST_SIZE_2 = 0.5  

SEED = 42

def create_dirs(base_path):
    """Creates the standard directory structure."""
    for split in ['train', 'val', 'test']:
        for sub in ['images', 'masks']:
            os.makedirs(os.path.join(base_path, split, sub), exist_ok=True)

def get_mask_path(img_path):
    """Finds the source mask path, handling neg_masks folder logic."""
    if 'neg_images' in img_path:
        m = img_path.replace('neg_images', 'neg_masks').replace('.jpg', '.png')
        if not os.path.exists(m): 
            m = m.replace('.png', '.jpg')
    else:
        m = img_path.replace('images', 'masks').replace('.jpg', '.png')
        if not os.path.exists(m): 
            m = m.replace('.png', '.jpg')
    
    if not os.path.exists(m):
        return None
    return m

def copy_files(wsi_ids, wsi_dict, split_name):
    """Copies all patches belonging to the listed WSI IDs to the target split folder."""
    
    dst_img_dir = os.path.join(DEST_DIR, split_name, "images")
    dst_mask_dir = os.path.join(DEST_DIR, split_name, "masks")
    
    for uid in tqdm(wsi_ids, desc=f"Processing {split_name}"):
        patches = wsi_dict[uid]
        
        for src_img_path in patches:
            src_mask_path = get_mask_path(src_img_path)
            if src_mask_path is None:
                continue 

            filename = os.path.basename(src_img_path)
            name_no_ext = os.path.splitext(filename)[0]
            
            dst_img_path = os.path.join(dst_img_dir, filename)
            dst_mask_name = f"{name_no_ext}{NEW_MASK_SUFFIX}.png" 
            dst_mask_path = os.path.join(dst_mask_dir, dst_mask_name)

            shutil.copy2(src_img_path, dst_img_path)
            shutil.copy2(src_mask_path, dst_mask_path)

def main():
    random.seed(SEED)
    
    if not os.path.exists(SOURCE_DIR):
        raise FileNotFoundError(f"Source directory not found: {SOURCE_DIR}")

    print(f"Creating standardized dataset at: {DEST_DIR}")
    if os.path.exists(DEST_DIR):
        print("Cleaning previous destination folder...")
        shutil.rmtree(DEST_DIR)
    
    create_dirs(DEST_DIR)

    print("Scanning source directory...")
    pos_img_paths = sorted(glob.glob(os.path.join(SOURCE_DIR, "images/*.jpg")))
    neg_img_paths = sorted(glob.glob(os.path.join(SOURCE_DIR, "neg_images/*.jpg")))
    
    if len(pos_img_paths) == 0 and len(neg_img_paths) == 0:
        raise ValueError("No images found! Check source path.")

    print("Grouping patches by WSI ID...")
    wsi_pattern = re.compile(r"(.+)_patch_level")
    wsi_dict = defaultdict(list)
    wsi_labels = {}

    for p in pos_img_paths:
        match = wsi_pattern.search(os.path.basename(p))
        if match:
            wsi_id = match.group(1)
            wsi_dict[wsi_id].append(p)
            wsi_labels[wsi_id] = 1

    for p in neg_img_paths:
        match = wsi_pattern.search(os.path.basename(p))
        if match:
            wsi_id = match.group(1)
            wsi_dict[wsi_id].append(p)
            if wsi_id not in wsi_labels:
                wsi_labels[wsi_id] = 0

    all_wsi_ids = list(wsi_dict.keys())
    all_labels = [wsi_labels[uid] for uid in all_wsi_ids]

    print(f"Found {len(all_wsi_ids)} unique WSIs.")
    print(f" - Positive: {sum(all_labels)}")
    print(f" - Negative: {len(all_labels) - sum(all_labels)}")

    print("Performing Stratified Split (WSI-level)...")
    
    train_ids, temp_ids, train_y, temp_y = train_test_split(
        all_wsi_ids, all_labels, 
        test_size=TEST_SIZE_1, 
        stratify=all_labels, 
        random_state=SEED
    )

    val_ids, test_ids, _, _ = train_test_split(
        temp_ids, temp_y, 
        test_size=TEST_SIZE_2, 
        stratify=temp_y, 
        random_state=SEED
    )

    print("-" * 30)
    print(f"Train WSIs: {len(train_ids)}")
    print(f"Val WSIs:   {len(val_ids)}")
    print(f"Test WSIs:  {len(test_ids)}")
    print("-" * 30)

    copy_files(train_ids, wsi_dict, "train")
    copy_files(val_ids, wsi_dict, "val")
    copy_files(test_ids, wsi_dict, "test")

    print("\nDONE. Update your config.yaml:")
    print("-" * 30)
    print(f"root_dir: {DEST_DIR}")
    print(f"mask_suffix: \"{NEW_MASK_SUFFIX}\"")
    print("train: { images: train/images, masks: train/masks }")
    print("val:   { images: val/images,   masks: val/masks }")
    print("test:  { images: test/images,  masks: test/masks }")
    print("-" * 30)

if __name__ == "__main__":
    main()