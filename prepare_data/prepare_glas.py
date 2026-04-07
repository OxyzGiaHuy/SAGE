import os
import glob
import shutil
import random
from tqdm import tqdm

SOURCE_DIR = "/path/to/Warwick_QU_Dataset"
DEST_DIR = "/path/to/dataset/processed_glas"

TRAIN_SPLIT = 0.8 
SEED = 42          
NEW_MASK_SUFFIX = "_mask"

def create_dirs(base_path):
    for sub in ['images', 'masks']:
        os.makedirs(os.path.join(base_path, sub), exist_ok=True)

def copy_and_rename_pair(img_path, dst_root_name):
    src_img = img_path
    src_mask = img_path.replace(".bmp", "_anno.bmp")
    if not os.path.exists(src_mask):
        src_mask = img_path.replace(".bmp", "_Anno.bmp")
    
    if not os.path.exists(src_mask):
        print(f"[Warning] Mask not found for {os.path.basename(src_img)}")
        return False

    dst_img_dir = os.path.join(DEST_DIR, dst_root_name, "images")
    dst_mask_dir = os.path.join(DEST_DIR, dst_root_name, "masks")

    filename = os.path.basename(img_path)
    filename_no_ext = os.path.splitext(filename)[0]
    
    dst_img_path = os.path.join(dst_img_dir, filename)
    
    new_mask_name = f"{filename_no_ext}{NEW_MASK_SUFFIX}.bmp"
    dst_mask_path = os.path.join(dst_mask_dir, new_mask_name)

    shutil.copy2(src_img, dst_img_path)
    shutil.copy2(src_mask, dst_mask_path)

    if dst_root_name in ['testA', 'testB']:
        dst_img_dir_ab = os.path.join(DEST_DIR, "testAB", "images")
        dst_mask_dir_ab = os.path.join(DEST_DIR, "testAB", "masks")
        
        shutil.copy2(src_img, os.path.join(dst_img_dir_ab, filename))
        shutil.copy2(src_mask, os.path.join(dst_mask_dir_ab, new_mask_name))

    return True

def main():
    random.seed(SEED)

    if not os.path.exists(SOURCE_DIR):
        raise FileNotFoundError(f"CRITICAL ERROR: Source not found at:\n{SOURCE_DIR}\nCheck capitalization (UNET vs UNet).")

    print(f"Creating dataset at: {DEST_DIR}")
    print(f"Applying mask suffix conversion: '_anno' -> '{NEW_MASK_SUFFIX}'")
    
    if os.path.exists(DEST_DIR):
        shutil.rmtree(DEST_DIR)
    
    create_dirs(os.path.join(DEST_DIR, "train"))
    create_dirs(os.path.join(DEST_DIR, "val"))
    create_dirs(os.path.join(DEST_DIR, "testA"))
    create_dirs(os.path.join(DEST_DIR, "testB"))
    create_dirs(os.path.join(DEST_DIR, "testAB"))

    print("Scanning source directory...")
    all_train_imgs = sorted(glob.glob(os.path.join(SOURCE_DIR, "train_*.bmp")))
    all_testA_imgs = sorted(glob.glob(os.path.join(SOURCE_DIR, "testA_*.bmp")))
    all_testB_imgs = sorted(glob.glob(os.path.join(SOURCE_DIR, "testB_*.bmp")))

    all_train_imgs = [f for f in all_train_imgs if "_anno" not in f.lower()]
    all_testA_imgs = [f for f in all_testA_imgs if "_anno" not in f.lower()]
    all_testB_imgs = [f for f in all_testB_imgs if "_anno" not in f.lower()]

    if len(all_train_imgs) == 0:
        raise ValueError("Found 0 images. Check path capitalization!")

    print(f"Found {len(all_train_imgs)} train, {len(all_testA_imgs)} testA, {len(all_testB_imgs)} testB.")

    random.shuffle(all_train_imgs)
    split_idx = int(len(all_train_imgs) * TRAIN_SPLIT)
    train_imgs = all_train_imgs[:split_idx]
    val_imgs = all_train_imgs[split_idx:]

    print("\nProcessing...")
    for p in tqdm(train_imgs, desc="Train"): 
        copy_and_rename_pair(p, "train")
    for p in tqdm(val_imgs, desc="Val"): 
        copy_and_rename_pair(p, "val")
    for p in tqdm(all_testA_imgs, desc="TestA"): 
        copy_and_rename_pair(p, "testA")
    for p in tqdm(all_testB_imgs, desc="TestB"): 
        copy_and_rename_pair(p, "testB")

    print("\nDONE. Your config.yaml should now look like this:")
    print("-" * 30)
    print(f"root_dir: {DEST_DIR}")
    print(f"mask_suffix: \"{NEW_MASK_SUFFIX}\"")
    print("train: { images: train/images, masks: train/masks }")
    print("val:   { images: val/images,   masks: val/masks }")
    print("test:  { images: testAB/images,  masks: testAB/masks }")
    print("-" * 30)

if __name__ == "__main__":
    main()