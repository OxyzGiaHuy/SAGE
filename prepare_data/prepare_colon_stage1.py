import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from joblib import Parallel, delayed
 
SRC_POS_DIR = "/path/to/Colon/tissue-train-pos-v1"
SRC_NEG_DIR = "/path/to/Colon/tissue-train-neg"
DST_ROOT = "/path/to/dataset/new_patched_colon"

PATCH_SIZE = 1536
STRIDE = 512
RESIZE_TO = None 

STD_THRESHOLD = 10 
MEAN_THRESHOLD = 230
NUM_CORES = -1

def is_tissue(patch):
    """Unsupervised check for tissue content."""
    if patch.size == 0: 
        return False
    mean_val = np.mean(patch)
    std_val = np.std(patch)

    return (std_val >= STD_THRESHOLD) and (mean_val <= MEAN_THRESHOLD)

def process_slide(img_path, is_positive_source):
    """Worker function for both positive and negative slides."""
    cv2.setNumThreads(0)
    local_scanned = 0
    local_kept = 0
    
    img = cv2.imread(img_path)
    if img is None: 
        return 0, 0
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    base_no_ext = os.path.splitext(img_path)[0]
    ext = os.path.splitext(img_path)[1]
    mask_path = f"{base_no_ext}_mask{ext}"
    
    if is_positive_source:
        if not os.path.exists(mask_path):
            if os.path.exists(f"{base_no_ext}_mask.png"): 
                mask_path = f"{base_no_ext}_mask.png"
            elif os.path.exists(f"{base_no_ext}_mask.jpg"): 
                mask_path = f"{base_no_ext}_mask.jpg"
            else: 
                return 0, 0 
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None: 
            return 0, 0
    else:
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    h, w = img.shape[:2]

    for y in range(0, h, STRIDE):
        for x in range(0, w, STRIDE):
            img_patch = img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            
            if img_patch.shape[0] != PATCH_SIZE or img_patch.shape[1] != PATCH_SIZE:
                continue

            local_scanned += 1
            if is_tissue(img_patch):
                
                mask_patch = mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                
                fname = os.path.basename(img_path).replace(ext, f"_patch_level0_x{x}_y{y}.jpg")
                
                if is_positive_source:
                    img_sub, mask_sub = "images", "masks"
                else:
                    img_sub, mask_sub = "neg_images", "neg_masks"

                # Save
                save_img_path = os.path.join(DST_ROOT, img_sub, fname)
                cv2.imwrite(save_img_path, cv2.cvtColor(img_patch, cv2.COLOR_RGB2BGR))
                
                mask_fname = fname.replace(".jpg", ".png")
                cv2.imwrite(os.path.join(DST_ROOT, mask_sub, mask_fname), mask_patch)
                
                local_kept += 1
                    
    return local_scanned, local_kept

def process_dataset():
    for sub in ['images', 'masks', 'neg_images', 'neg_masks']:
        os.makedirs(os.path.join(DST_ROOT, sub), exist_ok=True)
        
    pos_images = [f for f in glob(os.path.join(SRC_POS_DIR, "*")) if "_mask" not in f and os.path.isfile(f)]
    print(f"Processing {len(pos_images)} Positive Slides (Unsupervised Filter Only)...")
    
    pos_results = Parallel(n_jobs=NUM_CORES)(
        delayed(process_slide)(p, True) for p in tqdm(pos_images)
    )
    pos_kept = sum(r[1] for r in pos_results)

    neg_images = glob(os.path.join(SRC_NEG_DIR, "*"))
    print(f"Processing {len(neg_images)} Negative Slides (Unsupervised Filter Only)...")
    
    neg_results = Parallel(n_jobs=NUM_CORES)(
        delayed(process_slide)(p, False) for p in tqdm(neg_images)
    )
    neg_kept = sum(r[1] for r in neg_results)

    print("\n" + "="*50)
    print("FINAL STATISTICS (UNSUPERVISED PROTOCOL)")
    print("="*50)
    print(f"Positive Source Patches Saved: {pos_kept}")
    print(f"Negative Source Patches Saved: {neg_kept}")
    print("="*50)

if __name__ == "__main__":
    process_dataset()