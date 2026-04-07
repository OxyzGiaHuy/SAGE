"""
Generate WSI overlay visualizations from SAGE model predictions
Combines full WSI reconstruction with color-coded overlay (Green=TP, Red=FP/FN)

Based on evit_unet/wsi_overlay.py structure for consistency
"""

import os
import sys
import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import cv2
import numpy as np
import torch
import glob
from tqdm import tqdm
from collections import defaultdict
import re
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sage.networks.convnext_transformer_unet import create_convnext_transformer_unet


class MedicalDataset(Dataset):
    """Dataset for loading test patches with transforms"""
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            aug = self.transform(image=img)
            img = aug['image']
            
        return img, os.path.basename(img_path)


def get_mask_path(img_fname, mask_dir):
    """Robustly find the matching mask patch for a given image patch."""
    name, ext = os.path.splitext(img_fname)
    candidates = [
        f"{name}_mask{ext}",
        f"{name}_mask.png",
        f"{name}_mask.jpg",
        f"{name}_mask.bmp",
        f"{name}_mask.tif",
        img_fname,
        f"{name}.png"
    ]
    for cand in candidates:
        p = os.path.join(mask_dir, cand)
        if os.path.exists(p):
            return p
    return None


def load_model(config_path, checkpoint_path, device):
    """Load trained SAGE model from checkpoint"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Loading SAGE model from: {checkpoint_path}")
    
    use_sage = config['model']['use_sage']
    sage_config = config.get('sage', {}) if use_sage else None
    
    model = create_convnext_transformer_unet(
        num_classes=config['model']['num_classes'],
        img_size=config['model']['img_size'],
        convnext_variant=config['model'].get('convnext_variant', 'convnext_base'),
        vit_variant=config['model'].get('vit_variant', 'vit_base_patch32_224'),
        num_transformer_layers=config['model'].get('num_transformer_layers', 12),
        freeze_encoder=False,
        freeze_transformer=False,
        sage_config=sage_config,
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # Remove 'module.' prefix if present (from DataParallel/DDP)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    
    print("✓ Model loaded successfully!")
    return model, config


def parse_coords(filename):
    """Extract x, y coordinates from filename (e.g., testA_1_x512_y1024.jpg)"""
    match_x = re.search(r"_x(\d+)", filename)
    match_y = re.search(r"_y(\d+)", filename)
    x = int(match_x.group(1)) if match_x else -1
    y = int(match_y.group(1)) if match_y else -1
    return x, y


def extract_wsi_id(filename):
    """Extract WSI ID from patch filename"""
    # Pattern: testA_1_patch_level0_x512_y1024.jpg -> testA_1
    match = re.search(r"(.+)_patch_level", filename)
    if match:
        return match.group(1)
    # Fallback: take everything before _x
    match = re.search(r"(.+)_x\d+", filename)
    return match.group(1) if match else filename


def create_overlay(original_rgb, pred_mask, gt_mask, alpha=0.5):
    """
    Create overlay visualization:
    - Green: True Positive (correct prediction)
    - Red: False Positive / False Negative (incorrect)
    """
    tp = (pred_mask == 1) & (gt_mask == 1)
    fp = (pred_mask == 1) & (gt_mask == 0)
    fn = (pred_mask == 0) & (gt_mask == 1)
    
    overlay = np.zeros_like(original_rgb, dtype=np.float32)
    overlay[tp] = [0, 1, 0]      # Green for TP
    overlay[fp | fn] = [1, 0, 0] # Red for FP/FN
    
    # Blend with original
    if len(original_rgb.shape) == 2:
        original_rgb = cv2.cvtColor(original_rgb, cv2.COLOR_GRAY2RGB)
    
    blended = (1 - alpha) * (original_rgb / 255.0) + alpha * overlay
    return (blended * 255).astype(np.uint8)


def run_wsi_inference(model, test_img_dir, test_mask_dir, config, device, output_dir):
    """
    Run inference on all patches and reconstruct WSI with overlays
    Uses DataLoader for efficient batch processing
    """
    
    PHYSICAL_PATCH_SIZE = 1536  # WSI patch size at original resolution
    IMG_SIZE = config['data']['img_size']  # Model input size (512)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Find all test patches and group by WSI
    print("\n" + "="*80)
    print("GROUPING PATCHES BY WSI...")
    print("="*80)
    
    all_patches = glob.glob(os.path.join(test_img_dir, "*.[jp][pn][g]"))
    wsi_pattern = re.compile(r"(.+)_patch_level")
    
    patches_by_wsi = defaultdict(list)
    for p in all_patches:
        match = wsi_pattern.search(os.path.basename(p))
        if match:
            wsi_id = match.group(1)
            patches_by_wsi[wsi_id].append(p)
    
    selected_wsis = list(patches_by_wsi.keys())
    print(f"Found {len(selected_wsis)} WSIs with {len(all_patches)} total patches")
    
    # 2. Setup transforms (match training normalization)
    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])
    
    # 3. Process each WSI
    print("\n" + "="*80)
    print("PROCESSING WSIs...")
    print("="*80)
    
    for wsi_id in tqdm(selected_wsis, desc="WSI Processing"):
        patch_paths = patches_by_wsi[wsi_id]
        dataset = MedicalDataset(patch_paths, transform=transform)
        loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
        
        # Determine WSI canvas dimensions from patch coordinates
        max_x, max_y = 0, 0
        for p in patch_paths:
            fname = os.path.basename(p)
            match_x = re.search(r"_x(\d+)", fname)
            match_y = re.search(r"_y(\d+)", fname)
            if match_x and match_y:
                max_x = max(max_x, int(match_x.group(1)))
                max_y = max(max_y, int(match_y.group(1)))
        
        H = max_y + PHYSICAL_PATCH_SIZE
        W = max_x + PHYSICAL_PATCH_SIZE
        
        # Initialize canvas maps
        accum_map = np.zeros((H, W), dtype=np.float32)
        count_map = np.zeros((H, W), dtype=np.float32)
        gt_map = np.zeros((H, W), dtype=np.float32)
        
        # Run batch inference
        with torch.no_grad():
            for imgs, filenames in loader:
                imgs = imgs.to(device)
                
                # Forward pass
                logits = model(imgs)
                
                # Handle different output formats (dict, tuple, tensor)
                if isinstance(logits, dict):
                    logits = logits.get('logits', logits.get('out'))
                elif isinstance(logits, (list, tuple)):
                    logits = logits[0]
                
                # Get probability for foreground class (sigmoid for binary, softmax for multi-class)
                num_classes = config['model']['num_classes']
                if num_classes == 2:
                    probs = torch.softmax(logits, dim=1)[:, 1]  # Class 1 probability
                else:
                    probs = torch.sigmoid(logits[:, 0])
                
                probs = probs.cpu().numpy()
                
                # Stitch predictions and ground truth
                for i, fname in enumerate(filenames):
                    match_x = re.search(r"_x(\d+)", fname)
                    match_y = re.search(r"_y(\d+)", fname)
                    if not match_x or not match_y:
                        continue
                    
                    x, y = int(match_x.group(1)), int(match_y.group(1))
                    
                    # Resize prediction to physical patch size
                    prob_np = probs[i]
                    prob_resized = cv2.resize(prob_np, (PHYSICAL_PATCH_SIZE, PHYSICAL_PATCH_SIZE))
                    
                    # Accumulate prediction
                    y_end = min(y + PHYSICAL_PATCH_SIZE, H)
                    x_end = min(x + PHYSICAL_PATCH_SIZE, W)
                    h_len = y_end - y
                    w_len = x_end - x
                    
                    if h_len > 0 and w_len > 0:
                        accum_map[y:y_end, x:x_end] += prob_resized[:h_len, :w_len]
                        count_map[y:y_end, x:x_end] += 1.0
                    
                    # Load and stitch ground truth
                    mask_path = get_mask_path(fname, test_mask_dir)
                    if mask_path:
                        gt_patch = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        if gt_patch is not None:
                            if gt_patch.shape[:2] != (PHYSICAL_PATCH_SIZE, PHYSICAL_PATCH_SIZE):
                                gt_patch = cv2.resize(
                                    gt_patch, 
                                    (PHYSICAL_PATCH_SIZE, PHYSICAL_PATCH_SIZE),
                                    interpolation=cv2.INTER_NEAREST
                                )
                            gt_binary = (gt_patch > 0).astype(np.float32)
                            gt_map[y:y_end, x:x_end] = np.maximum(
                                gt_map[y:y_end, x:x_end],
                                gt_binary[:h_len, :w_len]
                            )
        
        # Average overlapping predictions
        mask_area = count_map > 0
        accum_map[mask_area] /= count_map[mask_area]
        
        # Threshold to binary masks
        pred_mask = (accum_map > 0.5).astype(np.uint8)
        gt_mask = (gt_map > 0.5).astype(np.uint8)
        
        # Calculate metrics
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        iou = intersection / (union + 1e-6) if union > 0 else 1.0
        dice = 2 * intersection / (pred_mask.sum() + gt_mask.sum() + 1e-6)
        
        # Create color overlay (Green=TP, Red=FP/FN)
        bgr_overlay = np.zeros((H, W, 3), dtype=np.uint8)
        
        # True Positive -> Green
        tp_mask = (pred_mask == 1) & (gt_mask == 1)
        bgr_overlay[tp_mask] = [0, 255, 0]
        
        # False Positive & False Negative -> Red
        fp_mask = (pred_mask == 1) & (gt_mask == 0)
        fn_mask = (pred_mask == 0) & (gt_mask == 1)
        bgr_overlay[fp_mask] = [0, 0, 255]
        bgr_overlay[fn_mask] = [0, 0, 255]
        
        # Save outputs (downscale to 10% to save disk space)
        base_name = f"{wsi_id}_iou{iou:.3f}_dice{dice:.3f}"
        cv2.imwrite(
            os.path.join(output_dir, f"{base_name}_overlay.jpg"),
            cv2.resize(bgr_overlay, (0, 0), fx=0.1, fy=0.1)
        )
        cv2.imwrite(
            os.path.join(output_dir, f"{base_name}_pred.png"),
            pred_mask * 255
        )
        cv2.imwrite(
            os.path.join(output_dir, f"{base_name}_gt.png"),
            gt_mask * 255
        )
        
        print(f"  {wsi_id:30s} IoU={iou:.4f} Dice={dice:.4f} Size=({H}x{W})")
    
    print(f"\n✓ WSI overlays saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate WSI overlays from SAGE predictions")
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--test_img_dir", required=True, help="Directory containing test image patches")
    parser.add_argument("--test_mask_dir", required=True, help="Directory containing test mask patches")
    parser.add_argument("--output_dir", default="wsi_overlays", help="Output directory for overlays")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*80)
    print("SAGE WSI OVERLAY GENERATION")
    print("="*80)
    print(f"Device      : {device}")
    print(f"Config      : {args.config}")
    print(f"Checkpoint  : {args.checkpoint}")
    print(f"Test Images : {args.test_img_dir}")
    print(f"Test Masks  : {args.test_mask_dir}")
    print(f"Output Dir  : {args.output_dir}")
    print("="*80)
    
    # Verify files exist
    if not os.path.exists(args.config):
        print(f"ERROR: Config file not found: {args.config}")
        return
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        return
    if not os.path.isdir(args.test_img_dir):
        print(f"ERROR: Test image directory not found: {args.test_img_dir}")
        return
    if not os.path.isdir(args.test_mask_dir):
        print(f"ERROR: Test mask directory not found: {args.test_mask_dir}")
        return
    
    # Load model
    model, config = load_model(args.config, args.checkpoint, device)
    
    # Run inference and generate overlays
    run_wsi_inference(
        model,
        args.test_img_dir,
        args.test_mask_dir,
        config,
        device,
        args.output_dir
    )
    
    print("\n" + "="*80)
    print("✓ COMPLETE!")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  - {{wsi_id}}_iouX.XXX_diceX.XXX_overlay.jpg  (Green=TP, Red=FP/FN)")
    print(f"  - {{wsi_id}}_iouX.XXX_diceX.XXX_pred.png     (prediction mask)")
    print(f"  - {{wsi_id}}_iouX.XXX_diceX.XXX_gt.png       (ground truth)")
    print()


if __name__ == "__main__":
    main()
