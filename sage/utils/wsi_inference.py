import os
import glob
import re
import cv2
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from .advanced_metrics import calculate_hd95_bf1

class WSIEvaluator:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        # Each WSI patch is 1536×1536 pixels on the WSI (physical space), stride=512.
        # Filenames encode the top-left corner in WSI coordinates (e.g. _x512_y1024).
        # The model receives a 512×512 downscaled version; outputs are upscaled back to 1536×1536
        # before being placed on the canvas at the correct coordinate.
        self.PHYSICAL_PATCH_SIZE = 1536
        self.patch_size = config['data']['img_size']   # 512 — model input size
        self.num_classes = config['model']['num_classes']

        # Raw WSI image folders (for loading ground-truth masks at original resolution)
        wsi_cfg = config.get('wsi', {})
        self.raw_pos_path = wsi_cfg.get('raw_pos_path', None)
        self.raw_neg_path = wsi_cfg.get('raw_neg_path', None)

        # Pre-scan raw files so lookup is O(1)
        self.pos_files = set()
        self.neg_files = set()
        if self.raw_pos_path and os.path.isdir(self.raw_pos_path):
            self.pos_files = {f for f in os.listdir(self.raw_pos_path) if f.endswith('.jpg')}
            print(f"[WSI] Scanned {len(self.pos_files)} files in POS folder")
        if self.raw_neg_path and os.path.isdir(self.raw_neg_path):
            self.neg_files = {f for f in os.listdir(self.raw_neg_path) if f.endswith('.jpg')}
            print(f"[WSI] Scanned {len(self.neg_files)} files in NEG folder")

    def get_raw_ground_truth(self, wsi_id):
        """
        Load ground-truth mask from actual raw WSI files.
        Returns: (gt_mask uint8, H, W, type_str)
        - POS slide: loads the paired *_mask.jpg for real GT
        - NEG slide: zero mask at image dimensions
        - UNK: no raw file found, falls back to patch-stitched GT
        """
        if not self.raw_pos_path and not self.raw_neg_path:
            return None, 0, 0, "UNK"  # raw paths not configured

        img_name = f"{wsi_id}.jpg"

        if img_name in self.pos_files:
            # Try exact mask name first, then glob
            mask_name = f"{wsi_id}_mask.jpg"
            mask_path = os.path.join(self.raw_pos_path, mask_name)
            if not os.path.exists(mask_path):
                cands = glob.glob(os.path.join(self.raw_pos_path, f"{wsi_id}*_mask.jpg"))
                if cands:
                    mask_path = cands[0]
            if os.path.exists(mask_path):
                gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if gt is not None:
                    return (gt > 0).astype(np.uint8), gt.shape[0], gt.shape[1], "POS"

        if img_name in self.neg_files:
            img_path = os.path.join(self.raw_neg_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                return np.zeros(img.shape[:2], dtype=np.uint8), img.shape[0], img.shape[1], "NEG"

        return None, 0, 0, "UNK"

    def parse_coords(self, filename):
        match_x = re.search(r"_x(\d+)", filename)
        match_y = re.search(r"_y(\d+)", filename)
        x = int(match_x.group(1)) if match_x else -1
        y = int(match_y.group(1)) if match_y else -1
        return x, y

    def run_inference(self, model, dataloader):
        """
        Runs WSI reconstruction and evaluation.

        Ground-truth strategy (mirrors SegFormer reference code):
        - If raw_pos_path / raw_neg_path are configured in config['wsi']:
            load actual raw WSI mask → correct canvas dimensions, POS/NEG distinction
        - Otherwise: stitch GT from patch labels (fallback, less accurate canvas size)
        """
        model.eval()

        wsi_pattern = re.compile(r"(.+)_patch_level")
        # Store only prob per patch; GT is loaded from raw file separately
        results_buffer = defaultdict(list)  # {wsi_id: [{x, y, prob float16}]}
        # Keep patch GT as fallback when raw files are not available
        gt_patch_buffer = defaultdict(list)  # {wsi_id: [{x, y, gt uint8}]}

        print("--> [WSI] Running Patch Inference...")
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Patch Inference"):
                images     = batch['image'].to(self.device)
                labels     = batch['label']        # CPU
                case_names = batch['case_name']

                outputs = model(images)
                if isinstance(outputs, dict):
                    outputs = outputs['logits']
                # Softmax → foreground probability channel (class 1)
                probs = torch.softmax(outputs, dim=1)[:, 1, :, :]  # [B, H, W]

                for i, name in enumerate(case_names):
                    base  = os.path.basename(name)
                    match = wsi_pattern.search(base)
                    if not match:
                        continue
                    wsi_id = match.group(1)
                    x, y   = self.parse_coords(base)
                    if x == -1:
                        continue

                    results_buffer[wsi_id].append({
                        'x': x, 'y': y,
                        'prob': probs[i].cpu().numpy().astype(np.float16),
                    })
                    gt_patch_buffer[wsi_id].append({
                        'x': x, 'y': y,
                        'gt': labels[i].numpy().astype(np.uint8),
                    })

        # ── Stitch & Evaluate ─────────────────────────────────────────────
        use_raw_gt  = bool(self.raw_pos_path or self.raw_neg_path)
        per_type    = defaultdict(lambda: defaultdict(list))  # type → metric → [values]
        all_metrics = defaultdict(list)

        print(f"--> [WSI] Stitching {len(results_buffer)} slides "
              f"(GT source: {'raw files' if use_raw_gt else 'patch labels'})...")

        for wsi_id, patches in tqdm(results_buffer.items(), desc="Stitching"):

            # ── Ground-truth & canvas size ─────────────────────────────────
            wsi_type = "UNK"
            if use_raw_gt:
                gt_raw, H, W, wsi_type = self.get_raw_ground_truth(wsi_id)
            else:
                gt_raw = None

            if gt_raw is None:
                # Fallback: reconstruct GT from patch labels & infer canvas from max coords
                wsi_type = "UNK"
                max_x    = max(p['x'] for p in patches) + self.PHYSICAL_PATCH_SIZE
                max_y    = max(p['y'] for p in patches) + self.PHYSICAL_PATCH_SIZE
                H, W     = max_y, max_x
            else:
                H, W = gt_raw.shape[0], gt_raw.shape[1]

            # ── Build prediction canvas ────────────────────────────────────
            accum_map = np.zeros((H, W), dtype=np.float32)
            count_map = np.zeros((H, W), dtype=np.float32)

            for p in patches:
                prob_resized = cv2.resize(
                    p['prob'].astype(np.float32),
                    (self.PHYSICAL_PATCH_SIZE, self.PHYSICAL_PATCH_SIZE),
                    interpolation=cv2.INTER_LINEAR,
                )
                y_p, x_p = p['y'], p['x']
                h_p, w_p = prob_resized.shape
                h_end = min(y_p + h_p, H)
                w_end = min(x_p + w_p, W)
                h_len = h_end - y_p
                w_len = w_end - x_p
                if h_len <= 0 or w_len <= 0:
                    continue
                accum_map[y_p:h_end, x_p:w_end] += prob_resized[:h_len, :w_len]
                count_map[y_p:h_end, x_p:w_end] += 1.0

            mask_area = count_map > 0
            accum_map[mask_area] /= count_map[mask_area]
            final_pred = (accum_map > 0.5).astype(np.uint8)

            # ── Build GT canvas ────────────────────────────────────────────
            if gt_raw is not None:
                final_gt = gt_raw
            else:
                gt_map = np.zeros((H, W), dtype=np.uint8)
                for p in gt_patch_buffer[wsi_id]:
                    gt_resized = cv2.resize(
                        p['gt'],
                        (self.PHYSICAL_PATCH_SIZE, self.PHYSICAL_PATCH_SIZE),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    y_p, x_p = p['y'], p['x']
                    h_end = min(y_p + self.PHYSICAL_PATCH_SIZE, H)
                    w_end = min(x_p + self.PHYSICAL_PATCH_SIZE, W)
                    h_len = h_end - y_p
                    w_len = w_end - x_p
                    if h_len > 0 and w_len > 0:
                        gt_map[y_p:h_end, x_p:w_end] = gt_resized[:h_len, :w_len]
                final_gt = gt_map

            # ── Per-WSI metrics ────────────────────────────────────────────
            intersection = np.logical_and(final_pred, final_gt).sum()
            union        = np.logical_or(final_pred, final_gt).sum()

            iou  = 1.0 if union == 0 else intersection / (union + 1e-6)
            dice = 1.0 if (final_pred.sum() + final_gt.sum()) == 0 \
                       else 2 * intersection / (final_pred.sum() + final_gt.sum() + 1e-6)
            acc  = (final_pred == final_gt).mean()
            hd95, bf1 = calculate_hd95_bf1(final_pred, final_gt)

            for k, v in zip(['iou', 'dice', 'acc', 'hd95', 'bf1'],
                             [iou,   dice,   acc,   hd95,   bf1]):
                all_metrics[k].append(v)
                per_type[wsi_type][k].append(v)

        # ── Aggregate results ──────────────────────────────────────────────
        final_results = {k: float(np.mean(v)) if v else 0.0
                         for k, v in all_metrics.items()}

        print("\n=== WSI RESULTS ===")
        for t, t_metrics in sorted(per_type.items()):
            n = len(t_metrics.get('iou', []))
            print(f"  [{t}] n={n}  "
                  f"IoU={np.mean(t_metrics['iou']):.4f}  "
                  f"Dice={np.mean(t_metrics['dice']):.4f}  "
                  f"BF1={np.mean(t_metrics['bf1']):.4f}  "
                  + (f"HD95={np.mean(t_metrics['hd95']):.4f}" if t == 'POS' else ""))
        print("-" * 40)
        print(f"  OVERALL  n={len(all_metrics['iou'])}  "
              f"IoU={final_results['iou']:.4f}  "
              f"Dice={final_results['dice']:.4f}  "
              f"BF1={final_results['bf1']:.4f}  "
              f"HD95={final_results['hd95']:.4f}")
        print("=" * 40)

        # Attach per-type breakdown to results dict for caller to log
        final_results['per_type'] = {
            t: {k: float(np.mean(v)) for k, v in m.items() if v}
            for t, m in per_type.items()
        }
        return final_results
