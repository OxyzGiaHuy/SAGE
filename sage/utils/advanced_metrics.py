import numpy as np
import cv2
import torch
from scipy.ndimage import distance_transform_edt as distance
from scipy.ndimage import label

def calculate_hd95_bf1(pred_mask, gt_mask):
    """
    Computes Hausdorff Distance (HD95) and Boundary F1 Score.
    Inputs: numpy arrays (H, W), binary (0/1).
    """
    # 1. Edge Case Handling
    if np.sum(gt_mask) == 0:
        return (0.0, 1.0) if np.sum(pred_mask) == 0 else (100.0, 0.0)
    if np.sum(pred_mask) == 0:
        return 100.0, 0.0

    # 2. Boundary Extraction
    pred_border = cv2.Canny(pred_mask.astype(np.uint8) * 255, 100, 200) > 0
    gt_border = cv2.Canny(gt_mask.astype(np.uint8) * 255, 100, 200) > 0

    if np.sum(pred_border) == 0 or np.sum(gt_border) == 0:
        return 100.0, 0.0

    # 3. Distance Transform & HD95
    dt_gt = distance(~gt_border)
    dt_pred = distance(~pred_border)
    
    d1 = dt_gt[pred_border]
    d2 = dt_pred[gt_border]
    
    hd95 = max(np.percentile(d1, 95), np.percentile(d2, 95))

    # 4. Boundary F1 (Tolerance = 2 pixels)
    tolerance = 2.0
    precision = np.sum(d1 <= tolerance) / (np.sum(pred_border) + 1e-6)
    recall = np.sum(d2 <= tolerance) / (np.sum(gt_border) + 1e-6)
    bf1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    return hd95, bf1

def calculate_object_dice(pred_mask, gt_mask, iou_threshold=0.5):
    """
    Calculates Object-level Dice (F1) for Instance Segmentation (GlaS).
    """
    pred_labeled, num_preds = label(pred_mask)
    gt_labeled, num_gts = label(gt_mask)

    if num_gts == 0:
        return 1.0 if num_preds == 0 else 0.0
    if num_preds == 0:
        return 0.0

    tp, fp = 0, 0
    matched_gt_ids = set()

    for i in range(1, num_preds + 1):
        pred_blob = (pred_labeled == i)
        intersection_ids = np.unique(gt_labeled[pred_blob])
        intersection_ids = intersection_ids[intersection_ids != 0]
        
        hit = False
        for gt_id in intersection_ids:
            gt_blob = (gt_labeled == gt_id)
            inter = np.logical_and(pred_blob, gt_blob).sum()
            union = np.logical_or(pred_blob, gt_blob).sum()
            iou = inter / (union + 1e-6)
            
            if iou > iou_threshold:
                hit = True
                matched_gt_ids.add(gt_id)
                break 
        
        if hit: tp += 1
        else: fp += 1

    fn = num_gts - len(matched_gt_ids)
    if tp == 0: return 0.0
    return (2 * tp) / (2 * tp + fp + fn)

def compute_dataset_specific_metrics(preds_tensor, labels_tensor, dataset_type):
    """
    Wrapper to compute batch-level average metrics based on dataset type.
    Inputs: Tensor (B, H, W)
    """
    # Convert to numpy for scipy/cv2
    preds_np = preds_tensor.cpu().numpy().astype(np.uint8)
    labels_np = labels_tensor.cpu().numpy().astype(np.uint8)
    
    batch_hd95, batch_bf1, batch_obj_dice = [], [], []

    for i in range(len(preds_np)):
        p, l = preds_np[i], labels_np[i]
        
        # All datasets get HD95 and BF1
        hd, bf1 = calculate_hd95_bf1(p, l)
        batch_hd95.append(hd)
        batch_bf1.append(bf1)
        
        # Only GlaS gets Object Dice
        if dataset_type == 'glas':
            obj_dice = calculate_object_dice(p, l)
            batch_obj_dice.append(obj_dice)

    results = {
        'hd95': np.mean(batch_hd95),
        'boundary_f1': np.mean(batch_bf1)
    }
    
    if dataset_type == 'glas':
        results['object_dice'] = np.mean(batch_obj_dice)
        
    return results