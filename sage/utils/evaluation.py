"""
Evaluation utilities for SAGE
Functions for model evaluation and metrics calculation
"""

import os
import logging
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import calculate_all_metrics


def evaluate_model(
    model,
    dataloader,
    device,
    num_classes,
    verbose=True
):
    """
    Evaluate model on a dataset
    
    Args:
        model (nn.Module): Model to evaluate
        dataloader (DataLoader): Data loader
        device (torch.device): Device to run on
        num_classes (int): Number of classes
        verbose (bool): Whether to print progress
        
    Returns:
        dict: Average metrics across dataset
    """
    model.eval()
    
    metrics_sum = defaultdict(float)
    metrics_list = defaultdict(list)
    num_samples = 0
    
    with torch.no_grad():
        iterator = tqdm(dataloader, desc="Evaluating") if verbose else dataloader
        
        for batch in iterator:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(images)
            predictions = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            
            # Calculate metrics for each sample in batch
            for i in range(images.size(0)):
                pred = predictions[i:i+1]
                label = labels[i:i+1]
                
                metrics = calculate_all_metrics(pred, label, num_classes)
                
                metrics_sum['pixel_accuracy'] += metrics['pixel_accuracy']
                metrics_sum['mean_iou'] += metrics['mean_iou']
                metrics_sum['mean_dice'] += metrics['mean_dice']
                
                metrics_list['pixel_accuracy'].append(metrics['pixel_accuracy'])
                metrics_list['mean_iou'].append(metrics['mean_iou'])
                metrics_list['mean_dice'].append(metrics['mean_dice'])
                
                # Store per-class metrics
                for cls_idx in range(num_classes):
                    metrics_sum[f'iou_class_{cls_idx}'] += metrics['class_iou'][cls_idx]
                    metrics_sum[f'dice_class_{cls_idx}'] += metrics['class_dice'][cls_idx]
                
                num_samples += 1
    
    # Calculate averages
    avg_metrics = {}
    for key, value in metrics_sum.items():
        avg_metrics[key] = value / num_samples
    
    # Calculate std dev
    std_metrics = {}
    for key, value_list in metrics_list.items():
        std_metrics[f'{key}_std'] = np.std(value_list)
    
    avg_metrics.update(std_metrics)
    
    model.train()
    
    return avg_metrics


def calculate_metrics_batch(
    predictions,
    labels,
    num_classes
):
    """
    Calculate metrics for a batch
    
    Args:
        predictions (Tensor): Predicted labels (B, H, W)
        labels (Tensor): Ground truth labels (B, H, W)
        num_classes (int): Number of classes
        
    Returns:
        dict: Average metrics for the batch
    """
    batch_size = predictions.size(0)
    metrics_sum = defaultdict(float)
    
    for i in range(batch_size):
        pred = predictions[i:i+1]
        label = labels[i:i+1]
        
        metrics = calculate_all_metrics(pred, label, num_classes)
        
        metrics_sum['pixel_accuracy'] += metrics['pixel_accuracy']
        metrics_sum['mean_iou'] += metrics['mean_iou']
        metrics_sum['mean_dice'] += metrics['mean_dice']
        
        for cls_idx in range(num_classes):
            metrics_sum[f'iou_class_{cls_idx}'] += metrics['class_iou'][cls_idx]
            metrics_sum[f'dice_class_{cls_idx}'] += metrics['class_dice'][cls_idx]
    
    # Calculate averages
    avg_metrics = {key: value / batch_size for key, value in metrics_sum.items()}
    
    return avg_metrics


def evaluate_per_sample(
    model,
    dataloader,
    device,
    num_classes,
    save_dir=None
):
    """
    Evaluate model and get per-sample metrics
    
    Args:
        model (nn.Module): Model to evaluate
        dataloader (DataLoader): Data loader with case_name in batch
        device (torch.device): Device to run on
        num_classes (int): Number of classes
        save_dir (str): Directory to save results (optional)
        
    Returns:
        pd.DataFrame: Per-sample metrics
    """
    model.eval()
    
    results = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Per-sample evaluation"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            case_names = batch.get('case_name', [f'sample_{i}' for i in range(images.size(0))])
            
            # Forward pass
            outputs = model(images)
            predictions = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            
            # Calculate metrics for each sample
            for i in range(images.size(0)):
                pred = predictions[i:i+1]
                label = labels[i:i+1]
                case_name = case_names[i] if isinstance(case_names, (list, tuple)) else case_names
                
                metrics = calculate_all_metrics(pred, label, num_classes)
                
                result = {
                    'case_name': case_name,
                    'pixel_accuracy': metrics['pixel_accuracy'],
                    'mean_iou': metrics['mean_iou'],
                    'mean_dice': metrics['mean_dice'],
                }
                
                # Add per-class metrics
                for cls_idx in range(num_classes):
                    result[f'iou_class_{cls_idx}'] = metrics['class_iou'][cls_idx]
                    result[f'dice_class_{cls_idx}'] = metrics['class_dice'][cls_idx]
                
                results.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        csv_path = os.path.join(save_dir, 'per_sample_metrics.csv')
        df.to_csv(csv_path, index=False)
        logging.info(f"Per-sample metrics saved to {csv_path}")
    
    model.train()
    
    return df


def generate_evaluation_report(
    metrics_df,
    save_path,
    class_names=None
):
    """
    Generate comprehensive evaluation report
    
    Args:
        metrics_df (pd.DataFrame): Per-sample metrics dataframe
        save_path (str): Path to save report
        class_names (list): Names of classes
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Overall statistics
        f.write("Overall Statistics:\n")
        f.write("-"*80 + "\n")
        
        main_metrics = ['pixel_accuracy', 'mean_iou', 'mean_dice']
        for metric in main_metrics:
            if metric in metrics_df.columns:
                mean_val = metrics_df[metric].mean()
                std_val = metrics_df[metric].std()
                min_val = metrics_df[metric].min()
                max_val = metrics_df[metric].max()
                
                f.write(f"{metric:20s}: {mean_val:.4f} ± {std_val:.4f} "
                       f"(min: {min_val:.4f}, max: {max_val:.4f})\n")
        
        f.write("\n")
        
        # Per-class statistics
        f.write("Per-Class Statistics:\n")
        f.write("-"*80 + "\n")
        
        # Detect number of classes
        iou_cols = [col for col in metrics_df.columns if col.startswith('iou_class_')]
        num_classes = len(iou_cols)
        
        for cls_idx in range(num_classes):
            class_name = class_names[cls_idx] if class_names else f"Class {cls_idx}"
            
            f.write(f"\n{class_name}:\n")
            
            iou_col = f'iou_class_{cls_idx}'
            dice_col = f'dice_class_{cls_idx}'
            
            if iou_col in metrics_df.columns:
                iou_mean = metrics_df[iou_col].mean()
                iou_std = metrics_df[iou_col].std()
                f.write(f"  IoU:  {iou_mean:.4f} ± {iou_std:.4f}\n")
            
            if dice_col in metrics_df.columns:
                dice_mean = metrics_df[dice_col].mean()
                dice_std = metrics_df[dice_col].std()
                f.write(f"  Dice: {dice_mean:.4f} ± {dice_std:.4f}\n")
        
        f.write("\n")
        f.write("="*80 + "\n")
        
        # Best and worst samples
        f.write("\nTop 5 Best Samples (by mIoU):\n")
        f.write("-"*80 + "\n")
        top_5 = metrics_df.nlargest(5, 'mean_iou')
        for idx, row in top_5.iterrows():
            f.write(f"{row['case_name']:30s} - mIoU: {row['mean_iou']:.4f}, "
                   f"mDice: {row['mean_dice']:.4f}\n")
        
        f.write("\nTop 5 Worst Samples (by mIoU):\n")
        f.write("-"*80 + "\n")
        bottom_5 = metrics_df.nsmallest(5, 'mean_iou')
        for idx, row in bottom_5.iterrows():
            f.write(f"{row['case_name']:30s} - mIoU: {row['mean_iou']:.4f}, "
                   f"mDice: {row['mean_dice']:.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    logging.info(f"Evaluation report saved to {save_path}")


def compare_model_predictions(
    model1,
    model2,
    dataloader,
    device,
    num_classes,
    model1_name="Model 1",
    model2_name="Model 2"
):
    """
    Compare predictions from two models
    
    Args:
        model1 (nn.Module): First model
        model2 (nn.Module): Second model
        dataloader (DataLoader): Data loader
        device (torch.device): Device to run on
        num_classes (int): Number of classes
        model1_name (str): Name for first model
        model2_name (str): Name for second model
        
    Returns:
        dict: Comparison results
    """
    logging.info(f"Comparing {model1_name} vs {model2_name}")
    
    # Evaluate both models
    metrics1 = evaluate_model(model1, dataloader, device, num_classes, verbose=True)
    metrics2 = evaluate_model(model2, dataloader, device, num_classes, verbose=True)
    
    # Calculate differences
    comparison = {
        'model1': model1_name,
        'model2': model2_name,
        'metrics1': metrics1,
        'metrics2': metrics2,
        'differences': {}
    }
    
    for key in ['pixel_accuracy', 'mean_iou', 'mean_dice']:
        if key in metrics1 and key in metrics2:
            diff = metrics2[key] - metrics1[key]
            comparison['differences'][key] = diff
            
            logging.info(f"{key}: {model1_name}={metrics1[key]:.4f}, "
                        f"{model2_name}={metrics2[key]:.4f}, "
                        f"diff={diff:+.4f}")
    
    return comparison
