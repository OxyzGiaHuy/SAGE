"""
Visualization utilities for SAGE
Functions for visualizing predictions, expert usage, and training progress
"""

import os
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import cv2


def visualize_predictions(
    images, 
    labels, 
    predictions, 
    save_path=None, 
    num_samples=4,
    class_names=None,
    title="Predictions"
):
    """
    Visualize model predictions alongside ground truth
    
    Args:
        images (Tensor): Input images (B, C, H, W)
        labels (Tensor): Ground truth labels (B, H, W)
        predictions (Tensor): Predicted labels (B, H, W)
        save_path (str): Path to save visualization
        num_samples (int): Number of samples to visualize
        class_names (list): Names of classes for legend
        title (str): Plot title
    """
    num_samples = min(num_samples, images.size(0))
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    for idx in range(num_samples):
        # Get image (first channel if multi-channel)
        if images.dim() == 4:
            img = images[idx, 0].cpu().numpy()
        else:
            img = images[idx].cpu().numpy()
        
        label = labels[idx].cpu().numpy()
        pred = predictions[idx].cpu().numpy()
        
        # Plot input image
        axes[idx, 0].imshow(img, cmap='gray')
        axes[idx, 0].set_title('Input Image')
        axes[idx, 0].axis('off')
        
        # Plot ground truth
        axes[idx, 1].imshow(label, cmap='tab10', vmin=0, vmax=9)
        axes[idx, 1].set_title('Ground Truth')
        axes[idx, 1].axis('off')
        
        # Plot prediction
        im = axes[idx, 2].imshow(pred, cmap='tab10', vmin=0, vmax=9)
        axes[idx, 2].set_title('Prediction')
        axes[idx, 2].axis('off')
    
    # Add colorbar
    if class_names:
        cbar = plt.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal', 
                           pad=0.05, fraction=0.05)
        cbar.set_label('Classes', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"Predictions visualization saved to {save_path}")
    
    plt.close()


def visualize_expert_usage(
    expert_stats,
    save_path,
    block_names=None,
    expert_names=None,
    title="Expert Usage Statistics"
):
    """
    Visualize expert usage patterns from SAGE routers
    
    Args:
        expert_stats (dict): Expert usage statistics
        save_path (str): Path to save visualization
        block_names (list): Names of blocks/stages
        expert_names (list): Names of experts
        title (str): Plot title
    """
    if not expert_stats:
        logging.warning("No expert statistics to visualize")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Extract CNN and Transformer stats
    cnn_stats = expert_stats.get('cnn_stages', [])
    trans_stats = expert_stats.get('transformer_blocks', [])
    
    if not cnn_stats and not trans_stats:
        logging.warning("No expert usage data found")
        plt.close()
        return
    
    all_stats = cnn_stats + trans_stats
    
    if block_names is None:
        block_names = [f"CNN_{i}" for i in range(len(cnn_stats))] + \
                     [f"Trans_{i}" for i in range(len(trans_stats))]
    
    # 1. Expert usage heatmap
    usage_counts = []
    for stat in all_stats:
        router_stats = stat.get('router_stats', {})
        counts = router_stats.get('expert_usage_count', [])
        usage_counts.append(counts)
    
    if usage_counts and len(usage_counts[0]) > 0:
        usage_matrix = np.array(usage_counts).T
        
        if expert_names is None:
            num_cnn = len(cnn_stats)
            num_experts = usage_matrix.shape[0]
            expert_names = [f"CNN_{i}" for i in range(num_cnn)] + \
                          [f"Trans_{i}" for i in range(num_experts - num_cnn)]
        
        im1 = axes[0, 0].imshow(usage_matrix, cmap='YlOrRd', aspect='auto')
        axes[0, 0].set_title('Expert Usage Heatmap', fontweight='bold')
        axes[0, 0].set_xlabel('Blocks')
        axes[0, 0].set_ylabel('Experts')
        axes[0, 0].set_xticks(range(len(block_names)))
        axes[0, 0].set_xticklabels(block_names, rotation=45, ha='right')
        axes[0, 0].set_yticks(range(len(expert_names)))
        axes[0, 0].set_yticklabels(expert_names)
        plt.colorbar(im1, ax=axes[0, 0], label='Usage Count')
    
    # 2. Alpha values (main path vs expert weights)
    alpha_values = [stat.get('alpha', 0) for stat in all_stats]
    if alpha_values:
        colors = ['skyblue' if i < len(cnn_stats) else 'lightcoral' 
                 for i in range(len(alpha_values))]
        axes[0, 1].bar(block_names, alpha_values, color=colors)
        axes[0, 1].set_title('Alpha Values (Main Path Weight)', fontweight='bold')
        axes[0, 1].set_ylabel('Alpha')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='skyblue', label='CNN Stages'),
            Patch(facecolor='lightcoral', label='Transformer Blocks')
        ]
        axes[0, 1].legend(handles=legend_elements)
    
    # 3. Success rates
    success_rates = [stat.get('success_rate', 0) for stat in all_stats]
    if success_rates:
        axes[1, 0].bar(block_names, success_rates, color='mediumseagreen')
        axes[1, 0].set_title('Expert Success Rates', fontweight='bold')
        axes[1, 0].set_ylabel('Success Rate')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        axes[1, 0].axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='Target: 0.8')
        axes[1, 0].legend()
    
    # 4. Expert distribution (pie chart for total usage)
    if usage_counts and len(usage_counts[0]) > 0:
        total_usage = usage_matrix.sum(axis=1)
        if total_usage.sum() > 0:
            axes[1, 1].pie(total_usage, labels=expert_names, autopct='%1.1f%%',
                          startangle=90, colors=plt.cm.Set3.colors)
            axes[1, 1].set_title('Overall Expert Usage Distribution', fontweight='bold')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logging.info(f"Expert usage visualization saved to {save_path}")
    plt.close()


def save_training_plots(
    train_losses,
    val_losses,
    train_metrics,
    val_metrics,
    save_dir,
    learning_rates=None
):
    """
    Save comprehensive training plots
    
    Args:
        train_losses (list): Training losses per epoch
        val_losses (list): Validation losses per epoch
        train_metrics (dict): Training metrics per epoch
        val_metrics (dict): Validation metrics per epoch
        save_dir (str): Directory to save plots
        learning_rates (list): Learning rates per epoch
    """
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = list(range(1, len(train_losses) + 1))
    
    # Create comprehensive plot
    n_plots = 4 if learning_rates else 3
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
    
    # 1. Loss curves
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2, marker='o')
    axes[0, 0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2, marker='s')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Loss Curves', fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. IoU scores
    if 'iou' in train_metrics and 'iou' in val_metrics:
        axes[0, 1].plot(epochs, train_metrics['iou'], 'b-', label='Train IoU', 
                       linewidth=2, marker='o')
        axes[0, 1].plot(epochs, val_metrics['iou'], 'r-', label='Val IoU', 
                       linewidth=2, marker='s')
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('IoU', fontsize=12)
        axes[0, 1].set_title('IoU Scores', fontweight='bold')
        axes[0, 1].legend(fontsize=11)
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Dice coefficients
    if 'dice' in train_metrics and 'dice' in val_metrics:
        axes[1, 0].plot(epochs, train_metrics['dice'], 'b-', label='Train Dice',
                       linewidth=2, marker='o')
        axes[1, 0].plot(epochs, val_metrics['dice'], 'r-', label='Val Dice',
                       linewidth=2, marker='s')
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Dice Coefficient', fontsize=12)
        axes[1, 0].set_title('Dice Coefficients', fontweight='bold')
        axes[1, 0].legend(fontsize=11)
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Learning rate schedule
    if learning_rates:
        axes[1, 1].plot(epochs, learning_rates, 'g-', linewidth=2, marker='D')
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
        axes[1, 1].set_title('Learning Rate Schedule', fontweight='bold')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        # If no LR, plot pixel accuracy
        if 'pixel_acc' in train_metrics and 'pixel_acc' in val_metrics:
            axes[1, 1].plot(epochs, train_metrics['pixel_acc'], 'b-', 
                           label='Train Pixel Acc', linewidth=2, marker='o')
            axes[1, 1].plot(epochs, val_metrics['pixel_acc'], 'r-', 
                           label='Val Pixel Acc', linewidth=2, marker='s')
            axes[1, 1].set_xlabel('Epoch', fontsize=12)
            axes[1, 1].set_ylabel('Pixel Accuracy', fontsize=12)
            axes[1, 1].set_title('Pixel Accuracy', fontweight='bold')
            axes[1, 1].legend(fontsize=11)
            axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logging.info(f"Training curves saved to {save_path}")
    plt.close()


def create_colored_segmentation_mask(mask, num_classes=3, colormap='tab10'):
    """
    Create a colored visualization of segmentation mask
    
    Args:
        mask (np.ndarray): Segmentation mask (H, W)
        num_classes (int): Number of classes
        colormap (str): Matplotlib colormap name
        
    Returns:
        np.ndarray: Colored mask (H, W, 3)
    """
    cmap = plt.cm.get_cmap(colormap, num_classes)
    colored_mask = cmap(mask / (num_classes - 1))[:, :, :3]
    colored_mask = (colored_mask * 255).astype(np.uint8)
    return colored_mask


def visualize_batch_predictions(
    model,
    dataloader,
    device,
    save_dir,
    num_batches=1,
    num_classes=3,
    epoch=None
):
    """
    Visualize predictions for batches from a dataloader
    
    Args:
        model (nn.Module): Trained model
        dataloader (DataLoader): Data loader
        device (torch.device): Device to run on
        save_dir (str): Directory to save visualizations
        num_batches (int): Number of batches to visualize
        num_classes (int): Number of segmentation classes
        epoch (int): Current epoch number (for filename)
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(images)
            predictions = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            
            # Create filename
            filename = f'batch_{batch_idx}'
            if epoch is not None:
                filename = f'epoch_{epoch}_{filename}'
            filename += '.png'
            
            save_path = os.path.join(save_dir, filename)
            
            # Visualize
            visualize_predictions(
                images, labels, predictions,
                save_path=save_path,
                num_samples=min(4, images.size(0)),
                title=f'Batch {batch_idx}' + (f' - Epoch {epoch}' if epoch else '')
            )
    
    model.train()
