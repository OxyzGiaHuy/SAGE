"""
Metrics for SAGE
Loss functions and evaluation metrics for medical image segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


class SimpleDiceLoss(nn.Module):
    """
    Dice Loss for multi-class segmentation
    Compatible with modular architecture
    """
    
    def __init__(self, n_classes, softmax=True, smooth=1e-5):
        """
        Args:
            n_classes (int): Number of segmentation classes
            softmax (bool): Whether to apply softmax to inputs
            smooth (float): Smoothing factor to avoid division by zero
        """
        super(SimpleDiceLoss, self).__init__()
        self.n_classes = n_classes
        self.softmax = softmax
        self.smooth = smooth

    def _one_hot_encoder(self, input_tensor):
        """Convert class indices to one-hot encoding"""
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        """Calculate Dice loss for a single class"""
        target = target.float()
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + self.smooth) / (z_sum + y_sum + self.smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        """
        Args:
            inputs (Tensor): Model predictions (B, C, H, W)
            target (Tensor): Ground truth labels (B, H, W)
            weight (list): Per-class weights
            softmax (bool): Whether to apply softmax
            
        Returns:
            Tensor: Dice loss value
        """
        if softmax or self.softmax:
            inputs = F.softmax(inputs, dim=1)
        
        num_classes_pred = inputs.shape[1]
        if self.n_classes != num_classes_pred:
            logging.warning(
                f"SimpleDiceLoss n_classes mismatch: configured for {self.n_classes}, "
                f"but model output has {num_classes_pred}. Using model's value."
            )
            self.n_classes = num_classes_pred
        
        target = self._one_hot_encoder(target)
        
        if weight is None:
            weight = [1] * self.n_classes
            
        assert inputs.size() == target.size(), \
            f'predict {inputs.size()} & target {target.size()} shape do not match'
        
        loss = 0.0
        for i in range(self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            loss += dice * weight[i]
            
        return loss / self.n_classes


class CombinedLoss(nn.Module):
    """
    Combined loss: Dice Loss + Cross Entropy
    Recommended for medical image segmentation
    """
    
    def __init__(self, n_classes, dice_weight=0.5, ce_weight=0.5):
        """
        Args:
            n_classes (int): Number of classes
            dice_weight (float): Weight for Dice loss
            ce_weight (float): Weight for Cross Entropy loss
        """
        super(CombinedLoss, self).__init__()
        self.n_classes = n_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        
        self.dice_loss = SimpleDiceLoss(n_classes, softmax=False)
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, inputs, target):
        """
        Args:
            inputs (Tensor): Model predictions (B, C, H, W)
            target (Tensor): Ground truth labels (B, H, W)
            
        Returns:
            Tensor: Combined loss value
        """
        dice = self.dice_loss(inputs, target, softmax=True)
        ce = self.ce_loss(inputs, target)
        
        return self.dice_weight * dice + self.ce_weight * ce


def calculate_pixel_accuracy(pred_tensor, target_tensor):
    """
    Calculate pixel-wise accuracy
    
    Args:
        pred_tensor (Tensor): Predicted labels (B, H, W) or (H, W)
        target_tensor (Tensor): Ground truth labels (B, H, W) or (H, W)
        
    Returns:
        float: Pixel accuracy [0, 1]
    """
    try:
        # Ensure same device
        if pred_tensor.device != target_tensor.device:
            target_tensor = target_tensor.to(pred_tensor.device)
        
        pred_flat = pred_tensor.flatten()
        target_flat = target_tensor.flatten()
        
        correct = torch.sum(pred_flat == target_flat).item()
        total = target_flat.numel()
        
        return correct / total if total > 0 else 0.0
        
    except Exception as e:
        logging.error(f"Error in calculate_pixel_accuracy: {e}")
        return 0.0


def calculate_iou(pred_tensor, target_tensor, num_classes):
    """
    Calculate Intersection over Union (IoU) for each class
    
    Args:
        pred_tensor (Tensor): Predicted labels (B, H, W) or (H, W)
        target_tensor (Tensor): Ground truth labels (B, H, W) or (H, W)
        num_classes (int): Number of classes
        
    Returns:
        tuple: (list of per-class IoU, mean IoU)
    """
    try:
        # Ensure same device
        if pred_tensor.device != target_tensor.device:
            target_tensor = target_tensor.to(pred_tensor.device)
        
        pred_flat = pred_tensor.flatten()
        target_flat = target_tensor.flatten()
        
        ious = []
        for cls in range(num_classes):
            pred_cls = (pred_flat == cls)
            target_cls = (target_flat == cls)
            
            intersection = torch.sum(pred_cls & target_cls).item()
            union = torch.sum(pred_cls | target_cls).item()
            
            if union == 0:
                # If class doesn't appear in either prediction or target
                iou = 1.0 if intersection == 0 else 0.0
            else:
                iou = intersection / union
                
            ious.append(iou)
        
        mean_iou = sum(ious) / len(ious)
        return ious, mean_iou
        
    except Exception as e:
        logging.error(f"Error in calculate_iou: {e}")
        return [0.0] * num_classes, 0.0


def calculate_dice_coefficient(pred_tensor, target_tensor, num_classes, smooth=1e-5):
    """
    Calculate Dice coefficient for each class
    
    Args:
        pred_tensor (Tensor): Predicted labels (B, H, W) or (H, W)
        target_tensor (Tensor): Ground truth labels (B, H, W) or (H, W)
        num_classes (int): Number of classes
        smooth (float): Smoothing factor
        
    Returns:
        tuple: (list of per-class Dice, mean Dice)
    """
    try:
        # Ensure same device
        if pred_tensor.device != target_tensor.device:
            target_tensor = target_tensor.to(pred_tensor.device)
        
        pred_flat = pred_tensor.flatten()
        target_flat = target_tensor.flatten()
        
        dice_scores = []
        for cls in range(num_classes):
            pred_cls = (pred_flat == cls).float()
            target_cls = (target_flat == cls).float()
            
            intersection = torch.sum(pred_cls * target_cls).item()
            pred_sum = torch.sum(pred_cls).item()
            target_sum = torch.sum(target_cls).item()
            
            dice = (2 * intersection + smooth) / (pred_sum + target_sum + smooth)
            dice_scores.append(dice)
        
        mean_dice = sum(dice_scores) / len(dice_scores)
        return dice_scores, mean_dice
        
    except Exception as e:
        logging.error(f"Error in calculate_dice_coefficient: {e}")
        return [0.0] * num_classes, 0.0


def calculate_all_metrics(pred_tensor, target_tensor, num_classes):
    """
    Calculate all metrics at once: Pixel Accuracy, IoU, and Dice
    
    Args:
        pred_tensor (Tensor): Predicted labels
        target_tensor (Tensor): Ground truth labels
        num_classes (int): Number of classes
        
    Returns:
        dict: Dictionary containing all metrics
    """
    pixel_acc = calculate_pixel_accuracy(pred_tensor, target_tensor)
    class_ious, mean_iou = calculate_iou(pred_tensor, target_tensor, num_classes)
    class_dice, mean_dice = calculate_dice_coefficient(pred_tensor, target_tensor, num_classes)
    
    return {
        'pixel_accuracy': pixel_acc,
        'mean_iou': mean_iou,
        'class_iou': class_ious,
        'mean_dice': mean_dice,
        'class_dice': class_dice
    }
