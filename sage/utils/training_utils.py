"""
Training utilities for SAGE
Functions for logging, reproducibility, and training management
"""

import os
import random
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns


def setup_logging(output_dir, experiment_name="experiment"):
    """
    Setup logging to write to both file and console
    
    Args:
        output_dir (str): Directory to save log file
        experiment_name (str): Name of the experiment
    """
    os.makedirs(output_dir, exist_ok=True)
    
    log_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-5.5s]  %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    root_logger.handlers = []
    
    # Log to file
    log_file_path = os.path.join(output_dir, f"{experiment_name}.log")
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)
    
    # Log to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    
    logging.info(f"Logging initialized. Log file: {log_file_path}")


def set_seed(seed=42):
    """
    Set seed for reproducibility across all random number generators
    
    Args:
        seed (int): Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Ensure cuDNN determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logging.info(f"Random seed set to {seed} for reproducibility")


def seed_worker(worker_id):
    """
    Worker initialization function for DataLoader to ensure reproducibility
    
    Args:
        worker_id (int): Worker ID passed by DataLoader
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class EarlyStopping:
    """
    Early stopping to stop training when validation metric doesn't improve
    """
    
    def __init__(
        self,
        patience=10,
        verbose=True,
        delta=0.0,
        path='checkpoint.pt',
        metric_name='IoU',
        mode='max',
        trace_func=logging.info
    ):
        """
        Args:
            patience (int): How long to wait after last time metric improved
            verbose (bool): If True, prints message for each validation improvement
            delta (float): Minimum change to qualify as improvement
            path (str): Path to save checkpoint
            metric_name (str): Name of metric being monitored
            mode (str): 'max' or 'min' - whether higher or lower is better
            trace_func (function): Function to use for printing
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.metric_name = metric_name
        self.mode = mode
        self.trace_func = trace_func
        
        # Track best metric value
        if mode == 'max':
            self.best_metric = -float('inf')
            self.comparison = lambda current, best: current > best + delta
        else:  # mode == 'min'
            self.best_metric = float('inf')
            self.comparison = lambda current, best: current < best - delta
    
    def __call__(self, metric_value, model, epoch=None):
        """
        Check if early stopping criteria is met
        
        Args:
            metric_value (float): Current validation metric value
            model (nn.Module): Model to save if improved
            epoch (int): Current epoch number
            
        Returns:
            bool: True if should stop training, False otherwise
        """
        score = metric_value if self.mode == 'max' else -metric_value
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(metric_value, model, epoch)
        elif not self.comparison(metric_value, self.best_metric):
            self.counter += 1
            if self.verbose:
                self.trace_func(
                    f"EarlyStopping counter: {self.counter} out of {self.patience} "
                    f"(best {self.metric_name}: {self.best_metric:.4f})"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(metric_value, model, epoch)
            self.counter = 0
        
        return self.early_stop
    
    def save_checkpoint(self, metric_value, model, epoch=None):
        """Save model when validation metric improves"""
        if self.verbose:
            epoch_str = f" (epoch {epoch})" if epoch is not None else ""
            self.trace_func(
                f"Validation {self.metric_name} improved "
                f"({self.best_metric:.4f} → {metric_value:.4f}){epoch_str}. "
                f"Saving model to {self.path}"
            )
        torch.save(model.state_dict(), self.path)
        self.best_metric = metric_value


class MetricsTracker:
    """
    Track and store training/validation metrics
    """
    
    def __init__(self):
        """Initialize metrics storage"""
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = {'iou': [], 'dice': [], 'pixel_acc': []}
        self.val_metrics = {'iou': [], 'dice': [], 'pixel_acc': []}
        self.learning_rates = []
        self.epochs = []
    
    def update(self, epoch, train_loss=None, val_loss=None, 
               train_iou=None, val_iou=None,
               train_dice=None, val_dice=None,
               train_pixel_acc=None, val_pixel_acc=None,
               lr=None):
        """
        Update metrics for current epoch
        
        Args:
            epoch (int): Current epoch number
            train_loss (float): Training loss
            val_loss (float): Validation loss
            train_iou (float): Training IoU
            val_iou (float): Validation IoU
            train_dice (float): Training Dice
            val_dice (float): Validation Dice
            train_pixel_acc (float): Training pixel accuracy
            val_pixel_acc (float): Validation pixel accuracy
            lr (float): Current learning rate
        """
        self.epochs.append(epoch)
        
        if train_loss is not None:
            self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        if train_iou is not None:
            self.train_metrics['iou'].append(train_iou)
        if val_iou is not None:
            self.val_metrics['iou'].append(val_iou)
        if train_dice is not None:
            self.train_metrics['dice'].append(train_dice)
        if val_dice is not None:
            self.val_metrics['dice'].append(val_dice)
        if train_pixel_acc is not None:
            self.train_metrics['pixel_acc'].append(train_pixel_acc)
        if val_pixel_acc is not None:
            self.val_metrics['pixel_acc'].append(val_pixel_acc)
        if lr is not None:
            self.learning_rates.append(lr)
    
    def plot_metrics(self, save_path, show_plot=False):
        """
        Plot all tracked metrics
        
        Args:
            save_path (str): Path to save plot
            show_plot (bool): Whether to display plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Training Metrics', fontsize=16)
        
        # Plot losses
        if self.train_losses and self.val_losses:
            axes[0, 0].plot(self.epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
            axes[0, 0].plot(self.epochs, self.val_losses, 'r-', label='Val Loss', linewidth=2)
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Loss Curves')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot IoU
        if self.train_metrics['iou'] and self.val_metrics['iou']:
            axes[0, 1].plot(self.epochs, self.train_metrics['iou'], 'b-', label='Train IoU', linewidth=2)
            axes[0, 1].plot(self.epochs, self.val_metrics['iou'], 'r-', label='Val IoU', linewidth=2)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('IoU')
            axes[0, 1].set_title('IoU Scores')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot Dice
        if self.train_metrics['dice'] and self.val_metrics['dice']:
            axes[1, 0].plot(self.epochs, self.train_metrics['dice'], 'b-', label='Train Dice', linewidth=2)
            axes[1, 0].plot(self.epochs, self.val_metrics['dice'], 'r-', label='Val Dice', linewidth=2)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Dice')
            axes[1, 0].set_title('Dice Coefficients')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot learning rate
        if self.learning_rates:
            axes[1, 1].plot(self.epochs, self.learning_rates, 'g-', linewidth=2)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        logging.info(f"Metrics plot saved to {save_path}")
    
    def get_best_metrics(self):
        """
        Get best validation metrics
        
        Returns:
            dict: Best validation metrics
        """
        best_metrics = {}
        
        if self.val_metrics['iou']:
            best_iou_idx = np.argmax(self.val_metrics['iou'])
            best_metrics['best_val_iou'] = self.val_metrics['iou'][best_iou_idx]
            best_metrics['best_iou_epoch'] = self.epochs[best_iou_idx]
        
        if self.val_metrics['dice']:
            best_dice_idx = np.argmax(self.val_metrics['dice'])
            best_metrics['best_val_dice'] = self.val_metrics['dice'][best_dice_idx]
            best_metrics['best_dice_epoch'] = self.epochs[best_dice_idx]
        
        if self.val_losses:
            best_loss_idx = np.argmin(self.val_losses)
            best_metrics['best_val_loss'] = self.val_losses[best_loss_idx]
            best_metrics['best_loss_epoch'] = self.epochs[best_loss_idx]
        
        return best_metrics
