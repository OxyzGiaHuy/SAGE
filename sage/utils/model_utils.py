"""
Model utilities for SAGE
Functions for model management, checkpointing, and expert handling
"""

import os
import json
import logging
from typing import List, Dict, Optional
import torch
import torch.nn as nn


def count_parameters(model, trainable_only=True):
    """
    Count number of parameters in model
    
    Args:
        model (nn.Module): PyTorch model
        trainable_only (bool): Whether to count only trainable parameters
        
    Returns:
        int: Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def print_model_summary(model, input_size=(1, 3, 224, 224)):
    """
    Print detailed model summary
    
    Args:
        model (nn.Module): PyTorch model
        input_size (tuple): Input tensor size
    """
    logging.info("="*80)
    logging.info("MODEL SUMMARY")
    logging.info("="*80)
    
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    
    logging.info(f"Total parameters:     {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")
    logging.info(f"Non-trainable params: {total_params - trainable_params:,}")
    logging.info(f"Input size:           {input_size}")
    
    # Calculate model size in MB
    param_size = total_params * 4 / (1024 ** 2)  # Assuming float32
    logging.info(f"Estimated size:       {param_size:.2f} MB")
    
    logging.info("="*80)


def save_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
    metrics,
    filepath,
    scheduler=None,
    extra_info=None
):
    """
    Save model checkpoint with training state
    
    Args:
        model (nn.Module): Model to save
        optimizer (Optimizer): Optimizer state
        epoch (int): Current epoch
        loss (float): Current loss
        metrics (dict): Current metrics
        filepath (str): Path to save checkpoint
        scheduler: Learning rate scheduler (optional)
        extra_info (dict): Additional information to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics,
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if extra_info is not None:
        checkpoint.update(extra_info)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    torch.save(checkpoint, filepath)
    logging.info(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    filepath,
    model,
    optimizer=None,
    scheduler=None,
    device='cpu',
    strict=True
):
    """
    Load model checkpoint
    
    Args:
        filepath (str): Path to checkpoint file
        model (nn.Module): Model to load weights into
        optimizer (Optimizer): Optimizer to load state (optional)
        scheduler: Learning rate scheduler to load state (optional)
        device (str): Device to map checkpoint to
        strict (bool): Whether to enforce an exact key/shape match
        
    Returns:
        dict: Checkpoint information (epoch, loss, metrics)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found at {filepath}")
    
    logging.info(f"Loading checkpoint from {filepath}")
    checkpoint = torch.load(filepath, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # Checkpoint is the state_dict itself
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Load model state
    try:
        incompatible = model.load_state_dict(state_dict, strict=strict)
    except RuntimeError as e:
        logging.error(
            "Failed to load checkpoint strictly. "
            "Ensure the inference architecture matches the training architecture."
        )
        raise
    
    # Report missing/unexpected keys when partial loading is allowed
    if not strict and incompatible is not None:
        if incompatible.missing_keys:
            logging.warning(f"Missing keys during checkpoint load: {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            logging.warning(f"Unexpected keys during checkpoint load: {incompatible.unexpected_keys}")
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    info = {
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', 0.0),
        'metrics': checkpoint.get('metrics', {})
    }
    
    logging.info(f"Checkpoint loaded from epoch {info['epoch']}")
    return info


def load_pretrained_weights(model, checkpoint_path, strict=True, exclude_keys=None):
    """
    Load pretrained weights with flexibility
    
    Args:
        model (nn.Module): Model to load weights into
        checkpoint_path (str): Path to checkpoint
        strict (bool): Whether to strictly enforce key matching
        exclude_keys (list): List of keys to exclude from loading
        
    Returns:
        tuple: (loaded keys, unexpected keys, missing keys)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    logging.info(f"Loading pretrained weights from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract state dict (handle both direct state dict and checkpoint dict)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Exclude specified keys
    if exclude_keys:
        state_dict = {k: v for k, v in state_dict.items() 
                     if not any(ex_key in k for ex_key in exclude_keys)}
        logging.info(f"Excluded keys matching: {exclude_keys}")
    
    # Load with strict=False to handle partial loading
    result = model.load_state_dict(state_dict, strict=strict)
    
    if not strict:
        logging.info(f"Missing keys: {result.missing_keys}")
        logging.info(f"Unexpected keys: {result.unexpected_keys}")
    
    logging.info("Pretrained weights loaded successfully")
    return result


def load_shared_expert_weights(
    model: nn.Module,
    checkpoint_path: str,
    shared_expert_indices: List[int],
    device='cpu'
):
    """
    Load weights ONLY for specified shared experts from a checkpoint
    Useful for two-stage training with SAGE
    
    Args:
        model (nn.Module): Model to load expert weights into
        checkpoint_path (str): Path to checkpoint with expert weights
        shared_expert_indices (list): List of expert indices to load
        device (str): Device to map weights to
    """
    if not os.path.exists(checkpoint_path):
        logging.error(f"Checkpoint not found at {checkpoint_path}")
        return
    
    logging.info(f"Loading shared expert weights for indices {shared_expert_indices}")
    logging.info(f"From checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        stage1_state_dict = checkpoint['model_state_dict']
    else:
        stage1_state_dict = checkpoint
    
    # Get current model state
    stage2_state_dict = model.state_dict()
    state_dict_to_load = stage2_state_dict.copy()
    
    params_loaded = 0
    
    # Iterate through checkpoint parameters
    for param_name, param_value in stage1_state_dict.items():
        # Look for expert_pool parameters
        parts = param_name.split('.')
        
        if 'expert_pool' in parts:
            try:
                expert_pool_idx = parts.index('expert_pool')
                expert_idx = int(parts[expert_pool_idx + 1])
                
                # Load only specified experts
                if expert_idx in shared_expert_indices:
                    if param_name in state_dict_to_load:
                        # Check shape compatibility
                        if state_dict_to_load[param_name].shape == param_value.shape:
                            state_dict_to_load[param_name] = param_value
                            params_loaded += 1
                        else:
                            logging.warning(
                                f"Shape mismatch for {param_name}: "
                                f"{state_dict_to_load[param_name].shape} vs {param_value.shape}"
                            )
            except (ValueError, IndexError):
                continue
    
    if params_loaded > 0:
        model.load_state_dict(state_dict_to_load)
        logging.info(f"Successfully loaded {params_loaded} parameters for shared experts")
    else:
        logging.warning("No expert parameters were loaded. Check indices and checkpoint.")


def freeze_model_components(model, freeze_encoder=False, freeze_transformer=False, freeze_decoder=False):
    """
    Freeze specific components of the model
    
    Args:
        model (nn.Module): Model with components to freeze
        freeze_encoder (bool): Whether to freeze encoder
        freeze_transformer (bool): Whether to freeze transformer/bottleneck
        freeze_decoder (bool): Whether to freeze decoder
    """
    if freeze_encoder and hasattr(model, 'encoder'):
        for param in model.encoder.parameters():
            param.requires_grad = False
        logging.info("Encoder frozen")
    
    if freeze_transformer:
        if hasattr(model, 'bottleneck'):
            for param in model.bottleneck.parameters():
                param.requires_grad = False
            logging.info("Transformer bottleneck frozen")
        elif hasattr(model, 'transformer'):
            for param in model.transformer.parameters():
                param.requires_grad = False
            logging.info("Transformer frozen")
    
    if freeze_decoder and hasattr(model, 'decoder'):
        for param in model.decoder.parameters():
            param.requires_grad = False
        logging.info("Decoder frozen")
    
    # Print trainable parameter count after freezing
    trainable_params = count_parameters(model, trainable_only=True)
    total_params = count_parameters(model, trainable_only=False)
    logging.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
                f"({100 * trainable_params / total_params:.1f}%)")


def get_model_info(model):
    """
    Get comprehensive model information
    
    Args:
        model (nn.Module): Model to analyze
        
    Returns:
        dict: Model information
    """
    info = {
        'total_parameters': count_parameters(model, trainable_only=False),
        'trainable_parameters': count_parameters(model, trainable_only=True),
        'model_type': type(model).__name__,
    }
    
    # Add component-specific info if available
    if hasattr(model, 'encoder'):
        info['encoder_type'] = type(model.encoder).__name__
        if hasattr(model.encoder, 'encoder_channels'):
            info['encoder_channels'] = model.encoder.encoder_channels
    
    if hasattr(model, 'bottleneck'):
        info['bottleneck_type'] = type(model.bottleneck).__name__
        if hasattr(model.bottleneck, 'transformer_dim'):
            info['transformer_dim'] = model.bottleneck.transformer_dim
    
    if hasattr(model, 'decoder'):
        info['decoder_type'] = type(model.decoder).__name__
    
    # SAGE info
    if hasattr(model, 'sage_config') and model.sage_config:
        info['sage_enabled'] = True
        info['sage_config'] = model.sage_config
    else:
        info['sage_enabled'] = False
    
    return info


def save_model_config(model, config_path):
    """
    Save model configuration to JSON
    
    Args:
        model (nn.Module): Model to save config from
        config_path (str): Path to save JSON config
    """
    info = get_model_info(model)
    
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(info, f, indent=4)
    
    logging.info(f"Model config saved to {config_path}")
