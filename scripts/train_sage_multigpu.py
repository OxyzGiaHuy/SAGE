"""
Multi-GPU 2-Stage Training for SAGE (ConvNeXt + ViT + SAGE)
Supports DataParallel for training across multiple GPUs with larger batch sizes

Stage 1: Train full model to learn expert specialization.
Stage 2: Freeze selected shared experts and fine-tune with differential LRs.
"""

import argparse
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from sage.networks import create_convnext_transformer_unet
from sage.utils.dataloader import ConfigurableMedicalDataset, get_transformations
from sage.utils.metrics import (
    SimpleDiceLoss,
    calculate_pixel_accuracy,
    calculate_iou,
    calculate_dice_coefficient,
)
from sage.utils.training_utils import (
    setup_logging,
    set_seed,
    seed_worker,
    EarlyStopping,
)
from sage.utils.gs_tracker import GsTracker


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_output_dir(config: Dict) -> str:
    base = config["output"]["experiment_name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("experiments", base, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    data_cfg = config["data"]
    dataset_config_path = data_cfg["dataset_config"]

    train_t, val_t = get_transformations(data_cfg["img_size"])

    train_ds = ConfigurableMedicalDataset(
        dataset_config_path,
        split="train",
        image_size=data_cfg["img_size"],
        transform=train_t,
    )
    val_ds = ConfigurableMedicalDataset(
        dataset_config_path,
        split="val",
        image_size=data_cfg["img_size"],
        transform=val_t,
    )

    g = torch.Generator().manual_seed(config["training"]["seed"])

    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=data_cfg.get("pin_memory", True),
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=data_cfg.get("pin_memory", True),
    )

    logging.info(
        f"Dataloaders ready | Train: {len(train_ds)} samples, Val: {len(val_ds)} samples"
    )
    return train_loader, val_loader


def create_model(config: Dict, device: torch.device, use_multi_gpu: bool = False) -> nn.Module:
    model_cfg = config["model"]
    
    # Only pass sage_config if use_sage is True AND sage section exists
    use_sage = model_cfg.get("use_sage", False)
    if use_sage and "sage" in config:
        sage_cfg = config["sage"]
    else:
        sage_cfg = None

    model = create_convnext_transformer_unet(
        num_classes=model_cfg["num_classes"],
        img_size=model_cfg["img_size"],
        convnext_variant=model_cfg.get("convnext_variant", "convnext_base"),
        vit_variant=model_cfg.get("vit_variant", "vit_base_patch32_224"),
        num_transformer_layers=model_cfg.get("num_transformer_layers", 12),
        freeze_encoder=model_cfg.get("freeze_encoder", False),
        freeze_transformer=model_cfg.get("freeze_transformer", False),
        sage_config=sage_cfg,
        gradient_checkpointing=model_cfg.get("gradient_checkpointing", False),
    )

    # Sanity-check: model.img_size must equal data.img_size or outputs will be wrong
    data_img_size = config["data"]["img_size"]
    if model_cfg["img_size"] != data_img_size:
        raise ValueError(
            f"[Config mismatch] model.img_size={model_cfg['img_size']} "
            f"!= data.img_size={data_img_size}. "
            f"Model will output ({model_cfg['img_size']}x{model_cfg['img_size']}) "
            f"but labels are ({data_img_size}x{data_img_size}). "
            f"Fix: set model.img_size: {data_img_size} in your config."
        )
    
    # Move to device first
    model.to(device)
    
    # Wrap with DataParallel if multi-GPU
    if use_multi_gpu and torch.cuda.device_count() > 1:
        gpu_ids = config["training"].get("gpu_ids", None)
        if gpu_ids:
            model = nn.DataParallel(model, device_ids=gpu_ids)
            logging.info(f"Using DataParallel with GPUs: {gpu_ids}")
        else:
            model = nn.DataParallel(model)
            logging.info(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
    
    return model


def compute_lb_loss(routing_infos: Dict) -> torch.Tensor:
    lb_loss = 0.0
    layer_infos = []
    if isinstance(routing_infos, dict):
        layer_infos.extend(routing_infos.get("cnn", []))
        layer_infos.extend(routing_infos.get("transformer", []))
    for info in layer_infos:
        if info and isinstance(info, dict) and "load_balance_loss" in info:
            lb_loss = lb_loss + info["load_balance_loss"]
    if layer_infos:
        lb_loss = lb_loss / len(layer_infos)
    if not isinstance(lb_loss, torch.Tensor):
        lb_loss = torch.tensor(lb_loss)
    return lb_loss


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    ce_loss_fn: nn.Module,
    dice_loss_fn: SimpleDiceLoss,
    device: torch.device,
    num_classes: int,
    epoch: int,
    stage: int,
    is_data_parallel: bool = False,
    gs_tracker = None,
) -> Tuple[float, Dict[str, float]]:
    model.train()
    total_loss = 0.0
    metrics_sum = {"pixel_acc": 0.0, "iou": 0.0, "dice": 0.0}
    
    # Collect g_s scores for this epoch
    epoch_routing_infos = []

    pbar = tqdm(loader, desc=f"Stage {stage} | Epoch {epoch+1}", leave=False)
    for batch in pbar:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        lb_loss = torch.tensor(0.0, device=device)
        
        # Access the base model for routing info check
        base_model = model.module if is_data_parallel else model

        # Forward pass
        outputs = model(images)

        # Shape guard
        if outputs.shape[2:] != labels.shape[1:]:
            raise RuntimeError(
                f"[Shape mismatch] model output {tuple(outputs.shape)} vs "
                f"labels {tuple(labels.shape)}. "
                f"Check that model.img_size == data.img_size in your config."
            )
        
        # Compute load balance loss ONLY if SAGE is enabled
        routing_infos = None
        if hasattr(base_model, "forward_with_routing_info") and hasattr(base_model, "expert_pool") and base_model.expert_pool:
            sample_size = min(4, images.size(0))
            sample_images = images[:sample_size].to(device)
            with torch.no_grad():
                detailed = base_model.forward_with_routing_info(sample_images)
                routing_infos = detailed.get("routing_infos", {})
            lb_loss = compute_lb_loss(routing_infos).to(device)
            
            # Collect for g_s tracking
            if gs_tracker and routing_infos:
                epoch_routing_infos.append(routing_infos)

        ce_loss = ce_loss_fn(outputs, labels)
        dice_loss = dice_loss_fn(outputs, labels, softmax=True)
        seg_loss = 1.0 * ce_loss + 1.5 * dice_loss
        loss = seg_loss + lb_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            metrics_sum["pixel_acc"] += calculate_pixel_accuracy(preds, labels)
            _, miou = calculate_iou(preds, labels, num_classes=num_classes)
            metrics_sum["iou"] += miou
            _, mdice = calculate_dice_coefficient(preds, labels, num_classes=num_classes)
            metrics_sum["dice"] += mdice

        total_loss += loss.item()
        
        # Only show lb_loss in progress bar if it's non-zero (SAGE enabled)
        lb_value = lb_loss.item() if isinstance(lb_loss, torch.Tensor) else 0.0
        if lb_value > 0:
            pbar.set_postfix(loss=loss.item(), lb=lb_value)
        else:
            pbar.set_postfix(loss=loss.item())

    n = len(loader)
    avg_loss = total_loss / n
    avg_metrics = {k: v / n for k, v in metrics_sum.items()}
    
    # Finalize g_s tracking for this epoch
    if gs_tracker and epoch_routing_infos:
        gs_tracker.collect_gs_from_epoch(epoch_routing_infos, epoch)
        gs_tracker.log_epoch_summary(epoch)
    
    return avg_loss, avg_metrics


def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    ce_loss_fn: nn.Module,
    dice_loss_fn: SimpleDiceLoss,
    device: torch.device,
    num_classes: int,
    is_data_parallel: bool = False,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    metrics_sum = {"pixel_acc": 0.0, "iou": 0.0, "dice": 0.0}

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating", leave=False)
        for batch in pbar:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            base_model = model.module if is_data_parallel else model
            
            # Only use routing info if SAGE is actually enabled
            if hasattr(base_model, "forward_with_routing_info") and hasattr(base_model, "expert_pool") and base_model.expert_pool is not None:
                detailed = base_model.forward_with_routing_info(images)
                outputs = detailed["logits"] if isinstance(detailed, dict) else detailed
            else:
                outputs = model(images)

            ce_loss = ce_loss_fn(outputs, labels)
            dice_loss = dice_loss_fn(outputs, labels, softmax=True)
            loss = 1.0 * ce_loss + 1.5 * dice_loss
            total_loss += loss.item()

            preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            metrics_sum["pixel_acc"] += calculate_pixel_accuracy(preds, labels)
            _, miou = calculate_iou(preds, labels, num_classes=num_classes)
            metrics_sum["iou"] += miou
            _, mdice = calculate_dice_coefficient(preds, labels, num_classes=num_classes)
            metrics_sum["dice"] += mdice

            pbar.set_postfix(loss=loss.item(), mIoU=miou, mDice=mdice)

    n = len(loader)
    avg_loss = total_loss / n
    avg_metrics = {k: v / n for k, v in metrics_sum.items()}
    return avg_loss, avg_metrics


def create_scheduler(optimizer: optim.Optimizer, config: Dict, total_epochs: int):
    sched_type = config["training"]["scheduler"]
    if sched_type == "CosineAnnealingLR":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)
    if sched_type == "StepLR":
        return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    if sched_type == "PolynomialLR":
        return optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: (1.0 - epoch / total_epochs) ** 0.9
        )
    if sched_type == "ReduceLROnPlateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config["training"].get("lr_factor", 0.1),
            patience=config["training"].get("lr_patience", 5),
            min_lr=1e-6,
        )
    return None


def create_stage2_optimizer(model: nn.Module, config: Dict, shared_expert_indices: List[int], is_data_parallel: bool = False):
    train_cfg = config["training"]
    base_lr = train_cfg["stage2_base_lr"]
    shared_lr = train_cfg["stage2_shared_lr"]
    optimizer_type = train_cfg["optimizer"]
    weight_decay = train_cfg["weight_decay"]

    # Get base model if wrapped in DataParallel
    base_model = model.module if is_data_parallel else model

    shared_params, other_params = [], []
    num_cnn_experts = len(base_model.convnext.stages) if hasattr(base_model, "convnext") else 0
    prefixes = set()
    for idx in shared_expert_indices:
        if idx < num_cnn_experts:
            prefixes.add(f"convnext.stages.{idx}.")
        else:
            prefixes.add(f"transformer_blocks.{idx - num_cnn_experts}.")

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Handle DataParallel prefix
        clean_name = name.replace("module.", "") if is_data_parallel else name
        if any(clean_name.startswith(p) for p in prefixes):
            shared_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {"params": shared_params, "lr": shared_lr, "name": "shared_experts"},
        {"params": other_params, "lr": base_lr, "name": "other_and_routers"},
    ]

    if optimizer_type == "AdamW":
        optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
    elif optimizer_type == "Adam":
        optimizer = optim.Adam(param_groups, weight_decay=weight_decay)
    elif optimizer_type == "SGD":
        optimizer = optim.SGD(
            param_groups, momentum=train_cfg.get("momentum", 0.9), weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    logging.info(
        f"Stage 2 optimizer ready | shared params: {len(shared_params)}, other params: {len(other_params)}"
    )
    return optimizer


def select_shared_experts(config: Dict, model: nn.Module, is_data_parallel: bool = False) -> List[int]:
    base_model = model.module if is_data_parallel else model
    sage_cfg = config.get("sage", {})
    explicit = sage_cfg.get("shared_expert_indices", [])
    if explicit:
        return explicit
    k = config["training"]["num_shared_experts"]
    total = len(base_model.convnext.stages) + len(base_model.transformer_blocks)
    return list(range(min(k, total)))


def save_checkpoint(model: nn.Module, path: str, is_data_parallel: bool = False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Save only the base model state dict (without DataParallel wrapper)
    state_dict = model.module.state_dict() if is_data_parallel else model.state_dict()
    torch.save(state_dict, path)
    logging.info(f"Saved checkpoint to {path}")


def main(config_path: str, resume_checkpoint: str = None, resume_epoch: int = 0, resume_best_iou: float = None):
    config = load_config(config_path)
    output_dir = build_output_dir(config)
    setup_logging(output_dir, config["output"]["experiment_name"])
    set_seed(config["training"]["seed"])

    # Determine master GPU from gpu_ids config (first entry = master/output device)
    use_multi_gpu = config["training"].get("use_multi_gpu", False)
    gpu_ids_cfg = config["training"].get("gpu_ids", None)
    if torch.cuda.is_available():
        master_gpu = gpu_ids_cfg[0] if gpu_ids_cfg else 0
        torch.cuda.set_device(master_gpu)
        device = torch.device(f"cuda:{master_gpu}")
    else:
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")

    # Check for multi-GPU training
    if use_multi_gpu:
        logging.info(f"Multi-GPU training enabled | Master GPU: {master_gpu} | Available GPUs: {torch.cuda.device_count()}")

    train_loader, val_loader = get_dataloaders(config)
    ce_loss_fn = nn.CrossEntropyLoss()
    dice_loss_fn = SimpleDiceLoss(n_classes=config["model"]["num_classes"])

    # ============================
    # Stage 1
    # ============================
    logging.info("=== Stage 1: Training full model ===")
    model = create_model(config, device, use_multi_gpu=use_multi_gpu)
    is_data_parallel = isinstance(model, nn.DataParallel)

    # Resume from checkpoint if provided
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        base_model = model.module if is_data_parallel else model
        state = torch.load(resume_checkpoint, map_location=device)
        # Handle DataParallel prefix
        if list(state.keys())[0].startswith('module.'):
            state = {k.replace('module.', ''): v for k, v in state.items()}
        base_model.load_state_dict(state, strict=True)
        logging.info(f"Resumed from checkpoint: {resume_checkpoint} (starting at epoch {resume_epoch})")    
    
    stage1_optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["base_lr"],
        weight_decay=config["training"]["weight_decay"],
    )
    stage1_scheduler = create_scheduler(stage1_optimizer, config, config["training"]["stage1_epochs"])
    early_cfg = config["training"].get("early_stopping", {"patience": 10, "delta": 0.0})
    stage1_es = EarlyStopping(
        patience=early_cfg.get("patience", 10),
        delta=early_cfg.get("delta", 0.0),
        path=os.path.join(output_dir, "stage1_best_iou_model.pth"),
        metric_name="mIoU",
        mode="max",
    )
    # Restore EarlyStopping state if resuming (so patience counter is correct)
    if resume_best_iou is not None:
        stage1_es.best_score = resume_best_iou
        stage1_es.best_metric = resume_best_iou
        logging.info(f"Restored EarlyStopping best_iou={resume_best_iou:.4f}")
        # Copy checkpoint to new output_dir so EarlyStopping can reference/save it
        import shutil
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            shutil.copy2(resume_checkpoint, stage1_es.path)

    stage1_history = defaultdict(list)
    for epoch in range(resume_epoch, config["training"]["stage1_epochs"]):
        train_loss, train_metrics = train_one_epoch(
            model,
            train_loader,
            stage1_optimizer,
            ce_loss_fn,
            dice_loss_fn,
            device,
            config["model"]["num_classes"],
            epoch,
            stage=1,
            is_data_parallel=is_data_parallel,
            gs_tracker=None,
        )
        val_loss, val_metrics = validate_one_epoch(
            model,
            val_loader,
            ce_loss_fn,
            dice_loss_fn,
            device,
            config["model"]["num_classes"],
            is_data_parallel=is_data_parallel,
        )

        stage1_history["train_loss"].append(train_loss)
        stage1_history["val_loss"].append(val_loss)
        stage1_history["val_iou"].append(val_metrics["iou"])
        stage1_history["val_dice"].append(val_metrics["dice"])

        if stage1_scheduler:
            if isinstance(stage1_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                stage1_scheduler.step(val_loss)
            else:
                stage1_scheduler.step()

        logging.info(
            f"[Stage1][Epoch {epoch+1}/{config['training']['stage1_epochs']}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_mIoU={val_metrics['iou']:.4f} val_mDice={val_metrics['dice']:.4f}"
        )

        # Handle early stopping with DataParallel
        base_model_for_save = model.module if is_data_parallel else model
        if stage1_es(val_metrics["iou"], base_model_for_save, epoch + 1):
            logging.info("Early stopping triggered for Stage 1.")
            break

        # Free fragmented cache between epochs to prevent OOM accumulation
        torch.cuda.empty_cache()

    # reload best Stage 1
    stage1_best_path = stage1_es.path
    if os.path.exists(stage1_best_path):
        base_model = model.module if is_data_parallel else model
        base_model.load_state_dict(torch.load(stage1_best_path, map_location=device))
        logging.info(f"Loaded best Stage 1 weights from {stage1_best_path}")

    # ============================
    # Stage 2 (skip if stage2_epochs == 0, e.g. baseline with no SAGE)
    # ============================
    stage2_epochs = config["training"].get("stage2_epochs", 0)
    if stage2_epochs <= 0:
        logging.info("Stage 2 disabled (stage2_epochs=0). Training complete.")
    else:
      logging.info("=== Stage 2: Fine-tuning with shared experts ===")
      shared_indices = select_shared_experts(config, model, is_data_parallel)
      logging.info(f"Shared experts: {shared_indices}")
      
      # Initialize G_s tracker for Stage 2 (track stage 2 only)
      gs_tracker = GsTracker(output_dir, experiment_name="gs_tracking")
      gs_tracker.set_stage("stage2")

      base_model = model.module if is_data_parallel else model
      if hasattr(base_model, "set_shared_experts"):
          base_model.set_shared_experts(shared_indices)

      stage2_optimizer = create_stage2_optimizer(model, config, shared_indices, is_data_parallel)
      stage2_scheduler = create_scheduler(stage2_optimizer, config, stage2_epochs)
      stage2_es = EarlyStopping(
          patience=early_cfg.get("patience", 10),
          delta=early_cfg.get("delta", 0.0),
          path=os.path.join(output_dir, "stage2_best_iou_model.pth"),
          metric_name="mIoU",
          mode="max",
      )

      for epoch in range(stage2_epochs):
          train_loss, train_metrics = train_one_epoch(
              model,
              train_loader,
              stage2_optimizer,
              ce_loss_fn,
              dice_loss_fn,
              device,
              config["model"]["num_classes"],
              epoch,
              stage=2,
              is_data_parallel=is_data_parallel,
              gs_tracker=gs_tracker,
          )
          val_loss, val_metrics = validate_one_epoch(
              model,
              val_loader,
              ce_loss_fn,
              dice_loss_fn,
              device,
              config["model"]["num_classes"],
              is_data_parallel=is_data_parallel,
          )

          if stage2_scheduler:
              if isinstance(stage2_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                  stage2_scheduler.step(val_loss)
              else:
                  stage2_scheduler.step()

          logging.info(
              f"[Stage2][Epoch {epoch+1}/{stage2_epochs}] "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_mIoU={val_metrics['iou']:.4f} val_mDice={val_metrics['dice']:.4f}"
          )

          # Handle early stopping with DataParallel
          base_model_for_save = model.module if is_data_parallel else model
          if stage2_es(val_metrics["iou"], base_model_for_save, epoch + 1):
              logging.info("Early stopping triggered for Stage 2.")
              break

          # Free fragmented cache between epochs
          torch.cuda.empty_cache()

      stage2_best_path = stage2_es.path
      if os.path.exists(stage2_best_path):
          logging.info(f"Stage 2 best model saved at: {stage2_best_path}")
    
    # Generate G_s tracking report
    logging.info("\n" + "="*80)
    logging.info("Finalizing G_s tracking...")
    logging.info("="*80)
    gs_tracker.generate_report()
    logging.info("G_s tracking report generated successfully!")
    logging.info("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-GPU 2-Stage Training for SAGE")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to stage1 checkpoint to resume from")
    parser.add_argument("--resume_epoch", type=int, default=0,
                        help="Epoch number to resume from (0-indexed, e.g. 1 to skip epoch 0)")
    parser.add_argument("--resume_best_iou", type=float, default=None,
                        help="Best val_mIoU achieved so far (for EarlyStopping state)")
    args = parser.parse_args()
    main(args.config, resume_checkpoint=args.resume, resume_epoch=args.resume_epoch, resume_best_iou=args.resume_best_iou)
