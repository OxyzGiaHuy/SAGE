"""
DDP 2-Stage Training for SAGE (ConvNeXt + ViT + SAGE)
Uses DistributedDataParallel for equal memory distribution across GPUs.

Launch with:
    torchrun --nproc_per_node=8 train_2stage_sage_ddp.py --config eccv/config/config_sage_colon.yaml

Stage 1: Train full model to learn expert specialization.
Stage 2: Freeze selected shared experts and fine-tune with differential LRs.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse
import logging
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
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


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def reduce_scalar(value: float, device: torch.device) -> float:
    """Average a scalar across all DDP ranks."""
    t = torch.tensor(value, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t / dist.get_world_size()).item()


# ---------------------------------------------------------------------------
# Config / output
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_output_dir(config: Dict) -> str:
    base = config["output"]["experiment_name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("experiments", base, timestamp)
    if is_main_process():
        os.makedirs(output_dir, exist_ok=True)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()  # all ranks wait until directory exists
    return output_dir


# ---------------------------------------------------------------------------
# Dataloaders with DistributedSampler
# ---------------------------------------------------------------------------

def get_dataloaders(config: Dict, rank: int, world_size: int) -> Tuple[DataLoader, DataLoader, DistributedSampler]:
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

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    g = torch.Generator().manual_seed(config["training"]["seed"])

    nw = data_cfg["num_workers"]
    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg["batch_size"],   # per-GPU batch size
        sampler=train_sampler,
        num_workers=nw,
        pin_memory=data_cfg.get("pin_memory", True),
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=True,
        persistent_workers=(nw > 0),  # keep workers alive between epochs
        prefetch_factor=4 if nw > 0 else None,  # pre-load next 4 batches
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=data_cfg["batch_size"],
        sampler=val_sampler,
        num_workers=nw,
        pin_memory=data_cfg.get("pin_memory", True),
        persistent_workers=(nw > 0),
        prefetch_factor=4 if nw > 0 else None,
    )

    if is_main_process():
        logging.info(
            f"Dataloaders ready | Train: {len(train_ds)} samples ({len(train_loader)} batches/rank)"
            f" | Val: {len(val_ds)} samples ({len(val_loader)} batches/rank)"
        )
    return train_loader, val_loader, train_sampler


# ---------------------------------------------------------------------------
# Model creation (no DDP wrap here — caller wraps with DDP)
# ---------------------------------------------------------------------------

def create_model_base(config: Dict, device: torch.device) -> nn.Module:
    model_cfg = config["model"]

    use_sage = model_cfg.get("use_sage", False)
    sage_cfg = config["sage"] if (use_sage and "sage" in config) else None

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

    data_img_size = config["data"]["img_size"]
    if model_cfg["img_size"] != data_img_size:
        raise ValueError(
            f"[Config mismatch] model.img_size={model_cfg['img_size']} "
            f"!= data.img_size={data_img_size}. Fix: set model.img_size: {data_img_size} in config."
        )

    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Load-balance loss from SAGE routing infos
# ---------------------------------------------------------------------------

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


def collect_lb_loss(base_model: nn.Module, device: torch.device) -> torch.Tensor:
    """Sum load_balance_loss from all SageLayers after a forward pass.
    Each layer caches its last lb_loss in _last_lb_loss (no extra forward needed).
    """
    from sage.components.sage_layer import SageLayer
    total = torch.tensor(0.0, device=device)
    count = 0
    for module in base_model.modules():
        if isinstance(module, SageLayer) and module._last_lb_loss is not None:
            lb = module._last_lb_loss
            if isinstance(lb, torch.Tensor) and lb.requires_grad:
                total = total + lb
                count += 1
    return total / count if count > 0 else total


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,           # DDP-wrapped
    base_model: nn.Module,      # model.module — for SAGE routing
    loader: DataLoader,
    train_sampler: DistributedSampler,
    optimizer: optim.Optimizer,
    ce_loss_fn: nn.Module,
    dice_loss_fn: SimpleDiceLoss,
    device: torch.device,
    num_classes: int,
    epoch: int,
    stage: int,
    use_amp: bool = True,
    lb_factor: float = 0.0,    # load_balance_factor from sage config
    gs_tracker = None,         # G_s tracking
) -> Tuple[float, Dict[str, float]]:
    # Must set epoch so each rank sees a different shuffle
    train_sampler.set_epoch(epoch)

    model.train()
    total_loss = 0.0
    metrics_sum = {"pixel_acc": 0.0, "iou": 0.0, "dice": 0.0, "lb": 0.0}
    
    # Collect g_s scores for this epoch (main process only)
    epoch_routing_infos = [] if (gs_tracker and is_main_process()) else None

    pbar = tqdm(loader, desc=f"Stage {stage} | Epoch {epoch+1}", leave=False, disable=not is_main_process())
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # faster than zero_grad()

        # BF16 AMP: A100 tensor cores, no loss scaling needed (bf16 has fp32 exponent range)
        with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
            # Forward pass through DDP model
            outputs = model(images)
            
            # Collect routing info for g_s tracking (on first rank only)
            routing_infos = None
            if gs_tracker and is_main_process() and hasattr(base_model, "forward_with_routing_info"):
                with torch.no_grad():
                    detailed = base_model.forward_with_routing_info(images)
                    routing_infos = detailed.get("routing_infos", {})
                    if routing_infos and epoch_routing_infos is not None:
                        epoch_routing_infos.append(routing_infos)

            # Shape guard (only check on main process to avoid duplicate noise)
            if is_main_process() and outputs.shape[2:] != labels.shape[1:]:
                raise RuntimeError(
                    f"[Shape mismatch] model output {tuple(outputs.shape)} vs "
                    f"labels {tuple(labels.shape)}. Check model.img_size == data.img_size."
                )

            # Collect load-balance loss from all SageLayers (cached during forward)
            # load_balance_factor=0 → disabled; otherwise weighted aux loss
            if lb_factor > 0:
                lb_loss = collect_lb_loss(base_model, device) * lb_factor
            else:
                lb_loss = torch.tensor(0.0, device=device)

            ce_loss   = ce_loss_fn(outputs, labels)
            dice_loss = dice_loss_fn(outputs, labels, softmax=True)
            seg_loss  = 1.0 * ce_loss + 1.5 * dice_loss
            loss      = seg_loss + lb_loss

        # BF16 doesn't need GradScaler — backward in full precision automatically
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            metrics_sum["pixel_acc"] += calculate_pixel_accuracy(preds, labels)
            _, miou  = calculate_iou(preds, labels, num_classes=num_classes)
            metrics_sum["iou"] += miou
            _, mdice = calculate_dice_coefficient(preds, labels, num_classes=num_classes)
            metrics_sum["dice"] += mdice
            metrics_sum["lb"] += lb_loss.item()

        total_loss += loss.item()

        if is_main_process():
            lb_value = lb_loss.item() if isinstance(lb_loss, torch.Tensor) else 0.0
            if lb_value > 0:
                pbar.set_postfix(loss=loss.item(), seg=seg_loss.item(), lb=f"{lb_value:.5f}")
            else:
                pbar.set_postfix(loss=loss.item())

    n = len(loader)
    # Average metrics across all DDP ranks so main process has global values
    avg_loss = reduce_scalar(total_loss / n, device)
    avg_metrics = {k: reduce_scalar(v / n, device) for k, v in metrics_sum.items()}
    
    # Finalize g_s tracking for this epoch
    if gs_tracker and is_main_process() and epoch_routing_infos:
        gs_tracker.collect_gs_from_epoch(epoch_routing_infos, epoch)
        gs_tracker.log_epoch_summary(epoch)
    
    return avg_loss, avg_metrics


# ---------------------------------------------------------------------------
# Validation loop
# ---------------------------------------------------------------------------

def validate_one_epoch(
    model: nn.Module,
    base_model: nn.Module,
    loader: DataLoader,
    ce_loss_fn: nn.Module,
    dice_loss_fn: SimpleDiceLoss,
    device: torch.device,
    num_classes: int,
    use_amp: bool = True,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    metrics_sum = {"pixel_acc": 0.0, "iou": 0.0, "dice": 0.0}

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating", leave=False, disable=not is_main_process())
        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                outputs = model(images)
                ce_loss   = ce_loss_fn(outputs, labels)
                dice_loss = dice_loss_fn(outputs, labels, softmax=True)
                loss      = 1.0 * ce_loss + 1.5 * dice_loss
            total_loss += loss.item()

            preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            metrics_sum["pixel_acc"] += calculate_pixel_accuracy(preds, labels)
            _, miou  = calculate_iou(preds, labels, num_classes=num_classes)
            metrics_sum["iou"] += miou
            _, mdice = calculate_dice_coefficient(preds, labels, num_classes=num_classes)
            metrics_sum["dice"] += mdice

            if is_main_process():
                pbar.set_postfix(loss=loss.item(), mIoU=miou, mDice=mdice)

    n = len(loader)
    avg_loss    = reduce_scalar(total_loss / n, device)
    avg_metrics = {k: reduce_scalar(v / n, device) for k, v in metrics_sum.items()}
    return avg_loss, avg_metrics


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Stage 2 helpers
# ---------------------------------------------------------------------------

def create_stage2_optimizer(
    model: nn.Module,
    base_model: nn.Module,
    config: Dict,
    shared_expert_indices: List[int],
) -> optim.Optimizer:
    train_cfg  = config["training"]
    base_lr    = train_cfg["stage2_base_lr"]
    shared_lr  = train_cfg["stage2_shared_lr"]
    optimizer_type = train_cfg["optimizer"]
    weight_decay   = train_cfg["weight_decay"]

    num_cnn_experts = len(base_model.convnext.stages) if hasattr(base_model, "convnext") else 0
    prefixes = set()
    for idx in shared_expert_indices:
        if idx < num_cnn_experts:
            prefixes.add(f"module.convnext.stages.{idx}.")        # DDP prefix
        else:
            prefixes.add(f"module.transformer_blocks.{idx - num_cnn_experts}.")

    shared_params, other_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(name.startswith(p) for p in prefixes):
            shared_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {"params": shared_params, "lr": shared_lr,  "name": "shared_experts"},
        {"params": other_params,  "lr": base_lr,    "name": "other_and_routers"},
    ]

    if optimizer_type == "AdamW":
        optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
    elif optimizer_type == "Adam":
        optimizer = optim.Adam(param_groups, weight_decay=weight_decay)
    elif optimizer_type == "SGD":
        optimizer = optim.SGD(param_groups, momentum=train_cfg.get("momentum", 0.9), weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    if is_main_process():
        logging.info(f"Stage 2 optimizer | shared params: {len(shared_params)}, other params: {len(other_params)}")
    return optimizer


def select_shared_experts(config: Dict, base_model: nn.Module) -> List[int]:
    sage_cfg = config.get("sage", {})
    explicit  = sage_cfg.get("shared_expert_indices", [])
    if explicit:
        return explicit
    k     = config["training"]["num_shared_experts"]
    total = len(base_model.convnext.stages) + len(base_model.transformer_blocks)
    return list(range(min(k, total)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str, resume_checkpoint: str = None, resume_epoch: int = 0, resume_best_iou: float = None):
    # ── DDP init ──────────────────────────────────────────────────────────────
    # CRITICAL: set device BEFORE init_process_group so NCCL allocates its
    # workspace on the correct GPU (not all processes on GPU 0).
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    rank       = dist.get_rank()

    # ── Speed optimizations (accuracy-neutral) ─────────────────────────────
    torch.backends.cudnn.benchmark = True       # auto-tune kernels for fixed input size
    torch.set_float32_matmul_precision('high')  # TF32 for fp32 matmuls on A100 (~2x faster)

    config     = load_config(config_path)
    use_amp    = config.get("training", {}).get("use_amp", True)   # BF16 AMP (default on)
    output_dir = build_output_dir(config)

    if is_main_process():
        setup_logging(output_dir, config["output"]["experiment_name"])
        logging.info(f"DDP training | world_size={world_size} | local_rank={local_rank}")
        logging.info(f"AMP={use_amp} (bf16) | cudnn.benchmark=True | TF32=high")
        logging.info(f"Output dir: {output_dir}")

    set_seed(config["training"]["seed"] + rank)  # different seed per rank

    train_loader, val_loader, train_sampler = get_dataloaders(config, rank, world_size)
    ce_loss_fn   = nn.CrossEntropyLoss()
    dice_loss_fn = SimpleDiceLoss(n_classes=config["model"]["num_classes"])

    # ======================================================================
    # Stage 1
    # ======================================================================
    if is_main_process():
        logging.info("=== Stage 1: Training full model ===")

    base_model = create_model_base(config, device)

    # Resume from checkpoint (all ranks load same weights)
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        state = torch.load(resume_checkpoint, map_location=device)
        if list(state.keys())[0].startswith("module."):
            state = {k.replace("module.", ""): v for k, v in state.items()}
        base_model.load_state_dict(state, strict=True)
        if is_main_process():
            logging.info(f"Resumed from: {resume_checkpoint} (starting at epoch {resume_epoch})")
    dist.barrier()  # all ranks sync after load

    # Wrap with DDP
    # find_unused_parameters=True: CRITICAL for SAGE — not all experts routed every forward
    # gradient_as_bucket_view=True: reduces memory copies, fixes stride warnings
    model = DDP(base_model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=True, gradient_as_bucket_view=True)

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
    if resume_best_iou is not None:
        stage1_es.best_score  = resume_best_iou
        stage1_es.best_metric = resume_best_iou
        if is_main_process():
            logging.info(f"Restored EarlyStopping best_iou={resume_best_iou:.4f}")
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            import shutil
            if is_main_process():
                shutil.copy2(resume_checkpoint, stage1_es.path)
    dist.barrier()

    stage1_history = defaultdict(list)
    should_stop    = False
    lb_factor = config.get("sage", {}).get("load_balance_factor", 0.0)  # 0 if baseline
    if is_main_process():
        logging.info(f"Load-balance factor: {lb_factor}")

    for epoch in range(resume_epoch, config["training"]["stage1_epochs"]):
        train_loss, train_metrics = train_one_epoch(
            model, base_model, train_loader, train_sampler,
            stage1_optimizer, ce_loss_fn, dice_loss_fn,
            device, config["model"]["num_classes"], epoch, stage=1,
            use_amp=use_amp, lb_factor=lb_factor, gs_tracker=None,
        )
        val_loss, val_metrics = validate_one_epoch(
            model, base_model, val_loader,
            ce_loss_fn, dice_loss_fn,
            device, config["model"]["num_classes"], use_amp=use_amp,
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

        if is_main_process():
            lb_avg = train_metrics.get('lb', 0.0)
            lb_str = f" train_lb={lb_avg:.5f}" if lb_factor > 0 else ""
            logging.info(
                f"[Stage1][Epoch {epoch+1}/{config['training']['stage1_epochs']}] "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_mIoU={val_metrics['iou']:.4f} val_mDice={val_metrics['dice']:.4f}"
                + lb_str
            )
            # Checkpoint & early stopping (rank 0 only)
            if stage1_es(val_metrics["iou"], base_model, epoch + 1):
                logging.info("Early stopping triggered for Stage 1.")
                should_stop = True

        # Broadcast early-stop decision from rank 0 to all ranks
        stop_tensor = torch.tensor(int(should_stop), device=device)
        dist.broadcast(stop_tensor, src=0)
        should_stop = bool(stop_tensor.item())

        if should_stop:
            break

    dist.barrier()

    # All ranks reload best Stage 1 weights
    stage1_best_path = stage1_es.path
    if os.path.exists(stage1_best_path):
        state = torch.load(stage1_best_path, map_location=device)
        if list(state.keys())[0].startswith("module."):
            state = {k.replace("module.", ""): v for k, v in state.items()}
        base_model.load_state_dict(state, strict=True)
        if is_main_process():
            logging.info(f"Loaded best Stage 1 weights from {stage1_best_path}")
    dist.barrier()

    # ======================================================================
    # Stage 2
    # ======================================================================
    stage2_epochs = config["training"].get("stage2_epochs", 0)
    if stage2_epochs <= 0:
        if is_main_process():
            logging.info("Stage 2 disabled (stage2_epochs=0). Training complete.")
    else:
        if is_main_process():
            logging.info("=== Stage 2: Fine-tuning with shared experts ===")

        shared_indices = select_shared_experts(config, base_model)
        if is_main_process():
            logging.info(f"Shared experts: {shared_indices}")
            # Initialize G_s tracker for Stage 2 (main process only, track stage 2 only)
            gs_tracker = GsTracker(output_dir, experiment_name="gs_tracking")
            gs_tracker.set_stage("stage2")
        else:
            gs_tracker = None

        if hasattr(base_model, "set_shared_experts"):
            base_model.set_shared_experts(shared_indices)

        stage2_optimizer = create_stage2_optimizer(model, base_model, config, shared_indices)
        stage2_scheduler = create_scheduler(stage2_optimizer, config, stage2_epochs)
        stage2_es = EarlyStopping(
            patience=early_cfg.get("patience", 10),
            delta=early_cfg.get("delta", 0.0),
            path=os.path.join(output_dir, "stage2_best_iou_model.pth"),
            metric_name="mIoU",
            mode="max",
        )

        should_stop = False
        for epoch in range(stage2_epochs):
            train_loss, train_metrics = train_one_epoch(
                model, base_model, train_loader, train_sampler,
                stage2_optimizer, ce_loss_fn, dice_loss_fn,
                device, config["model"]["num_classes"], epoch, stage=2,
                use_amp=use_amp, lb_factor=lb_factor, gs_tracker=gs_tracker,
            )
            val_loss, val_metrics = validate_one_epoch(
                model, base_model, val_loader,
                ce_loss_fn, dice_loss_fn,
                device, config["model"]["num_classes"], use_amp=use_amp,
            )

            if stage2_scheduler:
                if isinstance(stage2_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    stage2_scheduler.step(val_loss)
                else:
                    stage2_scheduler.step()

            if is_main_process():
                lb_avg = train_metrics.get('lb', 0.0)
                lb_str = f" train_lb={lb_avg:.5f}" if lb_factor > 0 else ""
                logging.info(
                    f"[Stage2][Epoch {epoch+1}/{stage2_epochs}] "
                    f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                    f"val_mIoU={val_metrics['iou']:.4f} val_mDice={val_metrics['dice']:.4f}"
                    + lb_str
                )
                if stage2_es(val_metrics["iou"], base_model, epoch + 1):
                    logging.info("Early stopping triggered for Stage 2.")
                    should_stop = True

            stop_tensor = torch.tensor(int(should_stop), device=device)
            dist.broadcast(stop_tensor, src=0)
            should_stop = bool(stop_tensor.item())

            if should_stop:
                break

        dist.barrier()
        if is_main_process():
            logging.info(f"Stage 2 best model saved at: {stage2_es.path}")
    
    # Generate G_s tracking report
    if gs_tracker and is_main_process():
        logging.info("\n" + "="*80)
        logging.info("Finalizing G_s tracking...")
        logging.info("="*80)
        gs_tracker.generate_report()
        logging.info("G_s tracking report generated successfully!")
        logging.info("="*80 + "\n")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDP 2-Stage Training for SAGE")
    parser.add_argument("--config",           type=str,   required=True, help="Path to YAML config file")
    parser.add_argument("--resume",           type=str,   default=None,  help="Path to stage1 checkpoint to resume from")
    parser.add_argument("--resume_epoch",     type=int,   default=0,     help="Epoch to resume from (0-indexed)")
    parser.add_argument("--resume_best_iou",  type=float, default=None,  help="Best val_mIoU so far (for EarlyStopping state)")
    args = parser.parse_args()
    main(args.config, resume_checkpoint=args.resume, resume_epoch=args.resume_epoch, resume_best_iou=args.resume_best_iou)
