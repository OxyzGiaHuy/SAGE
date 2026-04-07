"""
Query-Key/Affinity visualization for SAR router (SAGE).

Loads a checkpoint, runs forward_with_routing_info on one image, and saves:
- Original image
- Prediction and overlay (optional mask)
- Affinity/bar plots for a selected routing layer
- Full-layer affinity heatmap when possible
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import os
import sys
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

# Project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sage.networks.convnext_transformer_unet import create_convnext_transformer_unet


# ---------- IO helpers ----------

def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def preprocess_image(image_path: str, img_size: int) -> Tuple[torch.Tensor, np.ndarray]:
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    tensor = transform(image)
    rgb = np.array(image.resize((img_size, img_size))) / 255.0
    return tensor, rgb


def load_mask(mask_path: str, img_size: int) -> np.ndarray:
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize((img_size, img_size), Image.NEAREST)
    mask_array = np.array(mask, dtype=np.uint8)
    return (mask_array > 0).astype(np.uint8)


def create_overlay(rgb: np.ndarray, pred: np.ndarray, gt: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    tp = (pred == 1) & (gt == 1)
    fp = (pred == 1) & (gt == 0)
    fn = (pred == 0) & (gt == 1)
    overlay = np.zeros_like(rgb)
    overlay[tp] = [0, 1, 0]
    overlay[fp | fn] = [1, 0, 0]
    blended = (1 - alpha) * rgb + alpha * overlay
    return (blended * 255).astype(np.uint8)


# ---------- Model load ----------

def load_trained_model(config_path: str, checkpoint_path: str, device: torch.device):
    cfg = load_config(config_path)
    use_sage = cfg["model"].get("use_sage", False)
    sage_cfg = cfg.get("sage", {}) if use_sage else None

    model = create_convnext_transformer_unet(
        num_classes=cfg["model"]["num_classes"],
        img_size=cfg["model"]["img_size"],
        convnext_variant=cfg["model"].get("convnext_variant", "convnext_base"),
        vit_variant=cfg["model"].get("vit_variant", "vit_base_patch32_224"),
        num_transformer_layers=cfg["model"].get("num_transformer_layers", 12),
        freeze_encoder=False,
        freeze_transformer=False,
        sage_config=sage_cfg,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    cleaned = OrderedDict()
    for k, v in state_dict.items():
        cleaned[k[7:]] = v if k.startswith("module.") else v
        if not k.startswith("module."):
            cleaned[k] = v

    model_state = model.state_dict()
    params_to_load = {}
    for name, tensor in cleaned.items():
        if name in model_state and model_state[name].shape == tensor.shape:
            params_to_load[name] = tensor

    model.load_state_dict(params_to_load, strict=False)
    model.to(device).eval()
    return model, cfg


# ---------- Routing helpers ----------

def flatten_routing(routing_infos: Dict) -> List[Dict]:
    if not isinstance(routing_infos, dict):
        return []
    flat = []
    flat.extend(routing_infos.get("cnn", []))
    flat.extend(routing_infos.get("transformer", []))
    return [r for r in flat if r]


def pick_routing(routing_infos: Dict, index: int) -> Optional[Dict]:
    flat = flatten_routing(routing_infos)
    if not flat:
        return None
    if 0 <= index < len(flat):
        return flat[index]
    return flat[0]


def extract_affinity(routing: Dict) -> Optional[np.ndarray]:
    if routing is None:
        return None
    if "eval_base_logits_sample_0" in routing:
        return torch.sigmoid(torch.tensor(routing["eval_base_logits_sample_0"])).numpy()
    if "eval_top_k_logits_sample_0" in routing:
        return torch.sigmoid(torch.tensor(routing["eval_top_k_logits_sample_0"])).numpy()
    if "affinity_scores" in routing:
        try:
            return np.array(routing["affinity_scores"])
        except Exception:
            return None
    return None


def extract_weights(routing: Dict) -> Optional[np.ndarray]:
    if routing is None:
        return None
    for key in ("gating_weights_sample_0", "top_k_logits", "raw_logits_sample_0"):
        if key in routing and routing[key] is not None:
            return np.array(routing[key])
    return None


# ---------- Plotting ----------

def plot_affinity_bar(affinity: Optional[np.ndarray], save_path: str, title: str):
    plt.figure(figsize=(8, 4))
    if affinity is None or affinity.size == 0:
        plt.text(0.5, 0.5, "No affinity data", ha="center")
    else:
        plt.bar(range(len(affinity)), affinity, color="steelblue")
        plt.xlabel("Expert index")
        plt.ylabel("Affinity score")
        plt.ylim(0, 1)
        plt.title(title)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_full_affinity_heatmap(routing_infos: Dict, save_path: str):
    flat = flatten_routing(routing_infos)
    if not flat:
        return
    matrices = []
    layer_labels = []
    for idx, r in enumerate(flat):
        aff = extract_affinity(r)
        if aff is None or aff.size == 0:
            continue
        matrices.append(aff.reshape(1, -1))
        layer_labels.append(f"Layer {idx}")
    if not matrices:
        return
    full = np.concatenate(matrices, axis=0)
    plt.figure(figsize=(10, 6))
    plt.imshow(full, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    plt.colorbar(label="Affinity score")
    plt.yticks(range(len(layer_labels)), layer_labels)
    plt.xticks(range(full.shape[1]), [f"E{i}" for i in range(full.shape[1])], rotation=45, ha="right")
    plt.title("Affinity heatmap (layers × experts)")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------- Main ----------

def run(config: str, checkpoint: str, image: str, output_dir: str, mask: Optional[str], routing_index: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_trained_model(config, checkpoint, device)
    img_size = cfg["data"]["img_size"]

    tensor, rgb = preprocess_image(image, img_size)
    gt_mask = None
    if mask and os.path.exists(mask):
        gt_mask = load_mask(mask, img_size)
    else:
        auto = os.path.splitext(image)[0] + "_mask.bmp"
        if os.path.exists(auto):
            gt_mask = load_mask(auto, img_size)

    tensor = tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        detailed = model.forward_with_routing_info(tensor)
        logits = detailed["logits"]
        routing = pick_routing(detailed.get("routing_infos", {}), routing_index)
        pred_mask = torch.softmax(logits, dim=1)[0].argmax(dim=0).cpu().numpy()

    if gt_mask is None:
        gt_mask = np.zeros_like(pred_mask)

    overlay = create_overlay(rgb, pred_mask, gt_mask, alpha=0.5)
    weights = extract_weights(routing)
    affinity = extract_affinity(routing)

    image_name = os.path.splitext(os.path.basename(image))[0]
    os.makedirs(output_dir, exist_ok=True)

    # Save overlays
    Image.fromarray((rgb * 255).astype(np.uint8)).save(os.path.join(output_dir, f"{image_name}_original.png"))
    Image.fromarray((pred_mask * 255).astype(np.uint8)).save(os.path.join(output_dir, f"{image_name}_prediction.png"))
    Image.fromarray((overlay)).save(os.path.join(output_dir, f"{image_name}_overlay.png"))

    # Save affinity bar
    plot_affinity_bar(affinity, os.path.join(output_dir, f"{image_name}_affinity_bar.png"), f"Layer {routing_index} affinity")
    # Save weights bar
    plot_affinity_bar(weights, os.path.join(output_dir, f"{image_name}_router_weights.png"), f"Layer {routing_index} weights")
    # Save full heatmap
    plot_full_affinity_heatmap(detailed.get("routing_infos", {}), os.path.join(output_dir, f"{image_name}_affinity_heatmap.png"))

    print(f"Saved SAR router query-key visualizations to: {output_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="SAR router query-key visualization (SAGE)")
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    p.add_argument("--image", required=True, help="Path to input image")
    p.add_argument("--output_dir", required=True, help="Directory to save outputs")
    p.add_argument("--mask", default=None, help="Optional path to GT mask")
    p.add_argument("--routing_index", type=int, default=0, help="Index into routing layers (cnn then transformer)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.config, args.checkpoint, args.image, args.output_dir, args.mask, args.routing_index)
