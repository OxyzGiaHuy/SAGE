#!/usr/bin/env python3
"""
Expert Routing Visualization
- Affinity heatmap (scores per expert per layer)
- Selection map (top-k experts per layer)
"""

import os
import sys
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image
from torchvision import transforms
import argparse
from collections import OrderedDict
import yaml

sage_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, sage_root)

from sage.networks.convnext_transformer_unet import create_convnext_transformer_unet


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def preprocess_image(image_path: str, img_size: int):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    tensor = transform(image)
    rgb = np.array(image.resize((img_size, img_size))) / 255.0
    return tensor, rgb


def load_trained_model(config_path: str, checkpoint_path: str, device):
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
        if k.startswith("module."):
            cleaned[k[7:]] = v
        else:
            cleaned[k] = v

    model_state = model.state_dict()
    params_to_load = {}
    for name, tensor in cleaned.items():
        if name in model_state and model_state[name].shape == tensor.shape:
            params_to_load[name] = tensor

    model.load_state_dict(params_to_load, strict=False)
    model.to(device).eval()
    return model, cfg


def flatten_routing(routing_infos):
    if not isinstance(routing_infos, dict):
        return []
    flat = []
    flat.extend(routing_infos.get("cnn", []))
    flat.extend(routing_infos.get("transformer", []))
    return [r for r in flat if r]


def extract_affinity(routing):
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


def create_expert_routing_mask(affinity, top_k: int = 4):
    """Create binary mask: 1 = selected expert, 0 = not selected."""
    if affinity is None or affinity.size == 0:
        return None
    num_experts = len(affinity)
    top_k = min(top_k, num_experts)
    mask = np.zeros(num_experts, dtype=np.uint8)
    top_k_indices = np.argsort(affinity)[-top_k:][::-1]
    mask[top_k_indices] = 1
    return mask


def _set_plot_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.dpi": 160,
            "savefig.dpi": 320,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
        }
    )


def _affinity_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    # Custom gradient to match the provided palette
    # Low score -> light peach, high score -> deep purple/black
    colors = ["#f6d8c7", "#f0ad84", "#e06e6a", "#c0446f", "#6b2364", "#1a0b2a"]
    return LinearSegmentedColormap.from_list("affinity_custom", colors, N=256)


def _build_layer_labels(num_layers):
    labels = []
    for idx in range(num_layers):
        layer_type = "CNN" if idx < 4 else "ViT"
        labels.append(f"L{idx} ({layer_type})")
    return labels


def plot_affinity_heatmap(
    routing_infos,
    save_path: str,
    title_suffix: str = "",
    cmap=None,
    annotate: bool = False,
):
    """Create affinity score heatmap (per expert per layer)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _set_plot_style()

    flat = flatten_routing(routing_infos)
    if not flat:
        return 0

    vectors = []
    for r in flat:
        aff = extract_affinity(r)
        if aff is None or aff.size == 0:
            continue
        vectors.append(np.asarray(aff).reshape(-1))

    if not vectors:
        return 0

    max_len = max(v.size for v in vectors)
    matrix = np.full((len(vectors), max_len), np.nan, dtype=float)
    for i, v in enumerate(vectors):
        matrix[i, : v.size] = v

    mask = np.isnan(matrix)
    masked = np.ma.masked_invalid(matrix)

    fig, ax = plt.subplots(figsize=(15, 9))
    if cmap is None:
        # Flip palette direction: low -> dark, high -> light
        cmap = _affinity_cmap().reversed()

    im = ax.imshow(
        masked,
        aspect="auto",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )

    # Light grid for readability
    ax.set_xticks(np.arange(matrix.shape[1]) - 0.5, minor=True)
    ax.set_yticks(np.arange(matrix.shape[0]) - 0.5, minor=True)
    ax.grid(which="minor", color="#e6e6e6", linestyle="-", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Optional annotations
    if annotate:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if mask[i, j]:
                    continue
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#111111",
                )

    num_layers = matrix.shape[0]
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels(_build_layer_labels(num_layers))
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels([f"E{i}" for i in range(matrix.shape[1])], rotation=0)

    ax.set_xlabel("Expert Index", fontsize=18)
    ax.set_ylabel("Routing Layer", fontsize=18)

    title_text = "Expert Affinity Heatmap"
    if title_suffix:
        title_text += f"\n{title_suffix}"
    # ax.set_title(title_text, pad=16)

    # Clean look
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Affinity Score (sigmoid)", fontsize=18, fontweight="normal", labelpad=10)
    cbar.ax.tick_params(labelsize=9)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return num_layers


def plot_full_expert_routing_heatmap(routing_infos, save_path: str, top_k: int = 4, title_suffix: str = ""):
    """Create binary heatmap showing which experts are selected (1) vs not selected (0)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    _set_plot_style()
    
    flat = flatten_routing(routing_infos)
    if not flat:
        return 0
    
    matrices = []
    layer_labels = []
    for idx, r in enumerate(flat):
        aff = extract_affinity(r)
        if aff is None or aff.size == 0:
            continue
        mask = create_expert_routing_mask(aff, top_k)
        matrices.append(mask.reshape(1, -1))
        layer_labels.append(f"L{idx}")
    
    if not matrices:
        return 0
    
    full = np.concatenate(matrices, axis=0).astype(float)
    
    # Eye-friendly color scheme: Light background, vibrant accent colors
    # Not selected: Light gray (#e8eaed), Selected: Vibrant blue (#1f77b4)
    cmap = ListedColormap(["#e8eaed", "#0b3a6d"])  # light gray, deep blue
    norm = BoundaryNorm([0, 1, 2], cmap.N)
    
    fig, ax = plt.subplots(figsize=(15, 9))
    im = ax.imshow(full, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    
    # Labels - CNN vs Transformer
    # CNN: L0-L3 (4 layers from ConvNeXt-Large)
    # ViT: L4-L27 (24 layers from Vision Transformer)
    layer_colors = []
    for label in layer_labels:
        layer_idx = int(label[1:])
        # First 4 are CNN, rest are Transformer
        layer_colors.append('#d62728' if layer_idx < 4 else '#2ca02c')
    
    # Create custom yticklabels with layer type
    yticklabels_custom = []
    for label in layer_labels:
        layer_idx = int(label[1:])
        layer_type = "CNN" if layer_idx < 4 else "ViT"
        yticklabels_custom.append(f"{label} ({layer_type})")
    
    ax.set_yticks(range(len(layer_labels)))
    ax.set_yticklabels(yticklabels_custom, fontsize=9)
    ax.set_xticks(range(full.shape[1]))
    ax.set_xticklabels([f"E{i}" for i in range(full.shape[1])], rotation=0, ha="center", fontsize=10)
    
    # Add grid with subtle color
    ax.set_xticks(np.arange(full.shape[1]) - 0.5, minor=True)
    ax.set_yticks(np.arange(full.shape[0]) - 0.5, minor=True)
    ax.grid(which="minor", color="#000000", linestyle="-", linewidth=0.5)
    
    ax.set_xlabel("Expert Index", fontsize=18, fontweight="normal", labelpad=8)
    ax.set_ylabel("Routing Layer", fontsize=18, fontweight="normal", labelpad=8)
    
    title_text = f"Expert Routing Decision Map (top-{top_k} selected per layer)"
    if title_suffix:
        title_text += f"\n{title_suffix}"
    # ax.set_title(title_text, fontsize=14, fontweight="normal", pad=20)

    # Square legend blocks (no colorbar)
    legend_handles = [
        Patch(facecolor="#0b3a6d", edgecolor="none", label="Selected"),
        Patch(facecolor="#e8eaed", edgecolor="none", label="Not Selected"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(1.3, 1.0),
        frameon=False,
        handlelength=4.0,   # increase these to make legend squares bigger
        handleheight=4.0,
        borderpad=0.4,
        labelspacing=0.8,
        fontsize=14,
    )
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    
    return len(layer_labels)


def generate_routing_table(routing_infos, image_name: str, output_dir: str, top_k: int = 4):
    """Generate text report of expert routing decisions."""
    flat = flatten_routing(routing_infos)
    
    report_path = os.path.join(output_dir, f"{image_name}_expert_routing_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("EXPERT ROUTING DECISION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total routing layers: {len(flat)}\n")
        f.write(f"Top-K: {top_k}\n\n")
        
        for idx, r in enumerate(flat):
            aff = extract_affinity(r)
            if aff is None or aff.size == 0:
                continue
            
            # CNN: L0-L3 (4 layers), ViT: L4-L27 (24 layers)
            layer_type = "CNN" if idx < 4 else "ViT"
            mask = create_expert_routing_mask(aff, top_k)
            top_k_indices = np.argsort(aff)[-top_k:][::-1]
            
            f.write(f"\nLayer {idx} ({layer_type}):\n")
            f.write(f"  Selected experts: {list(top_k_indices)}\n")
            f.write(f"  Not selected experts: {list(np.where(mask == 0)[0])}\n")
            f.write(f"  Affinity scores:\n")
            for i in range(len(aff)):
                status = "✓ SELECTED" if mask[i] == 1 else "✗ not selected"
                f.write(f"    Expert {i:2d}: {aff[i]:.4f} {status}\n")
    
    return report_path


def run(config_path: str, checkpoint_path: str, image_path: str, output_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model, cfg = load_trained_model(config_path, checkpoint_path, device)
    img_size = cfg["data"]["img_size"]

    tensor, rgb = preprocess_image(image_path, img_size)
    tensor = tensor.unsqueeze(0).to(device)

    print("Running forward pass with routing info...")
    with torch.no_grad():
        detailed = model.forward_with_routing_info(tensor)
        routing_infos = detailed.get("routing_infos", {})
    
    flat = flatten_routing(routing_infos)
    print(f"✅ Found {len(flat)} routing layers\n")

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    top_k = cfg.get("sage", {}).get("top_k", 4)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Affinity heatmap
    print("Generating expert affinity heatmap...")
    affinity_map_path = os.path.join(output_dir, f"{image_name}_affinity_heatmap.png")
    num_layers = plot_affinity_heatmap(
        routing_infos,
        affinity_map_path,
        # title_suffix=f"Image: {image_name}",
        cmap=None,
        annotate=False,
    )
    print(f"✅ Saved: {affinity_map_path}")
    print(f"   (Shows {num_layers} routing layers with affinity scores)\n")

    # Selection map (top-k)
    print("Generating expert routing decision map...")
    routing_map_path = os.path.join(output_dir, f"{image_name}_routing_decision_map.png")
    num_layers = plot_full_expert_routing_heatmap(
        routing_infos, routing_map_path, top_k, f"Image: {image_name}"
    )
    print(f"✅ Saved: {routing_map_path}")
    print(f"   (Shows {num_layers} routing layers with expert selection status)\n")
    
    # Generate text report
    print("Generating expert routing report...")
    report_path = generate_routing_table(routing_infos, image_name, output_dir, top_k)
    print(f"✅ Saved: {report_path}\n")
    
    # Save original image
    Image.fromarray((rgb * 255).astype(np.uint8)).save(
        os.path.join(output_dir, f"{image_name}_original.png")
    )
    print(f"✅ Saved original image\n")
    
    print("=" * 80)
    print(f"All outputs saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate expert routing decision maps from EBHI stage 2")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output_dir", required=True, help="Directory to save outputs")
    args = parser.parse_args()
    
    run(args.config, args.checkpoint, args.image, args.output_dir)
