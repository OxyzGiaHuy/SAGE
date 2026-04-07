# 📖 Running Guide: SAGE

This document provides step-by-step instructions for reproducing the SAGE framework, from environment setup and data preparation to model training and verification.

---

## 1. Environment Setup

The SAGE framework (embodied as SAGE-ConvNeXt+ViT-UNet) codebase is built on **PyTorch > 2.0** and uses `timm` for pretrained Vision Transformers and ConvNeXt components. 

To set up the environment, we recommend using Conda:

```bash
# 1. Create a Conda environment
conda create -n sage python=3.10 -y
conda activate sage

# 2. Install PyTorch (ensure your CUDA version matches, example below is for CUDA 11.8 / 12.1)
# See https://pytorch.org/get-started/locally/ for the exact command for your system
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 3. Install required project packages
pip install -r requirements.txt
```

---

## 2. Setting Up Datasets

SAGE depends on histopathology segmentation datasets like **GlaS** (Glandular Structures), **Colon**, and **EBHI-SEG**. 

### Raw Data Location
By default, the automated scripts assume your raw / preprocessed dataset folder is located at `/path/to/dataset/`. If you wish to change where the processed data lands (or where it is read from), you can modify the `root_dir` fields in:
- `configs/datasets/glas.yaml`
- `configs/datasets/colon.yaml`
- `configs/datasets/ebhi.yaml`

### Running the Data Preparation Pipeline

We provide a specialized bash wrapper to handle the stratified splitting, resizing, and directory initialization required by our framework. Run the following command from the root of the repository:

```bash
# Prepare a single dataset:
./prepare_data/prepare_data.sh glas    # or colon, ebhi

# Prepare all datasets sequentially:
./prepare_data/prepare_data.sh all
```

This script will process the raw files and construct `{train, val, test}` subdirectories inside `{root_dir}/processed_<dataset_name>`.

---

## 3. Structural Verification Check

Before initializing expensive training loops, we highly recommend running our automated structural verification script. It loads the SAGE architecture locally, verifies module injection (Mixture of Local Experts layer wrapping), and passes a dummy tensor to ensure no dimension mismatches occur.

```bash
python tools/verify_model.py
```

If the console outputs `✨ VERIFICATION SUCCESSFUL`, the environment and network factories are good to go!

---

## 4. Single-GPU Training

Training in SAGE is done via an automated **Two-Stage** approach:
- **Stage 1**: Standard epoch iterations to allow all experts to specialize across differing spatial features.
- **Stage 2**: Core experts are systematically frozen, and routers / remaining specialized pathways are fine-tuned via differential learning rates.

All hyperparameters (including stage 1 & 2 distinct learning rates, batch sizes, and model configurations) are defined in `configs/experiments/`.

Run single-GPU training using `train_sage.py`:

```bash
# Example: Training on the GlaS dataset
CUDA_VISIBLE_DEVICES=0 python scripts/train_sage.py \
    --config configs/experiments/sage_glas.yaml
```

Logs, saved models (`.pth`), and tensor metrics are exported automatically to the `experiments/` directory grouped by timestamp.

---

## 5. Multi-GPU Training (DistributedDataParallel)

For advanced users and faster convergence, we provide `train_sage_ddp.py` implementing PyTorch's native `DistributedDataParallel`.

Use `torchrun` (PyTorch's recommended elastic launcher) to allocate multiple GPUs:

```bash
# Example: Training on exactly 4 GPUs
torchrun --nproc_per_node=4 scripts/train_sage_ddp.py \
    --config configs/experiments/sage_ebhi.yaml
```

*Note on DDP*: We recommend reducing the individual `batch_size` in your YAML config accordingly across devices. Effective Batch Size = `batch_size * nproc_per_node`.

---

## 6. Using Inference & Visualization Tools

All standalone testing, tracking metric extractions, and WSI (Whole Slide Image) overlays are available inside the `tools/` directory.

- **`tools/demo_e2e.py`**: A fast end-to-end inference script that creates side-by-side (Image / Raw Prediction / Ground Truth) previews.
- **`tools/visualize_expert_routing_decisions.py`**: Will render visual heatmaps showing how SAGE routers are distributing patch topologies against experts layer-by-layer. 
- **`tools/inspect_checkpoint.py`**: Useful to verify whether your Stage 2 fine-tuned weights successfully load over a raw initialized model.

Check `tools/README.md` for extended details.
