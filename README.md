# SAGE: Shape-Adapting Gated Experts for Adaptive Histopathology Image Segmentation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **CVPR 2026 Findings Track**

Official PyTorch implementation of **SAGE**, a novel MoLEx architecture designed to handle variable shape, size, and texture topologies in medical image segmentation. As embodied in SAGE-ConvNeXt+ViT-UNet, it adapts automatically to diverse glandular structures by locally routing image patches to highly specialized convolution and transformer experts.

## 🚀 Key Features

- **Hybrid CNN-Transformer Encoder:** Leverages ConvNeXt and ViT blocks to capture both local edge details and global spatial relationships.
- **Hierarchical SAGE Routing:** Patches are dynamically routed to a pool of 28 specialized experts based on their semantic and structural density (e.g., normal tissue vs. irregular carcinoma variants).
- **SAHub (Shape-Adapting Hub) Decoder:** Reconstructs complex boundaries using cross-attention mechanisms conditioned on expert embeddings.
- **Two-Stage Training:**
  - *Stage 1:* Train the full model to allow experts to learn diverse shape specializations.
  - *Stage 2:* Freeze core shared representations and selectively fine-tune adaptive pathways with differentiated learning rates.

---

## 📁 Repository Structure

```
SAGE/
├── sage/                   # Core Python package (layers, architectures, utilities)
├── scripts/                # Entry points for single-GPU and multi-GPU (DDP) training
├── configs/                # YAML configurations for datasets and experiments
├── prepare_data/           # Scrips for automated Data pre-processing/stratified splits
├── tools/                  # Verification, Inference, and Metrics utilities
├── docs/                   # Detailed documentation and running guides
└── requirements.txt        # All python dependencies
```

---

## 📖 Documentation & Guides

Please see the following guides for detailed instructions on how to set up the environment, prepare datasets, and run training pipelines:

1. **[Quick Start & Full Running Guide](docs/RUNNING_GUIDE.md)** ✨ (Start Here!)
2. [Dataset Configuration Help](configs/datasets/)
3. [Contributing & Code Style](CONTRIBUTING.md) 

---

## 🔧 Brief Setup Instructions

Requirements can be easily installed via standard package managers. We recommend using a Conda environment:

```bash
conda create -n sage python=3.10
conda activate sage

pip install -r requirements.txt
```

*For step-by-step training guidance, dataset extraction instructions, and DDP setup, see the **[RUNNING_GUIDE.md](docs/RUNNING_GUIDE.md)**.*

---

## 🙏 Acknowledgements

We gratefully acknowledge The University of Texas at Austin for supporting this research, and Trivita AI and AI VIET NAM for providing the GPU computing resources essential to this work.

---

## 📝 Citation

If you find this research useful, please consider citing our work:

```bibtex
@article{thai2026sage,
  title={SAGE: Shape-Adapting Gated Experts for Adaptive Histopathology Image Segmentation},
  author={Thai, Gia Huy and Vu, Hoang-Nguyen and Phan, Anh-Minh and Ly, Quang-Thinh and Dinh, Tram and Nguyen, Thi-Ngoc-Truc and Ho, Nhat},
  journal={arXiv preprint arXiv:2511.18493},
  year={2026}
}
```

## 📄 License
This project is licensed under the MIT License.
