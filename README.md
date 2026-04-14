<p align="center">
  <h1 align="center"><ins>SAGE</ins> 🌿<br>Shape-Adapting Gated Experts for Adaptive Histopathology Image Segmentation</h1>
  <h3 align="center">CVPR 2026 Findings Track</h3>
  <p align="center">
    <span class="author-block">
      <a href="https://orcid.org/0009-0006-6684-5323">Gia Huy Thai</a><sup>1,*</sup>
      ·
      <a href="https://scholar.google.com/citations?user=nGUt6_oAAAAJ&hl=en">Hoang-Nguyen Vu</a><sup>2,*</sup>
      ·
      <a href="https://scholar.google.com/citations?user=JCbuDwcAAAAJ&hl=vi">Anh-Minh Phan</a><sup>3</sup>
      ·
      <a href="https://orcid.org/0009-0006-9039-5887">Quang-Thinh Ly</a><sup>4</sup>
      ·
      <a href="https://scholar.google.com/citations?user=rTa5hJwAAAAJ&hl=en">Thi-Ngoc-Truc Nguyen</a><sup>2</sup>
      ·
      <a href="https://scholar.google.com.vn/citations?user=Xs7cKMwAAAAJ&hl=vi">Nhat Ho</a><sup>5</sup>
    </span>
  </p>
  <p align="center">
    <sup>1</sup>University of Science, VNU-HCM &nbsp;&nbsp;
    <sup>2</sup>Trivita AI &nbsp;&nbsp;
    <sup>3</sup>University of Technology, VNU-HCM &nbsp;&nbsp;
    <sup>4</sup>Michigan State University, USA &nbsp;&nbsp;
    <sup>5</sup>The University of Texas at Austin
  </p>
  <p align="center">
    <sup>*</sup>equal contribution
  </p>
  <div align="center">
  
  [![arXiv](https://img.shields.io/badge/arXiv-2511.18493-b31b1b.svg)](https://arxiv.org/abs/2511.18493)
  [![Paper](https://img.shields.io/badge/Paper-CVPR%202026-blue?logo=adobeacrobatreader)](./sage_paper.pdf)
  [![Project Page](https://img.shields.io/badge/Project-Page-brightgreen)](https://oxyzgiahuy.github.io/sage/)
  [![Code](https://img.shields.io/badge/GitHub-Code-black?logo=github)](https://github.com/OxyzGiaHuy/SAGE)
  [![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
  [![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
  
  </div>
</p>

---

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
