# SAGE Utility Tools

This directory contains standalone scripts for evaluating the model, analyzing checkpoint states, and producing rich visualizations (such as Whole Slide Image reconstructions, expert routing behaviors, and feature attention maps).

Below is a categorized list of tools and how to use them.

---

## 🏗️ Functional Verification & Debugging

- **`verify_model.py`**
  Runs an instantiation test and dummy forward pass confirming that all SAGE hierarchical routing modules and wrappers are properly bound.
  ```bash
  python tools/verify_model.py
  ```

- **`inspect_checkpoint.py`**
  Checks inner model weights from `.pth` files and ensures there are no `KeyError` issues around DataParallel modules or SAGE configuration conflicts.

---

## 🔭 Inference & Metric Extractions

- **`demo_e2e.py`**
  Fast End-to-End script demonstrating pure single inference pipeline. Loads a single testing split and measures throughput/accuracy.

- **`generate_wsi_overlay.py`**
  Standard script to stitch raw segmentation patch predictions back into their parent full-resolution Whole Slide Image (WSI) canvas. Overlays raw classifications directly onto real WSIs using a simple color code:
  - Green masks = True Positive
  - Red masks = False Positives & False Negatives
  ```bash
  python tools/generate_wsi_overlay.py \
      --config configs/experiments/sage_colon.yaml \
      --checkpoint /path/to/checkpoints/stage2_best_iou_model.pth \
      --test_img_dir /path/to/dataset/processed_colon/test/images \
      --test_mask_dir /path/to/dataset/processed_colon/test/masks \
      --output_dir wsi_reconstructions
  ```

- **`extract_gs_metrics.py`** / **`visualize_gs_evolution.py`**
  Used to ingest the metrics tracking dictionaries captured during routing stages (`GsTracker`) and plots load-balancing density curves across the epochs.

---

## 🎨 Expert Architecture Visualizations (Paper Ready)

These scripts analyze internal behavior inside the architecture directly:

- **`reproduce_visualization.py`**
  Reconstructs overarching visualization logic seen in the official paper figures (comparing baseline architectures to SAGE dynamic choices).

- **`visualize_expert_routing_decisions.py`**
  Hooks directly into `router` components inside `sage_layer` wrappers, yielding plots where each patch shows whether it was directed to a Transformer context expert vs. a local CNN edge expert.

- **`sar_query_key.py`**
  Extracts the inner mechanisms of the SAHub decoders by rendering cross-attention (Q, K, V) distributions. Useful to see how anatomical boundaries adapt back to their context.
