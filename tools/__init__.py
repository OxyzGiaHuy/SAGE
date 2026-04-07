"""
SAGE Visualization Module
==============================
Công cụ visualization đơn giản và mạnh mẽ cho SAGE.

Main Tool:
    - visualize_sar_xai.py: SAR Router + XAI Visualization

Features:
    1. Expert Affinity Heatmap
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    2. Expert Selection Pattern  
    3. XAI Attention Maps
    4. Prediction Overlay

Usage:
    python visualization/visualize_sar_xai.py \\
        --checkpoint ./checkpoints/model.pt \\
        --dataset_config ./config/config.yaml \\
        --output_dir ./outputs
"""

__version__ = "1.0.0"
__author__ = "SAGE Team"

__all__ = [
    "visualize_sar_xai",
    "sar_router_cli",
    "sar_query_key",
]
