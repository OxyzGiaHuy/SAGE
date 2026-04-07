"""
Reproduction of Gating Score (g_s) Evolution Visualization
===========================================================
This script recreates the multi-panel figure showing:
- (a) Distribution Evolution: Violin plots comparing Initial vs Final g_s distributions
- (b) Overall g_s Evolution: Time series with mean and confidence bands
- (c) CNN vs Transformer Layers: Comparison of layer-specific means across epochs
"""

import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path

# ============================================================================
# 1. Load and Prepare Data
# ============================================================================

def load_gs_history(filepath):
    """Load gating score history from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def prepare_data(data):
    """
    Prepare data for visualization using only stage 2.
    
    Returns:
    - initial_dist: All g_s values from stage2, epoch 0
    - final_dist: All g_s values from stage2 (last epoch with data)
    - epoch_data: List of (cnn_mean, transformer_mean) tuples for each epoch
    - epoch_numbers: List of epoch indices
    """
    # Get initial distribution (stage2, epoch 0)
    stage2_epoch0 = data['stage2']['0']
    initial_dist = np.concatenate([
        stage2_epoch0['cnn_layers'],
        stage2_epoch0['transformer_layers']
    ])
    
    # Get final distribution (stage2, last epoch with data)
    stage2_keys = sorted([int(k) for k in data['stage2'].keys() if data['stage2'][k]])
    final_epoch = str(max(stage2_keys))
    stage2_final = data['stage2'][final_epoch]
    final_dist = np.concatenate([
        stage2_final['cnn_layers'],
        stage2_final['transformer_layers']
    ])
    
    # Prepare epoch-wise statistics for time series (stage2 only)
    epoch_data = []
    epoch_numbers = []
    
    for epoch_str in sorted([int(k) for k in data['stage2'].keys() if data['stage2'][k]]):
        epoch_str = str(epoch_str)
        stage2_epoch = data['stage2'][epoch_str]
        
        if stage2_epoch['cnn_layers']:  # Check if data exists
            cnn_mean = np.mean(stage2_epoch['cnn_layers'])
            transformer_mean = np.mean(stage2_epoch['transformer_layers']) if stage2_epoch['transformer_layers'] else 0
            
            epoch_data.append({
                'epoch': int(epoch_str),
                'cnn_mean': cnn_mean,
                'transformer_mean': transformer_mean,
                'all_values': np.concatenate([
                    stage2_epoch['cnn_layers'],
                    stage2_epoch['transformer_layers']
                ]) if stage2_epoch['transformer_layers'] else np.array(stage2_epoch['cnn_layers'])
            })
            epoch_numbers.append(int(epoch_str))
    
    return initial_dist, final_dist, epoch_data, epoch_numbers


# ============================================================================
# 2. Create Figure with Three Subplots
# ============================================================================

def create_visualization(data, output_path='gs_evolution_figure.pdf'):
    """
    Create the complete 3-panel visualization with enhanced styling.
    
    Parameters:
    - data: Dictionary with stage1 and stage2 gating scores
    - output_path: Path to save the figure
    """
    
    # Prepare data
    initial_dist, final_dist, epoch_data, _ = prepare_data(data)
    
    # Set enhanced style
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        # 'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 13,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'legend.fontsize': 15,
        'legend.framealpha': 0.95,
        'legend.edgecolor': 'gray',
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.linewidth': 0.8,
        'axes.linewidth': 1.2,
        'xtick.major.width': 1.2,
        'xtick.major.size': 5,
        'ytick.major.width': 1.2,
        'ytick.major.size': 5,
        'figure.dpi': 100,
    })
    
    # Create figure with 1 row, 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(36, 10))
    
    # Set background color
    fig.patch.set_facecolor('#ffffff')
    
    # # Add main title with better styling
    # fig.suptitle('Gating Score (g_s) Evolution Analysis', 
    #              fontsize=16, fontweight='bold', y=0.98, color='#1f1f1f')
    LPAD = 15

    # ========================================================================
    # SUBPLOT (a): Distribution Evolution - Violin Plot
    # ========================================================================
    
    ax_a = axes[0]
    ax_a.set_facecolor('#f8f9fa')
    
    # Prepare data for violin plot
    violin_data = [initial_dist, final_dist]
    labels = ['Initial', 'Final']
    
    # Create violin plot with custom colors
    parts = ax_a.violinplot(
        violin_data,
        positions=[1, 2],
        widths=0.7,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )
    
    # Customize violin colors with gradients
    colors = ['#e3f2fd', '#ffebee']  # Light blue and light red
    edges = ['#1976d2', '#d32f2f']   # Dark blue and dark red
    
    for pc, color, edge in zip(parts['bodies'], colors, edges):
        pc.set_facecolor(color)
        pc.set_edgecolor(edge)
        pc.set_alpha(0.85)
        pc.set_linewidth(2.0)
    
    # Add mean lines with better styling
    for i, dist in enumerate(violin_data, 1):
        mean_val = np.mean(dist)
        ax_a.plot([i - 0.25, i + 0.25], [mean_val, mean_val], 
                 color='#000000', linewidth=3, zorder=10)
        
        # Add a subtle circle marker at mean
        ax_a.scatter([i], [mean_val], s=120, color='white', edgecolors='#000000', 
                    linewidth=2.5, zorder=11, marker='o')
    
    # Add horizontal line at g_s = 0.5 with better styling
    ax_a.axhline(y=0.5, color='#ff6b6b', linestyle='--', linewidth=2.5, 
                label='g_s = 0.5', alpha=0.85, zorder=1)
    
    # Customize subplot (a)
    ax_a.set_xticks([1, 2])
    ax_a.set_xticklabels(labels, fontsize=13)
    ax_a.set_ylabel('$g_s$ score', fontsize=18, labelpad=10)
    ax_a.set_xlabel('Epoch', fontsize = 18, labelpad=LPAD)
    ax_a.set_ylim([0, 1])
    ax_a.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.8)
    ax_a.legend(loc='upper right', framealpha=0.95, edgecolor='gray', fancybox=True)
    ax_a.set_axisbelow(True)
    # Add title below plot
    ax_a.text(0.5, -0.20, '(a) Distribution Evolution', transform=ax_a.transAxes,
             fontsize=22, ha='center', color='#1f1f1f')
    
    # ========================================================================
    # SUBPLOT (b): Overall g_s Evolution - Time Series with Confidence Bands
    # ========================================================================
    
    ax_b = axes[1]
    ax_b.set_facecolor('#f8f9fa')
    
    # Compute statistics per epoch
    epochs = np.array([ep['epoch'] for ep in epoch_data])
    all_values_per_epoch = [ep['all_values'] for ep in epoch_data]
    
    means = np.array([np.mean(vals) for vals in all_values_per_epoch])
    stds = np.array([np.std(vals) for vals in all_values_per_epoch])
    
    # Plot confidence band (mean ± 1 std) with gradient
    ax_b.fill_between(epochs, means - stds, means + stds, 
                      color='#1976d2', alpha=0.25, label='±1 Std Dev', 
                      edgecolor='#1976d2', linewidth=0.5)
    
    # Add upper confidence band
    ax_b.fill_between(epochs, means + stds, 1.0, 
                      color='#1976d2', alpha=0.08)
    
    # Plot mean line with enhanced styling
    ax_b.plot(epochs, means, color='#1976d2', linewidth=3.5, 
             label='Mean $g_s$', zorder=2, marker='o', markersize=4, 
             markerfacecolor='#1976d2', markeredgecolor='white', markeredgewidth=1.5)
    
    # Add horizontal line at g_s = 0.5
    ax_b.axhline(y=0.5, color='#ff6b6b', linestyle='--', linewidth=2.5, alpha=0.85, zorder=1)
    
    # Customize subplot (b)
    ax_b.set_xlabel('Epoch', fontsize=18, labelpad=LPAD)
    ax_b.set_ylabel('$g_s$ score', fontsize=18, labelpad=10)
    ax_b.set_ylim([0, 1])
    ax_b.set_xlim([min(epochs), max(epochs)])
    ax_b.grid(axis='both', alpha=0.3, linestyle='-', linewidth=0.8)
    ax_b.legend(loc='best', framealpha=0.95, edgecolor='gray', fancybox=True)
    ax_b.set_axisbelow(True)
    # Add title below plot
    ax_b.text(0.5, -0.20, '(b) Overall g_s Evolution', transform=ax_b.transAxes,
             fontsize=22, ha='center', color='#1f1f1f')
    
    # Set x-axis to show ticks at intervals of 10
    ax_b.xaxis.set_major_locator(ticker.MultipleLocator(10))
    
    # ========================================================================
    # SUBPLOT (c): CNN vs Transformer Layers
    # ========================================================================
    
    ax_c = axes[2]
    ax_c.set_facecolor('#f8f9fa')
    
    # Extract CNN and Transformer means
    cnn_means = np.array([ep['cnn_mean'] for ep in epoch_data])
    transformer_means = np.array([ep['transformer_mean'] for ep in epoch_data])
    
    # Plot both lines with enhanced styling
    ax_c.plot(epochs, cnn_means, color='#2e7d32', linewidth=3.5, 
             marker='o', markersize=5, label='CNN Layers', alpha=0.9,
             markerfacecolor='#2e7d32', markeredgecolor='white', markeredgewidth=1.5)
    
    ax_c.plot(epochs, transformer_means, color='#f57c00', linewidth=3.5, 
             marker='s', markersize=5, label='Transformer Layers', alpha=0.9,
             markerfacecolor='#f57c00', markeredgecolor='white', markeredgewidth=1.5)
    
    # Add horizontal line at g_s = 0.5
    ax_c.axhline(y=0.5, color='#ff6b6b', linestyle='--', linewidth=2.5, alpha=0.85, zorder=1)
    
    # Customize subplot (c)
    ax_c.set_xlabel('Epoch', fontsize=18, labelpad=LPAD)
    ax_c.set_ylabel('Mean $g_s$ score', fontsize=18, labelpad=10)
    ax_c.set_ylim([0, 1])
    ax_c.set_xlim([min(epochs), max(epochs)])
    ax_c.grid(axis='both', alpha=0.3, linestyle='-', linewidth=0.8)
    ax_c.legend(loc='best', framealpha=0.95, edgecolor='gray', fancybox=True)
    ax_c.set_axisbelow(True)
    # Add title below plot
    ax_c.text(0.5, -0.20, '(c) CNN vs Transformer Layers', transform=ax_c.transAxes,
             fontsize=22, ha='center', color='#1f1f1f')
    
    # Set x-axis to show ticks at intervals of 10
    ax_c.xaxis.set_major_locator(ticker.MultipleLocator(10))
    
    # ========================================================================
    # Finalize and Save
    # ========================================================================
    
    # Adjust layout with optimized spacing
    plt.tight_layout(pad=3.0, w_pad=8.0, h_pad=0.5)
    
    # Save in multiple formats
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as PDF and PNG with high quality
    pdf_path = str(output_path)
    png_path = pdf_path.replace('.pdf', '.png')
    
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf', 
               facecolor='white', edgecolor='none')
    print(f"✓ Figure saved to: {pdf_path}")
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight', format='png',
               facecolor='white', edgecolor='none')
    print(f"✓ Figure saved to: {png_path}")
    
    plt.show()
    
    return fig, axes


# ============================================================================
# 3. Main Execution
# ============================================================================

if __name__ == "__main__":
    # Load data
    data_path = Path(__file__).parent / 'gs_history.json'
    print(f"Loading data from: {data_path}")
    data = load_gs_history(data_path)
    
    # Create visualization
    output_path = Path(__file__).parent / 'gs_evolution_figure.pdf'
    fig, axes = create_visualization(data, output_path=str(output_path))
    
    print("\n✓ Visualization complete!")
