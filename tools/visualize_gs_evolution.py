"""
Visualization of Shared Gating Scalar (gs) Evolution

This module creates comprehensive visualizations showing:
- (a) Distribution evolution of gs scores (initial vs final training)
- (b) Overall gs evolution over epochs with uncertainty bands
- (c) Comparison of gs values between CNN and Transformer layers
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')


class GSEvolutionVisualizer:
    """Visualizer for shared gating scalar evolution during training."""
    
    def __init__(self, figsize=(15, 4), style='seaborn-v0_8-darkgrid'):
        """Initialize the visualizer.
        
        Args:
            figsize: Tuple of (width, height) for the figure
            style: Matplotlib style to use
        """
        self.figsize = figsize
        self.style = style
        plt.style.use(style)
        
    def create_demo_data(self, num_samples=1000, num_epochs=60):
        """Create realistic demo data matching the figure description.
        
        Args:
            num_samples: Number of samples for distribution plots
            num_epochs: Number of training epochs
            
        Returns:
            Dictionary with demo data
        """
        np.random.seed(42)
        
        # Initial gs distribution (more concentrated, lower values)
        gs_initial = np.random.beta(a=3, b=3, size=num_samples) * 0.6 + 0.2
        
        # Final gs distribution (more bimodal, peaks around 0.5 and higher)
        gs_final = np.concatenate([
            np.random.beta(a=2, b=2, size=num_samples//2) * 0.3 + 0.35,
            np.random.normal(loc=0.55, scale=0.15, size=num_samples//2)
        ])
        gs_final = np.clip(gs_final, 0, 1)
        
        # Evolution over epochs with noise
        epochs = np.arange(num_epochs)
        gs_mean = 0.45 + 0.08 * np.sin(epochs / 10) + 0.01 * np.random.randn(num_epochs)
        gs_mean = np.clip(gs_mean, 0.3, 0.7)
        
        # Standard deviation evolution
        gs_std = 0.15 - 0.05 * (epochs / num_epochs) + 0.02 * np.abs(np.sin(epochs / 5))
        
        # CNN layers: high gs (favor shared experts)
        cnn_mean = 0.65 + 0.03 * np.sin(epochs / 8)
        cnn_mean = np.clip(cnn_mean, 0.6, 0.75)
        
        # Transformer layers: neutral gs
        transformer_mean = 0.48 + 0.02 * np.sin(epochs / 6)
        transformer_mean = np.clip(transformer_mean, 0.45, 0.52)
        
        return {
            'gs_initial': gs_initial,
            'gs_final': gs_final,
            'epochs': epochs,
            'gs_mean': gs_mean,
            'gs_std': gs_std,
            'cnn_mean': cnn_mean,
            'transformer_mean': transformer_mean,
        }
    
    def plot_distribution_evolution(self, ax, gs_initial, gs_final):
        """Plot (a) Distribution Evolution - violin plots.
        
        Args:
            ax: Matplotlib axes
            gs_initial: Initial gs scores
            gs_final: Final gs scores
        """
        data_to_plot = [gs_initial, gs_final]
        positions = [0, 1]
        
        # Create violin plots
        parts = ax.violinplot(data_to_plot, positions=positions, widths=0.7,
                              showmeans=True, showmedians=True)
        
        # Color the violin plots
        colors = ['#808080', '#CD7672']  # Gray and rose
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)
        
        # Style the other components
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
            if partname in parts:
                parts[partname].set_edgecolor('black')
                parts[partname].set_linewidth(1.5)
        
        # Add horizontal line at gs=0.5
        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='$g_s = 0.5$')
        
        # Customize
        ax.set_ylabel('gs Score', fontsize=11, fontweight='bold')
        ax.set_xticks(positions)
        ax.set_xticklabels(['Initial', 'Final'], fontsize=10)
        ax.set_ylim([0, 1.0])
        ax.set_title('(a) Distribution Evolution', fontsize=12, fontweight='bold', loc='left')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(loc='upper right', fontsize=9)
    
    def plot_overall_evolution(self, ax, epochs, gs_mean, gs_std):
        """Plot (b) Overall gs Evolution - line chart with uncertainty band.
        
        Args:
            ax: Matplotlib axes
            epochs: Array of epoch numbers
            gs_mean: Array of mean gs values
            gs_std: Array of standard deviation values
        """
        # Main line
        ax.plot(epochs, gs_mean, color='#4A90E2', linewidth=2.5, label='Mean $g_s$')
        
        # Uncertainty band (±1 std dev)
        ax.fill_between(epochs, gs_mean - gs_std, gs_mean + gs_std,
                       color='#ADD8E6', alpha=0.4, label='±1 Std Dev')
        
        # Neutral line
        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='$g_s = 0.5$')
        
        # Customize
        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_ylabel('Mean gs Score', fontsize=11, fontweight='bold')
        ax.set_xlim([0, epochs[-1]])
        ax.set_ylim([0, 1.0])
        ax.set_title('(b) Overall $g_s$ Evolution', fontsize=12, fontweight='bold', loc='left')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
    
    def plot_layer_comparison(self, ax, epochs, cnn_mean, transformer_mean):
        """Plot (c) CNN vs Transformer Layers comparison.
        
        Args:
            ax: Matplotlib axes
            epochs: Array of epoch numbers
            cnn_mean: Array of CNN layer gs values
            transformer_mean: Array of Transformer layer gs values
        """
        # CNN layers line
        ax.plot(epochs, cnn_mean, color='#2E7D32', linewidth=2.5, label='CNN Layers')
        
        # Transformer layers line
        ax.plot(epochs, transformer_mean, color='#D4AF37', linewidth=2.5, label='Transformer Layers')
        
        # Neutral line
        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='$g_s = 0.5$')
        
        # Customize
        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_ylabel('Mean gs Score', fontsize=11, fontweight='bold')
        ax.set_xlim([0, epochs[-1]])
        ax.set_ylim([0, 1.0])
        ax.set_title('(c) CNN vs Transformer Layers', fontsize=12, fontweight='bold', loc='left')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
    
    def visualize(self, data_dict=None, save_path='gs_evolution.png', dpi=300):
        """Create and save the complete visualization.
        
        Args:
            data_dict: Dictionary with keys: gs_initial, gs_final, epochs, gs_mean, gs_std,
                      cnn_mean, transformer_mean. If None, demo data is generated.
            save_path: Path to save the figure
            dpi: Resolution of the saved figure
        """
        if data_dict is None:
            data_dict = self.create_demo_data()
        
        # Create figure and subplots
        fig, axes = plt.subplots(1, 3, figsize=self.figsize)
        fig.suptitle('Evolution of Shared Gating Scalar ($g_s$)', 
                    fontsize=14, fontweight='bold', y=1.02)
        
        # Plot each subplot
        self.plot_distribution_evolution(
            axes[0], data_dict['gs_initial'], data_dict['gs_final']
        )
        
        self.plot_overall_evolution(
            axes[1], data_dict['epochs'], data_dict['gs_mean'], data_dict['gs_std']
        )
        
        self.plot_layer_comparison(
            axes[2], data_dict['epochs'], data_dict['cnn_mean'], 
            data_dict['transformer_mean']
        )
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {save_path}")
        
        return fig, axes


def main():
    """Main function to demonstrate the visualization."""
    
    # Create visualizer
    visualizer = GSEvolutionVisualizer(figsize=(15, 4))
    
    # Generate demo data
    data = visualizer.create_demo_data(num_samples=1000, num_epochs=60)
    
    # Create and save visualization
    fig, axes = visualizer.visualize(
        data_dict=data,
        save_path='gs_evolution.png',
        dpi=300
    )
    
    plt.show()


if __name__ == '__main__':
    main()
