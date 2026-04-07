"""
G_s Tracker - Monitor Shared Expert Gate scores during STAGE 2 training

This module tracks the evolution of g_s (shared expert gate) scores
during Stage 2 (after expert selection) to understand how the model
balances between shared and dynamic experts.

Only tracks Stage 2 to save computational time during Stage 1.
"""

import os
import json
import logging
from typing import Dict, List, Optional
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns


class GsTracker:
    """
    Tracks g_s (Shared Expert Gate) scores during Stage 2 training.
    
    Generates 3 key visualizations:
    1. Violin plot: G_s distribution at beginning vs end of Stage 2
    2. Line chart: Mean G_s evolution with std deviation bands
    3. Bar plot: Average G_s for CNN layers vs Transformer layers
    """
    
    def __init__(self, output_dir: str, experiment_name: str = "gs_tracking"):
        """
        Args:
            output_dir: Directory to save tracking results
            experiment_name: Name for this tracking experiment
        """
        self.output_dir = os.path.join(output_dir, experiment_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Storage for g_s history (Stage 2 only)
        self.gs_history = defaultdict(list)  # gs_history[epoch] = all g_s scores
        self.gs_by_layer = defaultdict(lambda: {
            'cnn_layers': [],
            'transformer_layers': []
        })  # gs_by_layer[epoch] = {'cnn_layers': [...], 'transformer_layers': [...]}
        
        self.current_epoch = 0
        self.stage2_started = False
        self.first_epoch = None
        self.last_epoch = None
        
        logging.info(f"GsTracker initialized (STAGE 2 ONLY). Output: {self.output_dir}")
    
    def set_stage(self, stage: str):
        """
        Mark when Stage 2 starts.
        Only tracks data after Stage 2 begins.
        """
        if stage == "stage2":
            self.stage2_started = True
            logging.info("GsTracker: Stage 2 started - Tracking enabled")
    
    def collect_gs_from_epoch(self, routing_infos_list: List[Dict], epoch: int):
        """
        Collect g_s scores from all batches in an epoch.
        Only collects during Stage 2.
        
        Args:
            routing_infos_list: List of routing info dicts from all batches
            epoch: Current epoch number
        """
        # Skip if Stage 2 hasn't started
        if not self.stage2_started:
            return
        
        if not routing_infos_list:
            return
        
        self.current_epoch = epoch
        if self.first_epoch is None:
            self.first_epoch = epoch
        self.last_epoch = epoch
        
        # Collect g_s scores grouped by layer type
        all_gs_scores = []
        cnn_gs_scores = []
        transformer_gs_scores = []
        
        for routing_infos in routing_infos_list:
            if routing_infos is None:
                continue
            
            # Handle both formats: dict and list
            if isinstance(routing_infos, dict):
                # Format: {'cnn': [...], 'transformer': [...]}
                cnn_layer_infos = routing_infos.get('cnn', [])
                transformer_layer_infos = routing_infos.get('transformer', [])
                
                for info in cnn_layer_infos:
                    if info and isinstance(info, dict) and 'g_s_score_sample_0' in info:
                        g_s_val = float(info['g_s_score_sample_0'])
                        cnn_gs_scores.append(g_s_val)
                        all_gs_scores.append(g_s_val)
                
                for info in transformer_layer_infos:
                    if info and isinstance(info, dict) and 'g_s_score_sample_0' in info:
                        g_s_val = float(info['g_s_score_sample_0'])
                        transformer_gs_scores.append(g_s_val)
                        all_gs_scores.append(g_s_val)
            else:
                # Legacy format: list of routing info dicts
                for info in routing_infos:
                    if info is None or 'g_s_score_sample_0' not in info:
                        continue
                    
                    g_s_score = float(info['g_s_score_sample_0'])
                    all_gs_scores.append(g_s_score)
                    
                    layer_type = info.get('layer_type', 'unknown').lower()
                    if 'cnn' in layer_type:
                        cnn_gs_scores.append(g_s_score)
                    elif 'transformer' in layer_type:
                        transformer_gs_scores.append(g_s_score)
        
        # Store data
        self.gs_history[epoch] = all_gs_scores
        self.gs_by_layer[epoch] = {
            'cnn_layers': cnn_gs_scores,
            'transformer_layers': transformer_gs_scores
        }
        
        # Log summary
        self.log_epoch_summary(epoch)
    
    def log_epoch_summary(self, epoch: int):
        """Log summary statistics for the current epoch."""
        if epoch not in self.gs_history:
            return
        
        all_scores = self.gs_history[epoch]
        cnn_scores = self.gs_by_layer[epoch]['cnn_layers']
        trans_scores = self.gs_by_layer[epoch]['transformer_layers']
        
        logging.info(f"\n{'='*80}")
        logging.info(f"G_s TRACKING SUMMARY - [STAGE 2] Epoch {epoch}")
        logging.info(f"{'='*80}")
        
        if all_scores:
            mean_gs = np.mean(all_scores)
            std_gs = np.std(all_scores)
            logging.info(f"Overall G_s: {mean_gs:.4f} ± {std_gs:.4f}")
            logging.info(f"Range: [{np.min(all_scores):.4f}, {np.max(all_scores):.4f}]")
        
        if cnn_scores:
            mean_cnn = np.mean(cnn_scores)
            logging.info(f"CNN Layers Mean G_s: {mean_cnn:.4f}")
        
        if trans_scores:
            mean_trans = np.mean(trans_scores)
            logging.info(f"Transformer Layers Mean G_s: {mean_trans:.4f}")
        
        logging.info(f"{'='*80}\n")
    
    def save_history(self):
        """Save g_s tracking history to JSON file."""
        output_file = os.path.join(self.output_dir, "gs_history_stage2.json")
        
        # Convert to serializable format
        serializable_history = {}
        for epoch, scores in self.gs_history.items():
            serializable_history[str(epoch)] = {
                'all_scores': scores,
                'cnn_layers': self.gs_by_layer[epoch]['cnn_layers'],
                'transformer_layers': self.gs_by_layer[epoch]['transformer_layers'],
                'mean': float(np.mean(scores)) if scores else 0.0,
                'std': float(np.std(scores)) if scores else 0.0
            }
        
        with open(output_file, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        logging.info(f"G_s history saved to: {output_file}")
    
    def plot_violin_plot_comparison(self):
        """
        Violin plot comparing g_s distribution at start and end of Stage 2.
        """
        if not self.gs_history or self.first_epoch is None or self.last_epoch is None:
            logging.warning("Not enough data for violin plot comparison")
            return
        
        first_scores = self.gs_history.get(self.first_epoch, [])
        last_scores = self.gs_history.get(self.last_epoch, [])
        
        if not first_scores or not last_scores:
            logging.warning("Missing data for violin plot")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data for violin plot
        data_to_plot = [first_scores, last_scores]
        labels = [f'Epoch {self.first_epoch}\n(Stage 2 Start)', 
                  f'Epoch {self.last_epoch}\n(Stage 2 End)']
        
        # Create violin plot
        parts = ax.violinplot(data_to_plot, positions=[0, 1], showmeans=True, showextrema=True)
        
        # Customize colors
        for pc in parts['bodies']:
            pc.set_facecolor('skyblue')
            pc.set_alpha(0.7)
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylabel('G_s Score', fontsize=12)
        ax.set_title('G_s Distribution: Beginning vs End of Stage 2', fontsize=14, fontweight='bold')
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=2, label='G_s = 0.5')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
        
        plt.tight_layout()
        output_file = os.path.join(self.output_dir, "gs_violin_plot.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Violin plot saved to: {output_file}")
    
    def plot_mean_gs_with_std(self):
        """
        Line chart showing mean g_s evolution with std deviation bands.
        """
        if not self.gs_history:
            logging.warning("No data for mean G_s plot")
            return
        
        epochs = sorted(self.gs_history.keys())
        means = []
        stds = []
        
        for epoch in epochs:
            scores = self.gs_history[epoch]
            if scores:
                means.append(np.mean(scores))
                stds.append(np.std(scores))
            else:
                means.append(np.nan)
                stds.append(0)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Convert to numpy arrays
        epochs_arr = np.array(epochs)
        means_arr = np.array(means)
        stds_arr = np.array(stds)
        
        # Plot line with std bands
        ax.plot(epochs_arr, means_arr, 'b-', linewidth=2.5, label='Mean G_s')
        ax.fill_between(
            epochs_arr,
            means_arr - stds_arr,
            means_arr + stds_arr,
            alpha=0.3,
            color='blue',
            label='±1 Std Dev'
        )
        
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, linewidth=2, label='G_s = 0.5 (Balanced)')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('G_s Score', fontsize=12)
        ax.set_title('Mean G_s Evolution During Stage 2', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        plt.tight_layout()
        output_file = os.path.join(self.output_dir, "gs_mean_evolution.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Mean G_s evolution plot saved to: {output_file}")
    
    def plot_layer_comparison(self):
        """
        Bar plot comparing average g_s for CNN layers vs Transformer layers.
        """
        if not self.gs_by_layer:
            logging.warning("No data for layer comparison")
            return
        
        # Calculate average g_s for each layer type across all epochs
        all_cnn_scores = []
        all_trans_scores = []
        
        for epoch, layer_data in self.gs_by_layer.items():
            all_cnn_scores.extend(layer_data['cnn_layers'])
            all_trans_scores.extend(layer_data['transformer_layers'])
        
        if not all_cnn_scores and not all_trans_scores:
            logging.warning("No CNN or Transformer layer data")
            return
        
        # Compute statistics
        stats = {}
        if all_cnn_scores:
            stats['CNN Layers'] = {
                'mean': np.mean(all_cnn_scores),
                'std': np.std(all_cnn_scores),
                'count': len(all_cnn_scores)
            }
        
        if all_trans_scores:
            stats['Transformer Layers'] = {
                'mean': np.mean(all_trans_scores),
                'std': np.std(all_trans_scores),
                'count': len(all_trans_scores)
            }
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        layer_types = list(stats.keys())
        means = [stats[lt]['mean'] for lt in layer_types]
        stds = [stats[lt]['std'] for lt in layer_types]
        
        colors = ['#2ecc71', '#e74c3c']  # Green for CNN, Red for Transformer
        bars = ax.bar(layer_types, means, yerr=stds, capsize=10, alpha=0.7, 
                      color=colors, edgecolor='black', linewidth=1.5, error_kw={'linewidth': 2})
        
        # Add value labels on bars
        for i, (bar, mean) in enumerate(zip(bars, means)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean:.4f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=2, label='G_s = 0.5 (Balanced)')
        ax.set_ylabel('Average G_s Score', fontsize=12)
        ax.set_title('Average G_s: CNN vs Transformer Layers (Stage 2)', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
        
        # Add sample count as text
        for i, lt in enumerate(layer_types):
            count = stats[lt]['count']
            ax.text(i, -0.12, f'n={count}', ha='center', fontsize=10, transform=ax.get_xaxis_transform())
        
        plt.tight_layout()
        output_file = os.path.join(self.output_dir, "gs_layer_comparison.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Layer comparison plot saved to: {output_file}")
    
    def generate_report(self):
        """Generate comprehensive report with all visualizations and statistics."""
        if not self.stage2_started or not self.gs_history:
            logging.warning("No Stage 2 data collected. Skipping report generation.")
            return
        
        logging.info("\n" + "="*80)
        logging.info("GENERATING G_S TRACKING REPORT (STAGE 2)")
        logging.info("="*80)
        
        # Save history
        self.save_history()
        
        # Generate the 3 main plots
        logging.info("\nGenerating visualizations...")
        try:
            self.plot_violin_plot_comparison()
            self.plot_mean_gs_with_std()
            self.plot_layer_comparison()
        except Exception as e:
            logging.error(f"Error generating plots: {e}")
            import traceback
            traceback.print_exc()
        
        # Generate summary statistics
        self._generate_summary_report()
        
        logging.info(f"\nG_s tracking report completed. Results saved to: {self.output_dir}")
        logging.info("="*80 + "\n")
    
    def _generate_summary_report(self):
        """Generate text summary report."""
        report_file = os.path.join(self.output_dir, "gs_summary_report.txt")
        
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("G_s TRACKING SUMMARY REPORT - STAGE 2\n")
            f.write("="*80 + "\n\n")
            
            if not self.gs_history:
                f.write("No data collected during Stage 2.\n")
                return
            
            epochs = sorted(self.gs_history.keys())
            f.write(f"Total Epochs Tracked: {len(epochs)}\n")
            f.write(f"Epoch Range: {self.first_epoch} to {self.last_epoch}\n\n")
            
            # Overall statistics
            all_scores = []
            for scores in self.gs_history.values():
                all_scores.extend(scores)
            
            if all_scores:
                f.write("OVERALL STATISTICS (All Stage 2):\n")
                f.write(f"  Mean G_s: {np.mean(all_scores):.4f}\n")
                f.write(f"  Std:      {np.std(all_scores):.4f}\n")
                f.write(f"  Min:      {np.min(all_scores):.4f}\n")
                f.write(f"  Max:      {np.max(all_scores):.4f}\n")
                f.write(f"  Median:   {np.median(all_scores):.4f}\n\n")
            
            # First epoch statistics
            first_scores = self.gs_history[self.first_epoch]
            f.write(f"EPOCH {self.first_epoch} (Stage 2 Start):\n")
            if first_scores:
                f.write(f"  Mean G_s: {np.mean(first_scores):.4f}\n")
                f.write(f"  Std:      {np.std(first_scores):.4f}\n\n")
            
            # Last epoch statistics
            last_scores = self.gs_history[self.last_epoch]
            f.write(f"EPOCH {self.last_epoch} (Stage 2 End):\n")
            if last_scores:
                f.write(f"  Mean G_s: {np.mean(last_scores):.4f}\n")
                f.write(f"  Std:      {np.std(last_scores):.4f}\n\n")
            
            # Change analysis
            if first_scores and last_scores:
                mean_change = np.mean(last_scores) - np.mean(first_scores)
                f.write("G_s CHANGE (Stage 2 End - Stage 2 Start):\n")
                f.write(f"  Mean Change: {mean_change:+.4f}\n")
                if np.mean(first_scores) > 0:
                    pct_change = (mean_change / np.mean(first_scores)) * 100
                    f.write(f"  Percent Change: {pct_change:+.2f}%\n\n")
            
            # Layer analysis
            all_cnn = []
            all_trans = []
            for layer_data in self.gs_by_layer.values():
                all_cnn.extend(layer_data['cnn_layers'])
                all_trans.extend(layer_data['transformer_layers'])
            
            f.write("LAYER TYPE ANALYSIS:\n")
            if all_cnn:
                f.write(f"  CNN Layers Average G_s: {np.mean(all_cnn):.4f}\n")
            if all_trans:
                f.write(f"  Transformer Layers Average G_s: {np.mean(all_trans):.4f}\n")
            
            f.write("\n" + "="*80 + "\n")
        
        logging.info(f"Summary report saved to: {report_file}")
