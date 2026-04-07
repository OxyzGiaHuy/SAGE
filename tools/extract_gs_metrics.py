"""
Utility functions to extract gs metrics from training logs and checkpoints.

This module provides utilities to:
1. Extract gs statistics from training logs
2. Load gs values from saved checkpoints
3. Prepare data for visualization
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings


class GSMetricsExtractor:
    """Extract and process gs metrics from training data."""
    
    def __init__(self, verbose=True):
        """Initialize the extractor.
        
        Args:
            verbose: Whether to print status messages
        """
        self.verbose = verbose
    
    def extract_from_log_file(self, log_file: str) -> Dict:
        """Extract gs metrics from a training log file.
        
        Expected log format (line-by-line):
        epoch,gs_mean,gs_std,gs_min,gs_max
        0,0.45,0.15,0.10,0.95
        1,0.48,0.14,0.12,0.93
        ...
        
        Args:
            log_file: Path to the training log file
            
        Returns:
            Dictionary with extracted metrics
        """
        data = {
            'epochs': [],
            'gs_mean': [],
            'gs_std': [],
            'gs_min': [],
            'gs_max': [],
        }
        
        try:
            with open(log_file, 'r') as f:
                # Skip header if present
                header = f.readline()
                
                for line in f:
                    if line.strip():
                        parts = line.strip().split(',')
                        if len(parts) >= 5:
                            data['epochs'].append(int(parts[0]))
                            data['gs_mean'].append(float(parts[1]))
                            data['gs_std'].append(float(parts[2]))
                            data['gs_min'].append(float(parts[3]))
                            data['gs_max'].append(float(parts[4]))
            
            # Convert to numpy arrays
            for key in data:
                data[key] = np.array(data[key])
            
            if self.verbose:
                print(f"✓ Extracted metrics from {log_file}")
                print(f"  Epochs: {len(data['epochs'])}, Mean range: [{data['gs_mean'].min():.3f}, {data['gs_mean'].max():.3f}]")
            
            return data
            
        except FileNotFoundError:
            print(f"✗ Log file not found: {log_file}")
            return None
    
    def extract_from_checkpoint(self, checkpoint_path: str, key: str = 'gs_stats') -> Optional[Dict]:
        """Extract gs metrics from a saved checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file (supports .pt, .pth, .pkl, .pickle)
            key: Key to look for in the checkpoint dictionary
            
        Returns:
            Dictionary with extracted metrics or None if not found
        """
        try:
            file_ext = Path(checkpoint_path).suffix.lower()
            
            if file_ext in ['.pt', '.pth']:
                import torch
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
            elif file_ext in ['.pkl', '.pickle']:
                with open(checkpoint_path, 'rb') as f:
                    checkpoint = pickle.load(f)
            else:
                print(f"✗ Unsupported file format: {file_ext}")
                return None
            
            if key in checkpoint:
                if self.verbose:
                    print(f"✓ Found '{key}' in checkpoint")
                return checkpoint[key]
            else:
                if self.verbose:
                    print(f"✗ Key '{key}' not found in checkpoint")
                    print(f"  Available keys: {list(checkpoint.keys())}")
                return None
                
        except Exception as e:
            print(f"✗ Error loading checkpoint: {e}")
            return None
    
    def extract_distribution(self, gs_values: np.ndarray, num_bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Extract histogram distribution of gs values.
        
        Args:
            gs_values: Array of gs values
            num_bins: Number of bins for histogram
            
        Returns:
            Tuple of (counts, bin_edges)
        """
        counts, bin_edges = np.histogram(gs_values, bins=num_bins, range=(0, 1))
        return counts, bin_edges
    
    def extract_layer_metrics(self, gs_per_layer: Dict[str, np.ndarray]) -> Dict:
        """Extract aggregated metrics for different layer types.
        
        Args:
            gs_per_layer: Dictionary mapping layer names to gs values arrays
                         e.g., {'CNN': [gs_values], 'Transformer': [gs_values]}
            
        Returns:
            Dictionary with aggregated statistics per layer type
        """
        metrics = {}
        
        for layer_name, gs_list in gs_per_layer.items():
            gs_array = np.array(gs_list)
            metrics[layer_name] = {
                'mean': float(np.mean(gs_array)),
                'std': float(np.std(gs_array)),
                'min': float(np.min(gs_array)),
                'max': float(np.max(gs_array)),
                'median': float(np.median(gs_array)),
                'count': len(gs_array),
            }
        
        if self.verbose:
            print("Layer-wise gs statistics:")
            for layer_name, stats in metrics.items():
                print(f"  {layer_name}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, "
                     f"min={stats['min']:.3f}, max={stats['max']:.3f}")
        
        return metrics
    
    def compare_initial_final(self, gs_initial: np.ndarray, 
                            gs_final: np.ndarray) -> Dict:
        """Compare initial and final gs distributions.
        
        Args:
            gs_initial: Array of initial gs values
            gs_final: Array of final gs values
            
        Returns:
            Dictionary with comparison statistics
        """
        comparison = {
            'initial': {
                'mean': float(np.mean(gs_initial)),
                'std': float(np.std(gs_initial)),
                'median': float(np.median(gs_initial)),
                'high_routing_ratio': float(np.sum(gs_initial > 0.5) / len(gs_initial)),
            },
            'final': {
                'mean': float(np.mean(gs_final)),
                'std': float(np.std(gs_final)),
                'median': float(np.median(gs_final)),
                'high_routing_ratio': float(np.sum(gs_final > 0.5) / len(gs_final)),
            },
            'change': {
                'mean_diff': float(np.mean(gs_final) - np.mean(gs_initial)),
                'std_change': float(np.std(gs_final) - np.std(gs_initial)),
                'high_routing_change': float(
                    np.sum(gs_final > 0.5) / len(gs_final) - 
                    np.sum(gs_initial > 0.5) / len(gs_initial)
                ),
            }
        }
        
        if self.verbose:
            print("Initial vs Final gs distribution:")
            print(f"  Initial - mean: {comparison['initial']['mean']:.3f}, "
                 f"ratio (>0.5): {comparison['initial']['high_routing_ratio']:.2%}")
            print(f"  Final   - mean: {comparison['final']['mean']:.3f}, "
                 f"ratio (>0.5): {comparison['final']['high_routing_ratio']:.2%}")
            print(f"  Change  - mean delta: {comparison['change']['mean_diff']:+.3f}, "
                 f"ratio delta: {comparison['change']['high_routing_change']:+.2%}")
        
        return comparison
    
    def prepare_visualization_data(self, 
                                  gs_initial: np.ndarray,
                                  gs_final: np.ndarray,
                                  gs_evolution: np.ndarray,  # Shape: (num_epochs,)
                                  gs_std_evolution: np.ndarray,  # Shape: (num_epochs,)
                                  cnn_evolution: np.ndarray,  # Shape: (num_epochs,)
                                  transformer_evolution: np.ndarray,  # Shape: (num_epochs,)
                                  ) -> Dict:
        """Prepare data for visualization.
        
        Args:
            gs_initial: Initial gs distribution
            gs_final: Final gs distribution
            gs_evolution: Mean gs per epoch
            gs_std_evolution: Std of gs per epoch
            cnn_evolution: Mean gs for CNN layers per epoch
            transformer_evolution: Mean gs for Transformer layers per epoch
            
        Returns:
            Dictionary ready for GSEvolutionVisualizer.visualize()
        """
        num_epochs = len(gs_evolution)
        
        data = {
            'gs_initial': np.array(gs_initial),
            'gs_final': np.array(gs_final),
            'epochs': np.arange(num_epochs),
            'gs_mean': np.array(gs_evolution),
            'gs_std': np.array(gs_std_evolution),
            'cnn_mean': np.array(cnn_evolution),
            'transformer_mean': np.array(transformer_evolution),
        }
        
        # Validate data ranges
        for key in ['gs_initial', 'gs_final', 'gs_mean', 'gs_std', 'cnn_mean', 'transformer_mean']:
            data[key] = np.clip(data[key], 0, 1)
        
        if self.verbose:
            print("✓ Data prepared for visualization")
            print(f"  Epochs: {num_epochs}")
            print(f"  Initial samples: {len(data['gs_initial'])}")
            print(f"  Final samples: {len(data['gs_final'])}")
        
        return data
    
    def save_metrics(self, metrics: Dict, output_path: str, format: str = 'json'):
        """Save extracted metrics to file.
        
        Args:
            metrics: Dictionary of metrics to save
            output_path: Path to save the metrics
            format: Format to use ('json' or 'pkl')
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format == 'json':
                with open(output_path, 'w') as f:
                    # Convert numpy types to native Python types for JSON
                    json_metrics = self._convert_to_serializable(metrics)
                    json.dump(json_metrics, f, indent=2)
            elif format == 'pkl':
                with open(output_path, 'wb') as f:
                    pickle.dump(metrics, f)
            else:
                print(f"✗ Unsupported format: {format}")
                return
            
            if self.verbose:
                print(f"✓ Metrics saved to {output_path}")
                
        except Exception as e:
            print(f"✗ Error saving metrics: {e}")
    
    @staticmethod
    def _convert_to_serializable(obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: GSMetricsExtractor._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [GSMetricsExtractor._convert_to_serializable(item) for item in obj]
        return obj


def main():
    """Example usage of GSMetricsExtractor."""
    
    print("GSMetricsExtractor - Example Usage")
    print("=" * 60)
    
    extractor = GSMetricsExtractor(verbose=True)
    
    # Example 1: Load from log file
    print("\nExample 1: Loading from log file")
    print("-" * 60)
    # metrics = extractor.extract_from_log_file('training_log.csv')
    
    # Example 2: Generate synthetic data for demonstration
    print("\nExample 2: Preparing synthetic data for visualization")
    print("-" * 60)
    
    np.random.seed(42)
    gs_initial = np.random.beta(3, 3, size=1000) * 0.6 + 0.2
    gs_final = np.concatenate([
        np.random.beta(2, 2, size=500) * 0.3 + 0.35,
        np.random.normal(loc=0.55, scale=0.15, size=500)
    ])
    gs_final = np.clip(gs_final, 0, 1)
    
    epochs = np.arange(60)
    gs_evolution = 0.5 + 0.08 * np.sin(epochs / 10)
    gs_std_evolution = 0.15 - 0.05 * (epochs / 60)
    cnn_evolution = 0.65 + 0.05 * np.sin(epochs / 8)
    transformer_evolution = 0.48 + 0.02 * np.sin(epochs / 6)
    
    # Prepare visualization data
    viz_data = extractor.prepare_visualization_data(
        gs_initial, gs_final, gs_evolution, 
        gs_std_evolution, cnn_evolution, transformer_evolution
    )
    
    # Compare initial and final
    comparison = extractor.compare_initial_final(gs_initial, gs_final)
    
    # Extract layer metrics
    layer_metrics = extractor.extract_layer_metrics({
        'CNN': gs_evolution,
        'Transformer': transformer_evolution,
    })
    
    # Save metrics
    all_metrics = {
        'comparison': comparison,
        'layer_metrics': layer_metrics,
    }
    # extractor.save_metrics(all_metrics, 'gs_metrics.json', format='json')
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")


if __name__ == '__main__':
    main()
