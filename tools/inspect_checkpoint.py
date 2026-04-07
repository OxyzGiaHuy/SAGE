"""
Utility script to inspect checkpoint structure and extract gs metrics.

Use this to understand checkpoint format and locate gs-related data.
"""

import sys
import torch
from pathlib import Path
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



def inspect_checkpoint(checkpoint_path, verbose=True):
    """Inspect checkpoint structure.
    
    Args:
        checkpoint_path: Path to .pth checkpoint
        verbose: Whether to print detailed information
        
    Returns:
        Dictionary with checkpoint structure information
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"✗ Error loading checkpoint: {e}")
        return None
    
    info = {
        'type': type(checkpoint).__name__,
        'keys': [],
        'structure': {},
    }
    
    if isinstance(checkpoint, dict):
        for key, value in checkpoint.items():
            item_info = {
                'type': type(value).__name__,
                'size': None,
            }
            
            if isinstance(value, torch.Tensor):
                item_info['shape'] = tuple(value.shape)
                item_info['dtype'] = str(value.dtype)
            elif isinstance(value, dict):
                item_info['keys'] = list(value.keys())[:10]  # First 10 keys
                if len(value.keys()) > 10:
                    item_info['keys'].append('...')
            elif isinstance(value, (list, tuple)):
                item_info['length'] = len(value)
            
            info['structure'][key] = item_info
            info['keys'].append(key)
    
    if verbose:
        print(f"Checkpoint Type: {info['type']}")
        print(f"Top-level Keys ({len(info['keys'])}): {info['keys']}")
        print("\nDetailed Structure:")
        for key, item in info['structure'].items():
            print(f"\n  {key}:")
            for k, v in item.items():
                print(f"    {k}: {v}")
    
    return info


def find_gs_metrics(checkpoint_path):
    """Find gs-related data in checkpoint.
    
    Args:
        checkpoint_path: Path to .pth checkpoint
        
    Returns:
        List of tuples (path, type, description)
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"✗ Error loading checkpoint: {e}")
        return []
    
    results = []
    gs_keywords = ['gs', 'gating', 'routing', 'expert', 'moe', 'metrics', 'logs', 'history']
    
    def search_dict(d, prefix=''):
        for key, value in d.items():
            current_path = f"{prefix}.{key}" if prefix else key
            
            # Check if key matches gs-related keywords
            key_lower = key.lower()
            if any(keyword in key_lower for keyword in gs_keywords):
                results.append((
                    current_path,
                    type(value).__name__,
                    f"Matches keyword: {[kw for kw in gs_keywords if kw in key_lower]}"
                ))
            
            # Recursively search nested dicts
            if isinstance(value, dict):
                search_dict(value, current_path)
    
    if isinstance(checkpoint, dict):
        search_dict(checkpoint)
    
    return results


def extract_gs_data(checkpoint_path, extraction_path=None):
    """Try to extract gs data from checkpoint.
    
    Args:
        checkpoint_path: Path to .pth checkpoint
        extraction_path: Specific path in checkpoint to extract (e.g., 'logs.gs')
        
    Returns:
        Extracted data or None
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"✗ Error loading checkpoint: {e}")
        return None
    
    # Try to find gs data
    if extraction_path:
        keys = extraction_path.split('.')
        data = checkpoint
        for key in keys:
            if isinstance(data, dict) and key in data:
                data = data[key]
            else:
                print(f"✗ Path not found: {extraction_path}")
                return None
        return data
    
    # Try common paths
    common_paths = [
        'gs_metrics',
        'gs_evolution',
        'metrics.gs',
        'logs.gs',
        'history.gs',
        'training_logs.gs',
    ]
    
    for path in common_paths:
        keys = path.split('.')
        data = checkpoint
        try:
            for key in keys:
                data = data[key]
            print(f"✓ Found gs data at: {path}")
            return data
        except (KeyError, TypeError):
            continue
    
    return None


def main():
    """Main function with example usage."""
    
    checkpoint_path = '/path/to/checkpoints/sage_unet_2stage_glas.pth'
    
    print("=" * 70)
    print("Checkpoint Inspection Tool")
    print("=" * 70)
    print()
    
    if not Path(checkpoint_path).exists():
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        print("\nUsage examples:")
        print("  python inspect_checkpoint.py <checkpoint_path>")
        print("  python inspect_checkpoint.py --find-gs <checkpoint_path>")
        print("  python inspect_checkpoint.py --extract <path> <checkpoint_path>")
        return
    
    # Inspect structure
    print("Step 1: Inspecting checkpoint structure...")
    print("-" * 70)
    info = inspect_checkpoint(checkpoint_path)
    
    # Find gs-related data
    print("\n\nStep 2: Finding gs-related data...")
    print("-" * 70)
    gs_results = find_gs_metrics(checkpoint_path)
    
    if gs_results:
        print(f"Found {len(gs_results)} gs-related entries:")
        for path, dtype, description in gs_results:
            print(f"\n  Path: {path}")
            print(f"  Type: {dtype}")
            print(f"  Info: {description}")
    else:
        print("No gs-related data found in checkpoint")
        print("\nCommon storage patterns:")
        print("  - checkpoint['metrics']['gs']")
        print("  - checkpoint['logs']['gs']")
        print("  - checkpoint['gs_metrics']")
        print("  - checkpoint['training_history']['gs']")
    
    # Try to extract data
    print("\n\nStep 3: Attempting to extract gs data...")
    print("-" * 70)
    gs_data = extract_gs_data(checkpoint_path)
    
    if gs_data is not None:
        print("✓ Successfully extracted gs data")
        if isinstance(gs_data, dict):
            print(f"  Keys: {list(gs_data.keys())}")
        elif isinstance(gs_data, torch.Tensor):
            print(f"  Shape: {gs_data.shape}")
        elif isinstance(gs_data, (list, tuple)):
            print(f"  Length: {len(gs_data)}")
    else:
        print("✗ Could not automatically extract gs data")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == '--find-gs' and len(sys.argv) > 2:
            checkpoint = sys.argv[2]
            print("Finding gs-related data...")
            results = find_gs_metrics(checkpoint)
            for path, dtype, desc in results:
                print(f"  {path} ({dtype}): {desc}")
        elif sys.argv[1] == '--extract' and len(sys.argv) > 3:
            path_to_extract = sys.argv[2]
            checkpoint = sys.argv[3]
            print(f"Extracting from {path_to_extract}...")
            data = extract_gs_data(checkpoint, path_to_extract)
            if data is not None:
                print(f"✓ Extracted: {type(data)}")
        else:
            checkpoint = sys.argv[1]
            inspect_checkpoint(checkpoint)
    else:
        main()
