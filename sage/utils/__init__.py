"""
Utilities for SAGE
Modular utility functions for training, evaluation, and visualization
"""

from .metrics import (
    SimpleDiceLoss,
    calculate_dice_coefficient,
    calculate_iou,
    calculate_pixel_accuracy,
)

from .training_utils import (
    setup_logging,
    set_seed,
    seed_worker,
    EarlyStopping,
)

from .visualization import (
    visualize_predictions,
    visualize_expert_usage,
    save_training_plots,
)

from .model_utils import (
    count_parameters,
    load_checkpoint,
    save_checkpoint,
    load_shared_expert_weights,
)

from .evaluation import (
    evaluate_model,
    calculate_metrics_batch,
    generate_evaluation_report,
)

__all__ = [
    # Metrics
    "SimpleDiceLoss",
    "calculate_dice_coefficient",
    "calculate_iou",
    "calculate_pixel_accuracy",
    
    # Training utilities
    "setup_logging",
    "set_seed",
    "seed_worker",
    "EarlyStopping",
    
    # Visualization
    "visualize_predictions",
    "visualize_expert_usage",
    "save_training_plots",
    
    # Model utilities
    "count_parameters",
    "load_checkpoint",
    "save_checkpoint",
    "load_shared_expert_weights",
    
    # Evaluation
    "evaluate_model",
    "calculate_metrics_batch",
    "generate_evaluation_report",
]
