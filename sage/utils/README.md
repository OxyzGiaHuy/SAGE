# Utils Module (SAGE)


## Main function
- `metrics.py`: `SimpleDiceLoss`, `calculate_dice_coefficient`, `calculate_iou`, `calculate_pixel_accuracy`.
- `training_utils.py`: `setup_logging`, `set_seed`, `seed_worker`, `EarlyStopping`.
- `model_utils.py`: `count_parameters`, `save_checkpoint`, `load_checkpoint`, `load_shared_expert_weights`.
- `visualization.py`: `visualize_predictions`, `visualize_expert_usage`, `save_training_plots`.
- `evaluation.py`: `evaluate_model`, `calculate_metrics_batch`, `generate_evaluation_report`.
- `dataloader.py`: `ConfigurableMedicalDataset`, `get_transformations` (import trực tiếp từ file khi cần).

## Example
```python
from utils import (
    SimpleDiceLoss, calculate_iou, calculate_dice_coefficient,
    setup_logging, set_seed, seed_worker, EarlyStopping,
    visualize_predictions, save_training_plots,
    count_parameters, save_checkpoint, load_checkpoint,
    evaluate_model,
)

# loss & metrics
criterion = SimpleDiceLoss(n_classes=2)
# ... forward ...
class_iou, mean_iou = calculate_iou(preds, labels, num_classes=2)

# logging/seed
setup_logging(output_dir="logs", experiment_name="run1")
set_seed(42)

# early stopping
es = EarlyStopping(patience=10, path="best.pt", metric_name="IoU", mode="max")

# checkpoint
save_checkpoint(model, optimizer, epoch, loss, metrics, filepath="ckpt.pt")
# model, optimizer = load_checkpoint("ckpt.pt", model, optimizer)
```
