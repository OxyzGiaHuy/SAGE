import logging
import torch.nn as nn

logger = logging.getLogger(__name__)


def extract_convnext_blocks(model):
    """
    Extract all residual blocks from ConvNeXt model as individual experts.
    """
    blocks = []

    if hasattr(model, "stages"):
        for stage_idx, stage in enumerate(model.stages):
            for block_idx, block in enumerate(stage):
                blocks.append(block)
                logger.info("  Extracted: Stage %s, Block %s", stage_idx, block_idx)

    return blocks


class TupleSafeWrapper(nn.Module):
    """
    Wrapper to ensure module returns a Tensor (not tuple).
    Some timm blocks return (tensor, other) -> we keep tensor.
    Also delegates attribute access for SAHub compatibility.
    """

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, *args, **kwargs):
        out = self.module(x, *args, **kwargs)
        if isinstance(out, (tuple, list)):
            print(
                f"TupleSafeWrapper: {self.module.__class__.__name__} returned tuple of {len(out)} elements, taking first"
            )
            return out[0]
        return out

    def __getattr__(self, name):
        # Delegate attribute access to wrapped module for SAHub
        protected = {
            "module",
            "_modules",
            "_parameters",
            "_buffers",
            "_backward_hooks",
            "_forward_hooks",
            "_forward_pre_hooks",
            "_state_dict_hooks",
            "_load_state_dict_pre_hooks",
            "_modules_containers",
        }
        if name in protected:
            return super().__getattr__(name)
        return getattr(self.module, name)
