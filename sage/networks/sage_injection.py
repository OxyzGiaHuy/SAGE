import logging
import torch.nn as nn

from sage.components.sage_layer import SageLayer, create_sage_layer
from sage.components.router import create_sage_router
from sage.components.sa_hub import SAHub

from .wrappers import TupleSafeWrapper

logger = logging.getLogger(__name__)


def inject_sage_layers(
    convnext,
    transformer_blocks,
    encoder_channels,
    stem_channels,
    transformer_dim,
    sage_config,
):
    """
    Inject SAGE wrappers into ConvNeXt stages and Transformer blocks in-place.
    Returns the assembled expert pool.
    """
    logger.info("=" * 80)
    logger.info("BUILDING HETEROGENEOUS EXPERT POOL (CNN Stages + Transformer Blocks)")
    logger.info("=" * 80)

    total_experts_to_wrap = len(convnext.stages) + len(transformer_blocks)
    expert_infos = []

    for i in range(len(convnext.stages)):
        expert_infos.append({"type": "cnn", "name": f"convnext_stage_{i}", "index": i})
    for i in range(len(transformer_blocks)):
        expert_infos.append(
            {
                "type": "transformer",
                "name": f"transformer_block_{i}",
                "index": len(convnext.stages) + i,
            }
        )

    sa_hub = SAHub()
    sage_layer_config = dict(sage_config) if sage_config else {}
    router_config = sage_layer_config

    logger.info("\nStep 1a: Wrapping ConvNeXt stages IN-PLACE...")
    stage_in_channels = [stem_channels] + encoder_channels[:-1]

    for stage_idx in range(len(convnext.stages)):
        original_stage = convnext.stages[stage_idx]

        router = create_sage_router(
            in_channels=stage_in_channels[stage_idx],
            expert_pool_size=total_experts_to_wrap,
            expert_infos=expert_infos,
            config=router_config,
        )

        sage_wrapper = create_sage_layer(
            main_block=TupleSafeWrapper(original_stage),
            router=router,
            sa_hub=sa_hub,
            config=sage_layer_config,
        )

        convnext.stages[stage_idx] = sage_wrapper

    logger.info("\nStep 1b: Wrapping Transformer blocks IN-PLACE...")
    for layer_idx in range(len(transformer_blocks)):
        original_block = transformer_blocks[layer_idx]

        router = create_sage_router(
            in_channels=transformer_dim,
            expert_pool_size=total_experts_to_wrap,
            expert_infos=expert_infos,
            config=router_config,
        )

        sage_wrapper = create_sage_layer(
            main_block=TupleSafeWrapper(original_block),
            router=router,
            sa_hub=sa_hub,
            config=sage_layer_config,
        )

        transformer_blocks[layer_idx] = sage_wrapper

    logger.info("\nStep 2: Building expert pool from wrapped .main_block attributes...")
    expert_pool = nn.ModuleList()

    for stage_wrapper in convnext.stages:
        expert_pool.append(stage_wrapper.main_block)
    logger.info("  - Added %s CNN main_blocks to pool.", len(convnext.stages))

    for block_wrapper in transformer_blocks:
        expert_pool.append(block_wrapper.main_block)
    logger.info("  - Added %s Transformer main_blocks to pool.", len(transformer_blocks))

    logger.info(
        "\n\u2713 Built expert pool with %s experts, all referencing active .main_block modules.",
        len(expert_pool),
    )
    logger.info("=" * 80)
    return expert_pool


def pre_populate_sa_hubs(model, encoder_channels, stem_channels, transformer_dim):
    """
    Pre-calculates and creates all possible channel/dimension adapters that the SAHubs might need.
    """
    logger.info("\n" + "=" * 80)
    logger.info("PRE-POPULATING SAHUB ADAPTERS")
    logger.info("=" * 80)

    cnn_dims = set(encoder_channels)
    if stem_channels is not None:
        cnn_dims.add(stem_channels)

    transformer_dims = {transformer_dim}
    all_dims = cnn_dims.union(transformer_dims)

    master_adapters = nn.ModuleDict()
    adapters_created_count = 0
    for in_dim in all_dims:
        for out_dim in all_dims:
            if in_dim == out_dim:
                continue

            conv_key = f"conv_{in_dim}_to_{out_dim}"
            if conv_key not in master_adapters:
                master_adapters[conv_key] = nn.Conv2d(in_dim, out_dim, kernel_size=1)
                adapters_created_count += 1

            linear_key = f"linear_{in_dim}_to_{out_dim}"
            if linear_key not in master_adapters:
                master_adapters[linear_key] = nn.Linear(in_dim, out_dim)
                adapters_created_count += 1

    logger.info("   - Created a master set of %s potential adapter layers.", adapters_created_count)

    hubs_populated = 0
    for module in model.modules():
        if isinstance(module, SageLayer):
            for key, adapter in master_adapters.items():
                module.sa_hub.add_adapter(key, adapter)
            hubs_populated += 1

    logger.info("   - Populated %s SAHub instances with all adapter layers.", hubs_populated)
    logger.info("=" * 80)
    return hubs_populated
