"""
ConvNeXt-Transformer UNet for Semantic Segmentation

This module implements a hybrid UNet architecture that combines:
1. ConvNeXt backbone for hierarchical feature extraction
2. Vision Transformer blocks for global context modeling at the bottleneck
3. UNet-style decoder with skip connections

Architecture Flow:
    Input Image (H, W, 3)
        ↓
    ConvNeXt Stages (4 stages with progressive downsampling)
        ↓ (save feature maps for skip connections)
    Tokenization & Projection
        ↓
    Transformer Blocks (global context modeling)
        ↓
    Reshape to Spatial Features
        ↓
    UNet Decoder (upsampling + skip connections)
        ↓
    Segmentation Output (H, W, num_classes)

Author: AI Research Team
Date: October 2025
"""

import logging
import os
import sys
from typing import List

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

# Add parent directory to path for imports

from sage.components.sage_layer import SageLayer
from sage.components.router import SageRouter

from .decoder_block import DecoderBlock
from .sage_injection import inject_sage_layers, pre_populate_sa_hubs

logger = logging.getLogger(__name__)


class ConvNeXtTransformerUNet(nn.Module):
    """
    Hybrid UNet architecture combining ConvNeXt encoder with Transformer bottleneck.
    
    This model uses pretrained ConvNeXt for hierarchical feature extraction and
    pretrained Vision Transformer blocks for capturing global dependencies at the
    bottleneck. The decoder follows a standard UNet design with skip connections.
    
    Args:
        num_classes (int): Number of segmentation classes
        img_size (int): Input image size (assumes square images)
        convnext_model_name (str): Name of the ConvNeXt model from timm
            Options: 'convnext_base', 'convnext_large', 'convnext_xlarge'
        vit_model_name (str): Name of the ViT model from timm for Transformer blocks
            Options: 'vit_base_patch32_224', 'vit_large_patch32_224'
        num_transformer_layers (int): Number of Transformer layers to use at bottleneck
        freeze_encoder (bool): Whether to freeze ConvNeXt weights during training
        freeze_transformer (bool): Whether to freeze Transformer weights during training
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        img_size: int = 224,
        convnext_model_name: str = 'convnext_base',
        vit_model_name: str = 'vit_base_patch32_224',
        num_transformer_layers: int = 12,
        freeze_encoder: bool = False,
        freeze_transformer: bool = False,
        sage_config: dict = None,
        gradient_checkpointing: bool = False,
    ):
        super(ConvNeXtTransformerUNet, self).__init__()
        
        self.num_classes = num_classes
        self.img_size = img_size
        self.num_transformer_layers = num_transformer_layers
        # SAGE configuration dictionary (optional)
        self.sage_config = sage_config  # Keep None if provided as None, don't convert to {}
        # Gradient checkpointing: recompute activations in backward to save memory at cost of ~30% speed
        self.use_gradient_checkpointing = gradient_checkpointing
        
        # ========================================================================
        # 1. ENCODER: ConvNeXt Backbone
        # ========================================================================
        logger.info(f"Loading pretrained ConvNeXt model: {convnext_model_name}")
        
        # Load ConvNeXt model normally (not features_only) to access all stages
        self.convnext = timm.create_model(
            convnext_model_name,
            pretrained=True,
            features_only=False  # Load full model to access stages for expert extraction
        )
        
        # Get feature dimensions for each stage by running dummy input
        dummy_input = torch.randn(1, 3, img_size, img_size)
        
        # Extract stage features manually for dimension information
        self.encoder_channels = []
        self.encoder_spatial_sizes = []
        
        if hasattr(self.convnext, 'stages'):
            with torch.no_grad():
                x = self.convnext.stem(dummy_input)
                # Save stem output channels for CNN router input dims
                self.stem_channels = x.shape[1]
                for stage in self.convnext.stages:
                    x = stage(x)
                    self.encoder_channels.append(x.shape[1])
                    self.encoder_spatial_sizes.append(x.shape[2])
        
        logger.info(f"ConvNeXt encoder channels: {self.encoder_channels}")
        logger.info(f"ConvNeXt spatial sizes: {self.encoder_spatial_sizes}")
        
        # Optionally freeze ConvNeXt encoder
        if freeze_encoder:
            for param in self.convnext.parameters():
                param.requires_grad = False
            logger.info("ConvNeXt encoder frozen")
        
        # ========================================================================
        # 2. BOTTLENECK: Transformer Blocks
        # ========================================================================
        logger.info(f"Loading Transformer blocks from: {vit_model_name}")
        
        # Load full ViT model to extract Transformer blocks
        vit_full = timm.create_model(vit_model_name, pretrained=True)
        
        # Extract the Transformer configuration
        self.transformer_dim = vit_full.embed_dim
        self.num_heads = vit_full.num_heads if hasattr(vit_full, 'num_heads') else 12
        
        # Extract only the Transformer encoder blocks
        # Most ViT models store blocks in vit_full.blocks
        if hasattr(vit_full, 'blocks'):
            all_blocks = vit_full.blocks
        else:
            raise AttributeError(f"Cannot find transformer blocks in {vit_model_name}")
        
        # Use the first N transformer layers
        self.transformer_blocks = nn.ModuleList([
            all_blocks[i] for i in range(min(num_transformer_layers, len(all_blocks)))
        ])
        
        logger.info(f"Extracted {len(self.transformer_blocks)} Transformer blocks")
        logger.info(f"Transformer dimension: {self.transformer_dim}")
        
        # Optionally freeze Transformer blocks
        if freeze_transformer:
            for block in self.transformer_blocks:
                for param in block.parameters():
                    param.requires_grad = False
            logger.info("Transformer blocks frozen")

        # Initialize the expert_pool attribute before injection
        self.expert_pool = None 

        # Inject SAGE wrappers ONLY if sage_config is provided and not None
        if self.sage_config is not None:
            try:
                self.expert_pool = inject_sage_layers(
                    convnext=self.convnext,
                    transformer_blocks=self.transformer_blocks,
                    encoder_channels=self.encoder_channels,
                    stem_channels=self.stem_channels,
                    transformer_dim=self.transformer_dim,
                    sage_config=self.sage_config,
                )
                logger.info("SAGE wrappers injected into ConvNeXt-Transformer UNet")

                # After injecting, pre-populate all SAHub adapters
                pre_populate_sa_hubs(
                    model=self,
                    encoder_channels=self.encoder_channels,
                    stem_channels=self.stem_channels,
                    transformer_dim=self.transformer_dim,
                )
            except Exception as e:
                logger.warning(f"SAGE injection skipped or failed: {e}")
        else:
            logger.info("SAGE disabled (sage_config=None). Using pure baseline model.")
        
        # ========================================================================
        # 3. PROJECTION: ConvNeXt -> Transformer
        # ========================================================================
        # The last ConvNeXt stage output needs to be projected to Transformer dim
        self.bottleneck_channels = self.encoder_channels[-1]  # Last stage channels
        self.bottleneck_spatial_size = self.encoder_spatial_sizes[-1]  # e.g., 7x7
        
        # Linear projection from ConvNeXt channels to Transformer dimension
        # Shape: (B, C_convnext, H, W) -> (B, H*W, D_transformer)
        self.convnext_to_transformer = nn.Linear(
            self.bottleneck_channels,
            self.transformer_dim
        )
        
        # Layer normalization before Transformer (standard practice)
        self.pre_transformer_norm = nn.LayerNorm(self.transformer_dim)
        
        # Positional embeddings for the spatial tokens
        num_patches = self.bottleneck_spatial_size ** 2
        self.positional_embeddings = nn.Parameter(
            torch.zeros(1, num_patches, self.transformer_dim)
        )
        nn.init.trunc_normal_(self.positional_embeddings, std=0.02)
        
        logger.info(f"Bottleneck: {self.bottleneck_channels} channels, "
                   f"{self.bottleneck_spatial_size}x{self.bottleneck_spatial_size} spatial")
        logger.info(f"Projection: {self.bottleneck_channels} -> {self.transformer_dim}")
        
        # ========================================================================
        # 4. REPROJECTION: Transformer -> Decoder
        # ========================================================================
        # After Transformer, we need to reshape back to spatial format
        # Shape: (B, H*W, D_transformer) -> (B, C_decoder, H, W)
        
        # We'll use the same channel dimension as the last ConvNeXt stage
        # for easier integration with the decoder
        self.transformer_to_decoder = nn.Linear(
            self.transformer_dim,
            self.bottleneck_channels
        )
        
        self.post_transformer_norm = nn.LayerNorm(self.bottleneck_channels)
        
        # ========================================================================
        # 5. DECODER: UNet-style Upsampling with Skip Connections
        # ========================================================================
        # Build decoder blocks that upsample and fuse with skip connections
        # Decoder path: stage3 -> stage2 -> stage1 -> stage0 -> output
        
        self.decoder_blocks = nn.ModuleList()
        
        # Reverse the encoder channels for decoder (bottom-up)
        decoder_in_channels = list(reversed(self.encoder_channels))
        
        # Decoder block for each upsampling stage
        for i in range(len(decoder_in_channels) - 1):
            in_ch = decoder_in_channels[i]
            skip_ch = decoder_in_channels[i + 1]
            out_ch = decoder_in_channels[i + 1]
            
            decoder_block = DecoderBlock(
                in_channels=in_ch,
                skip_channels=skip_ch,
                out_channels=out_ch
            )
            self.decoder_blocks.append(decoder_block)
        
        # ========================================================================
        # 6. SEGMENTATION HEAD
        # ========================================================================
        # Final layers to produce segmentation map
        final_channels = decoder_in_channels[-1]  # Channels after last decoder block
        
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(final_channels, final_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(final_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(final_channels // 2, num_classes, kernel_size=1)
        )
        
        logger.info(f"Decoder initialized with {len(self.decoder_blocks)} upsampling blocks")
        logger.info(f"Segmentation head: {final_channels} -> {num_classes} classes")

    def set_shared_experts(self, shared_expert_indices: List[int]):
        """
        Update all SAGE routers in the model with a new set of shared expert indices.
        """
        logger.info(f"Setting shared experts: {shared_expert_indices}")
        for module in self.modules():
            if isinstance(module, SageRouter):
                module.shared_expert_indices = shared_expert_indices
                # Some router implementations keep a mask; update if available
                if hasattr(module, '_update_shared_mask'):
                    module._update_shared_mask()

    def get_expert_usage_statistics(self):
        """
        Collect expert usage statistics from SAGE-wrapped modules (if they expose `get_stats`).
        Returns a dict grouped by module type.
        """
        stats = {'cnn_stages': [], 'transformer_blocks': []}
        
        # Collect stats from ConvNeXt stages (wrapped as SageLayer)
        if hasattr(self.convnext, 'stages'):
            for stage in self.convnext.stages:
                if isinstance(stage, SageLayer) and hasattr(stage, 'get_stats'):
                    stats['cnn_stages'].append(stage.get_stats())
        
        # Collect stats from Transformer blocks (wrapped as SageLayer)
        for block in self.transformer_blocks:
            if isinstance(block, SageLayer) and hasattr(block, 'get_stats'):
                stats['transformer_blocks'].append(block.get_stats())

        return stats

    def accumulate_routing_info(self, routing_infos):
        """
        Accumulate routing information from training batches.
        This updates the internal stats of SAGE layers.
        """
        if not routing_infos:
            return

        for routing_info in routing_infos:
            if isinstance(routing_info, dict) and 'routing_info' in routing_info:
                layer_routing = routing_info['routing_info']
                # Find corresponding SAGE layer and update its stats
                layer_idx = 0
                for name, module in self.named_modules():
                    if isinstance(module, SageLayer):
                        if layer_idx < len(layer_routing):
                            # Update the router stats in the SAGE layer
                            if hasattr(module.router, 'update_stats'):
                                module.router.update_stats(layer_routing[layer_idx])
                        layer_idx += 1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ConvNeXt-Transformer UNet.
        
        Args:
            x (torch.Tensor): Input images of shape (B, 3, H, W)
            
        Returns:
            torch.Tensor: Segmentation logits of shape (B, num_classes, H, W)
        """
        # ====================================================================
        # STEP 1: ConvNeXt Encoder - Extract hierarchical features
        # ====================================================================
        # Get feature maps from all stages
        # encoder_features[i] has shape (B, C_i, H_i, W_i)
        
        # Since ConvNeXt is now loaded normally (not features_only),
        # we need to extract features manually from each stage
        encoder_features = []
        x_feat = self.convnext.stem(x)
        
        # Access wrapped stages through self.convnext.stages (where they were wrapped IN-PLACE)
        for stage_idx, stage in enumerate(self.convnext.stages):
            if self.expert_pool is not None:
                # SAGE enabled - pass expert_pool
                out = stage(x_feat, self.expert_pool)
                if isinstance(out, tuple):
                    # SAGE layer returns (output, routing_info)
                    x_feat = out[0]
                else:
                    x_feat = out
            else:
                # Baseline mode - no expert_pool
                x_feat = stage(x_feat)
            encoder_features.append(x_feat)
        
        # Store skip connections (we'll use them in reverse order in decoder)
        skip_connections = encoder_features[:-1]  # All except the last stage
        
        # Get the bottleneck feature map (output of the last ConvNeXt stage)
        bottleneck_features = encoder_features[-1]
        # Shape: (B, bottleneck_channels, H_bn, W_bn)
        # e.g., (B, 1024, 7, 7) for ConvNeXt-Base with 224x224 input
        
        B, _, H, W = bottleneck_features.shape
        
        # ====================================================================
        # STEP 2: Tokenization - Convert spatial features to sequence
        # ====================================================================
        # Reshape from (B, C, H, W) to (B, H*W, C)
        tokens = bottleneck_features.flatten(2).transpose(1, 2)
        # Shape: (B, num_patches, bottleneck_channels)
        # where num_patches = H * W
        
        # ====================================================================
        # STEP 3: Projection - Map ConvNeXt features to Transformer space
        # ====================================================================
        # Project from ConvNeXt dimension to Transformer dimension
        tokens = self.convnext_to_transformer(tokens)
        # Shape: (B, num_patches, transformer_dim)
        
        # Add positional embeddings (dynamic sizing)
        B, num_patches, dim = tokens.shape
        if num_patches > self.positional_embeddings.shape[1]:
            # Need larger positional embeddings - interpolate or extend
            pos_emb = torch.nn.functional.interpolate(
                self.positional_embeddings.transpose(1, 2), 
                size=num_patches, 
                mode='linear'
            ).transpose(1, 2)
        elif num_patches < self.positional_embeddings.shape[1]:
            # Crop to match
            pos_emb = self.positional_embeddings[:, :num_patches, :]
        else:
            pos_emb = self.positional_embeddings
        tokens = tokens + pos_emb
        
        # Normalize before Transformer
        tokens = self.pre_transformer_norm(tokens)
        
        # ====================================================================
        # STEP 4: Transformer Blocks - Global context modeling
        # ====================================================================
        # Pass through each Transformer block (now wrapped with SAGE)
        # Access wrapped blocks through self.transformer_blocks (where they were wrapped IN-PLACE)
        
        for transformer_block in self.transformer_blocks:
            if self.expert_pool is not None:
                # SAGE enabled — capture pool in closure so grad_checkpoint only sees tensors
                if self.use_gradient_checkpointing and self.training:
                    pool = self.expert_pool
                    def _ckpt_fn(x, blk=transformer_block, ep=pool):
                        out = blk(x, ep)
                        return out[0] if isinstance(out, tuple) else out
                    tokens = grad_checkpoint(_ckpt_fn, tokens, use_reentrant=False)
                else:
                    output = transformer_block(tokens, self.expert_pool)
                    tokens = output[0] if isinstance(output, tuple) else output
            else:
                # Baseline mode — no expert_pool
                if self.use_gradient_checkpointing and self.training:
                    tokens = grad_checkpoint(transformer_block, tokens, use_reentrant=False)
                else:
                    tokens = transformer_block(tokens)
        # Shape: (B, num_patches, transformer_dim)
        
        # ====================================================================
        # STEP 5: Reprojection - Map back to decoder dimension
        # ====================================================================
        # Project from Transformer dimension back to bottleneck channels
        tokens = self.transformer_to_decoder(tokens)
        # Shape: (B, num_patches, bottleneck_channels)
        
        tokens = self.post_transformer_norm(tokens)
        
        # ====================================================================
        # STEP 6: Spatial Reshape - Convert sequence back to spatial format
        # ====================================================================
        # Reshape from (B, H*W, C) to (B, C, H, W)
        decoder_input = tokens.transpose(1, 2).reshape(B, self.bottleneck_channels, H, W)
        # Shape: (B, bottleneck_channels, H_bn, W_bn)
        
        # ====================================================================
        # STEP 7: UNet Decoder - Upsampling with skip connections
        # ====================================================================
        x_dec = decoder_input
        
        # Process through decoder blocks (bottom-up)
        # Reverse skip connections to match decoder order (deep to shallow)
        reversed_skips = list(reversed(skip_connections))
        
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip = reversed_skips[i]
            x_dec = decoder_block(x_dec, skip)
            # x_dec shape increases: (B, C, H, W) -> (B, C/2, 2H, 2W)
        
        # ====================================================================
        # STEP 8: Segmentation Head - Final prediction
        # ====================================================================
        logits = self.segmentation_head(x_dec)
        # Shape: (B, num_classes, H, W)
        
        # Upsample to original input size if needed
        if logits.shape[2:] != (self.img_size, self.img_size):
            logits = F.interpolate(
                logits,
                size=(self.img_size, self.img_size),
                mode='bilinear',
                align_corners=False
            )
        
        return logits
    
    def forward_with_routing_info(self, x: torch.Tensor) -> dict:
        """
        Forward pass that captures routing information from SAGE layers.
        
        Returns:
            dict with keys:
                - 'logits': Segmentation predictions (B, num_classes, H, W)
                - 'routing_infos': List of routing info from each SAGE layer
        """
        # ====================================================================
        # STEP 1: ConvNeXt Encoder - Extract hierarchical features
        # ====================================================================
        encoder_features = []
        x_feat = self.convnext.stem(x)
        cnn_routing_infos = []
        
        # Access wrapped stages through self.convnext.stages (where they were wrapped IN-PLACE)
        for stage_idx, stage in enumerate(self.convnext.stages):
            if self.expert_pool is not None:
                # SAGE enabled - pass expert_pool
                out = stage(x_feat, self.expert_pool)
                if isinstance(out, tuple):
                    # SAGE layer returns (output, routing_info)
                    x_feat, routing_info = out
                    cnn_routing_infos.append(routing_info)
                else:
                    x_feat = out
                    cnn_routing_infos.append(None)
            else:
                # Baseline mode - no expert_pool
                x_feat = stage(x_feat)
                cnn_routing_infos.append(None)
            encoder_features.append(x_feat)
        
        skip_connections = encoder_features[:-1]
        bottleneck_features = encoder_features[-1]
        
        B, _, H, W = bottleneck_features.shape
        
        # ====================================================================
        # STEP 2: Tokenization
        # ====================================================================
        tokens = bottleneck_features.flatten(2).transpose(1, 2)
        
        # ====================================================================
        # STEP 3: Projection to Transformer space
        # ====================================================================
        tokens = self.convnext_to_transformer(tokens)
        
        # Add positional embeddings
        B, num_patches, dim = tokens.shape
        if num_patches > self.positional_embeddings.shape[1]:
            pos_emb = torch.nn.functional.interpolate(
                self.positional_embeddings.transpose(1, 2), 
                size=num_patches, 
                mode='linear'
            ).transpose(1, 2)
        elif num_patches < self.positional_embeddings.shape[1]:
            pos_emb = self.positional_embeddings[:, :num_patches, :]
        else:
            pos_emb = self.positional_embeddings
        tokens = tokens + pos_emb
        tokens = self.pre_transformer_norm(tokens)
        
        # ====================================================================
        # STEP 4: Transformer Blocks with routing info capture
        # ====================================================================
        routing_infos = []
        # Access wrapped blocks through self.transformer_blocks (where they were wrapped IN-PLACE)
        for transformer_block in self.transformer_blocks:
            if self.expert_pool is not None:
                # SAGE enabled - pass expert_pool
                output = transformer_block(tokens, self.expert_pool)
                # Capture routing info if available
                if isinstance(output, tuple):
                    tokens, routing_info = output
                    routing_infos.append(routing_info)
                else:
                    tokens = output
                    routing_infos.append(None)
            else:
                # Baseline mode - no expert_pool
                tokens = transformer_block(tokens)
                routing_infos.append(None)
        
        # ====================================================================
        # STEP 5: Reprojection
        # ====================================================================
        tokens = self.transformer_to_decoder(tokens)
        tokens = self.post_transformer_norm(tokens)
        
        # ====================================================================
        # STEP 6: Spatial Reshape
        # ====================================================================
        decoder_input = tokens.transpose(1, 2).reshape(B, self.bottleneck_channels, H, W)
        
        # ====================================================================
        # STEP 7: UNet Decoder
        # ====================================================================
        x_dec = decoder_input
        reversed_skips = list(reversed(skip_connections))
        
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip = reversed_skips[i]
            x_dec = decoder_block(x_dec, skip)
        
        # ====================================================================
        # STEP 8: Segmentation Head
        # ====================================================================
        logits = self.segmentation_head(x_dec)
        
        if logits.shape[2:] != (self.img_size, self.img_size):
            logits = F.interpolate(
                logits,
                size=(self.img_size, self.img_size),
                mode='bilinear',
                align_corners=False
            )
        
        return {
            'logits': logits,
            'routing_infos': {
                'cnn': cnn_routing_infos,
                'transformer': routing_infos
            }
        }
    
    def get_model_info(self) -> dict:
        """
        Get information about the model architecture.
        
        Returns:
            dict: Model configuration and statistics
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'num_classes': self.num_classes,
            'img_size': self.img_size,
            'encoder_channels': self.encoder_channels,
            'transformer_dim': self.transformer_dim,
            'num_transformer_layers': len(self.transformer_blocks),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params
        }


def create_convnext_transformer_unet(
    num_classes: int = 3,
    img_size: int = 224,
    convnext_variant: str = 'base',
    vit_variant: str = 'base',
    num_transformer_layers: int = 12,
    freeze_encoder: bool = False,
    freeze_transformer: bool = False,
    sage_config: dict | None = None,
    gradient_checkpointing: bool = False,
) -> ConvNeXtTransformerUNet:
    """
    Factory function to create ConvNeXt-Transformer UNet model.
    
    Args:
        num_classes (int): Number of segmentation classes
        img_size (int): Input image size
        convnext_variant (str): ConvNeXt variant ('base', 'large', 'xlarge')
        vit_variant (str): ViT variant for Transformer ('base', 'large')
        num_transformer_layers (int): Number of Transformer layers at bottleneck
        freeze_encoder (bool): Freeze ConvNeXt encoder
        freeze_transformer (bool): Freeze Transformer blocks
        sage_config (dict | None): SAGE configuration dictionary
        
    Returns:
        ConvNeXtTransformerUNet: Initialized model
    """
    # Map variants to timm model names while allowing direct model names
    convnext_models = {
        'tiny': 'convnext_tiny',
        'small': 'convnext_small',
        'base': 'convnext_base',
        'large': 'convnext_large',
        'xlarge': 'convnext_xlarge',
        'convnext_tiny': 'convnext_tiny',
        'convnext_small': 'convnext_small',
        'convnext_base': 'convnext_base',
        'convnext_large': 'convnext_large',
        'convnext_xlarge': 'convnext_xlarge',
    }
    
    vit_models = {
        'base': 'vit_base_patch32_224',
        'large': 'vit_large_patch32_224',
        'vit_base_patch32_224': 'vit_base_patch32_224',
        'vit_large_patch32_224': 'vit_large_patch32_224',
    }
    
    convnext_model_name = convnext_models.get(convnext_variant, convnext_variant)
    vit_model_name = vit_models.get(vit_variant, vit_variant)
    
    model = ConvNeXtTransformerUNet(
        num_classes=num_classes,
        img_size=img_size,
        convnext_model_name=convnext_model_name,
        vit_model_name=vit_model_name,
        num_transformer_layers=num_transformer_layers,
        freeze_encoder=freeze_encoder,
        freeze_transformer=freeze_transformer,
        sage_config=sage_config,
        gradient_checkpointing=gradient_checkpointing,
    )
    
    return model


if __name__ == "__main__":
    """
    Test script to verify model creation and forward pass.
    """
    print("=" * 80)
    print("ConvNeXt-Transformer UNet Test")
    print("=" * 80)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Test configuration
    batch_size = 2
    num_classes = 3
    img_size = 224
    
    print("\n1. Creating model...")
    model = create_convnext_transformer_unet(
        num_classes=num_classes,
        img_size=img_size,
        convnext_variant='base',
        vit_variant='base',
        num_transformer_layers=12,
        freeze_encoder=False,
        freeze_transformer=False
    )
    
    print("\n2. Model Information:")
    info = model.get_model_info()
    for key, value in info.items():
        if 'parameters' in key:
            print(f"  {key}: {value:,}")
        else:
            print(f"  {key}: {value}")
    
    print("\n3. Testing forward pass...")
    # Create dummy input
    x = torch.randn(batch_size, 3, img_size, img_size)
    print(f"  Input shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"  Output shape: {output.shape}")
    print(f"  Expected shape: ({batch_size}, {num_classes}, {img_size}, {img_size})")
    
    # Verify output shape
    assert output.shape == (batch_size, num_classes, img_size, img_size), \
        f"Output shape mismatch! Got {output.shape}"
    
    print("\n4. Testing with different input size...")
    x_large = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        output_large = model(x_large)
    print(f"  Input shape: {x_large.shape}")
    print(f"  Output shape: {output_large.shape}")
    
    print("\n5. Testing gradient flow...")
    x.requires_grad = True
    output = model(x)
    loss = output.sum()
    loss.backward()
    print("  Gradient computed successfully!")
    print(f"  Input gradient shape: {x.grad.shape if x.grad is not None else 'None'}")
    
    print("\n" + "=" * 80)
    print("✓ All tests passed successfully!")
    print("=" * 80)
