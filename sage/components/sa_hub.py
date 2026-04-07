"""
SAHub - Shape-Adapting Hub for SAGE Architecture

This module implements the Shape-Adapting Hub, a critical component for handling
heterogeneous expert architectures in the SAGE system.

Key Features:
    - Format Detection: Automatically identifies CNN vs. Transformer formats
    - Bidirectional Conversion: Supports all format transformations
    - Channel Adaptation: Learnable projections for dimension matching
    - Spatial Adaptation: Interpolation for resolution matching
    
Supported Tensor Formats:
    - CNN: (B, C, H, W) - Convolutional feature maps
    - Transformer: (B, N, D) - Sequence of embeddings

Adaptation Scenarios:
    1. CNN ↔ Transformer: Format conversion with dimension adaptation
    2. CNN ↔ CNN: Channel and spatial resolution matching
    3. Transformer ↔ Transformer: Embedding and sequence length adaptation
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F



class SAHub(nn.Module):
    """
    SAHub - Shape-Adapting Hub for Heterogeneous Expert Architectures
    
    A stateless shape converter that enables seamless integration of experts
    with different input/output formats and dimensions.
    
    Core Functionality:
        - Automatic format detection (CNN vs. Transformer)
        - Bidirectional format conversion
        - Learnable channel/embedding adaptation
        - Spatial/sequence length interpolation
    
    Design Principles:
        - Single unified `adapt()` method for all conversions
        - Stateless operation (except for cached adapters)
        - Robust error handling with detailed logging
    
    Attributes:
        adapters (nn.ModuleDict): Cached learnable adapters for efficiency
    """
    
    def __init__(self):
        """
        Initialize SAHub.
        
        Creates a stateless shape converter with an empty adapter cache.
        Adapters are created on-demand during forward passes.
        """
        super().__init__()
        
        # Lightweight adapter cache for channel/feature dimension changes
        self.adapters = nn.ModuleDict()
        
        # Logger for debugging and monitoring
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def add_adapter(self, adapter_key: str, adapter_module: nn.Module):
        """
        Manually add an adapter to the cache.
        
        Useful for pre-populating adapters during model initialization
        to avoid runtime creation overhead.
        
        Args:
            adapter_key: Unique identifier for the adapter
            adapter_module: The adapter module (Conv2d or Linear)
        """
        if adapter_key not in self.adapters:
            self.adapters[adapter_key] = adapter_module
    
    # ========================================================================
    # Private Methods - Format Detection and Inference
    # ========================================================================

    def _get_tensor_format(self, tensor: torch.Tensor) -> str:
        """
        Infer tensor format from shape.
        
        Args:
            tensor: Input tensor
            
        Returns:
            'cnn' for 4D tensors (B, C, H, W)
            'transformer' for 3D tensors (B, N, D)
        """
        if tensor.dim() == 4:
            return 'cnn'
        elif tensor.dim() == 3:
            return 'transformer'
        else:
            self.logger.warning(
                f"Unsupported tensor dimension: {tensor.dim()}. "
                "Defaulting to 'cnn'."
            )
            return 'cnn'

    def _infer_expert_type(self, expert_module: nn.Module) -> str:
        """
        Infer expert architecture type from module structure.
        
        Checks for transformer indicators (attention mechanisms),
        otherwise assumes CNN architecture.
        
        Args:
            expert_module: The expert module to analyze
            
        Returns:
            'transformer' if attention modules found, 'cnn' otherwise
        """
        # Check for transformer characteristics (attention mechanisms)
        for name, module in expert_module.named_modules():
            if 'attention' in name.lower() or 'attn' in name.lower():
                return 'transformer'
            if isinstance(module, nn.MultiheadAttention):
                return 'transformer'
        
        # Default to CNN if no transformer indicators found
        return 'cnn'
    
    def _get_expert_embed_dim(self, expert_module: nn.Module) -> Optional[int]:
        """
        Extract expected embedding dimension from a transformer expert.
        
        Tries multiple methods to find the embedding dimension:
        1. hidden_size attribute (custom TransUNet blocks)
        2. norm1.normalized_shape (timm ViT blocks)
        3. attention.embed_dim
        4. Computed from num_heads * head_dim
        
        Args:
            expert_module: Transformer expert module
            
        Returns:
            Expected embedding dimension, or None if cannot be determined
        """
        # Method 1: Check for hidden_size attribute
        if hasattr(expert_module, 'hidden_size'):
            return expert_module.hidden_size
        
        # Method 2: Check norm1.normalized_shape (timm ViT blocks)
        if hasattr(expert_module, 'norm1') and hasattr(expert_module.norm1, 'normalized_shape'):
            norm_shape = expert_module.norm1.normalized_shape
            if isinstance(norm_shape, (tuple, list)) and len(norm_shape) > 0:
                return norm_shape[0]
        
        # Method 3: Check attention module embed_dim
        if hasattr(expert_module, 'attention') and hasattr(expert_module.attention, 'embed_dim'):
            return expert_module.attention.embed_dim
        
        # Method 4: Compute from num_heads and head_dim (timm style)
        if hasattr(expert_module, 'attn'):
            attn = expert_module.attn
            if hasattr(attn, 'num_heads') and hasattr(attn, 'head_dim'):
                return attn.num_heads * attn.head_dim
        
        return None
    
    # ========================================================================
    # Private Methods - Format Conversion
    # ========================================================================
    
    # ========================================================================
    # Private Methods - Format Conversion
    # ========================================================================
    
    def _format_cnn_to_transformer(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert CNN format (B, C, H, W) to Transformer format (B, N, C).
        
        Flattens spatial dimensions and rearranges to sequence format:
        N = H * W (sequence length)
        
        Args:
            x: CNN format tensor (B, C, H, W)
            
        Returns:
            Transformer format tensor (B, N, C) where N = H*W
        """
        assert x.dim() == 4, "Input must be in CNN format (B, C, H, W)"
        B, C, H, W = x.shape
        # flatten(2) combines H, W dims; transpose swaps seq_len and channel dims
        return x.flatten(2).transpose(1, 2).contiguous()

    def _format_transformer_to_cnn(
        self,
        x: torch.Tensor,
        target_spatial_dims: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Convert Transformer format (B, N, C) to CNN format (B, C, H, W).
        
        Reshapes sequence back to 2D spatial grid. If sequence length doesn't
        match target spatial dimensions, applies interpolation.
        
        Args:
            x: Transformer format tensor (B, N, C)
            target_spatial_dims: Target spatial dimensions (H, W)
            
        Returns:
            CNN format tensor (B, C, H, W)
        """
        assert x.dim() == 3, "Input must be in Transformer format (B, N, C)"
        B, N, C = x.shape
        H, W = target_spatial_dims

        # Handle sequence length mismatch via interpolation
        if N != H * W:
            self.logger.debug(
                f"Resizing sequence length {N} to match target "
                f"spatial dims {H}x{W}"
            )
            x = x.transpose(1, 2)  # (B, C, N)
            
            # Try to reshape to square for 2D interpolation
            sqrt_N = int(N**0.5)
            if sqrt_N * sqrt_N == N:  # Can be reshaped to square
                x_reshaped = x.view(B, C, sqrt_N, sqrt_N)
                x_interpolated = F.interpolate(
                    x_reshaped,
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                )
                return x_interpolated.contiguous()
            else:  # Use 1D interpolation
                x_interpolated = F.interpolate(
                    x.unsqueeze(-1),
                    size=(H * W, 1),
                    mode='bilinear',
                    align_corners=False
                )
                return x_interpolated.squeeze(-1).view(B, C, H, W).contiguous()

        # If lengths match, just reshape
        return x.transpose(1, 2).view(B, C, H, W).contiguous()
    
    # ========================================================================
    # Private Methods - Channel/Dimension Adaptation
    # ========================================================================
    
    def _adapt_channels(
        self,
        x: torch.Tensor,
        target_channels: int
    ) -> torch.Tensor:
        """
        Adapt channel/feature dimensions using learnable projections.
        
        Uses cached 1x1 Conv2d (for CNN) or Linear (for Transformer) layers.
        Raises error if required adapter is not pre-initialized.
        
        Args:
            x: Input tensor (CNN or Transformer format)
            target_channels: Target channel/feature dimension
            
        Returns:
            Adapted tensor with target_channels dimension
            
        Raises:
            RuntimeError: If required adapter not found in cache
        """
        if x.dim() == 4:  # CNN format (B, C, H, W)
            current_channels = x.shape[1]
            if current_channels == target_channels:
                return x
            
            adapter_key = f"conv_{current_channels}_to_{target_channels}"
            if adapter_key not in self.adapters:
                self.logger.error(
                    f"FATAL: Missing required SAHub adapter: {adapter_key}. "
                    "Model was not pre-initialized correctly."
                )
                raise RuntimeError(f"Missing required SAHub adapter: {adapter_key}")
            return self.adapters[adapter_key](x)

        elif x.dim() == 3:  # Transformer format (B, N, D)
            current_features = x.shape[-1]
            if current_features == target_channels:
                return x
            
            adapter_key = f"linear_{current_features}_to_{target_channels}"
            if adapter_key not in self.adapters:
                self.logger.error(
                    f"FATAL: Missing required SAHub adapter: {adapter_key}. "
                    "Model was not pre-initialized correctly."
                )
                raise RuntimeError(f"Missing required SAHub adapter: {adapter_key}")
            return self.adapters[adapter_key](x)
        
        return x
    
    # ========================================================================
    # Public Methods - Unified Adaptation
    # ========================================================================
    
    def adapt(
        self,
        input_tensor: torch.Tensor,
        target_block: nn.Module,
        main_path_shape: Optional[torch.Size] = None
    ) -> Tuple[torch.Tensor, bool]:
        """
        Unified adaptation method for all tensor format conversions.
        
        Handles bidirectional conversions between CNN and Transformer formats,
        with automatic channel/embedding dimension adaptation and spatial/sequence
        length interpolation.
        
        Usage Modes:
            1. Input Adaptation: Prepare input for expert
               - main_path_shape=None
               - Adapts to expert's expected input format
            
            2. Output Adaptation: Match expert output to main path
               - main_path_shape=<main_output.shape>
               - Adapts expert output to match main path format
        
        Args:
            input_tensor: Tensor to adapt
            target_block: Target module (determines expected format)
            main_path_shape: If provided, adapts to match this shape (output mode)
            
        Returns:
            Tuple of (adapted_tensor, success_flag)
        """
        try:
            input_format = self._get_tensor_format(input_tensor)
            target_format = self._infer_expert_type(target_block)
            
            # Log adaptation details for debugging
            self.logger.debug(
                f"SAHub.adapt: input_shape={input_tensor.shape}, "
                f"input_format={input_format}, target_format={target_format}, "
                f"target_block={target_block.__class__.__name__}, "
                f"main_path_shape={main_path_shape}"
            )
            
            # Get expert embedding dimension if it's a transformer
            if target_format == 'transformer':
                expert_dim = self._get_expert_embed_dim(target_block)
                self.logger.debug(
                    f"SAHub: Target transformer expects embed_dim={expert_dim}"
                )
            
            if input_format != target_format:
                self.logger.debug(
                    f"SAHub: Adapting {input_format.upper()} "
                    f"({input_tensor.shape}) -> {target_format.upper()} expert."
                )

            adapted = input_tensor

            # Case 1: CNN -> Transformer conversion
            if input_format == 'cnn' and target_format == 'transformer':
                adapted = self._format_cnn_to_transformer(adapted)
                
                # Determine target dimension priority:
                # 1. main_path_shape (output adaptation)
                # 2. expert's expected dimension (input adaptation)
                # 3. current dimension (fallback)
                if main_path_shape:
                    target_dim = main_path_shape[-1]
                else:
                    expert_dim = self._get_expert_embed_dim(target_block)
                    target_dim = (
                        expert_dim if expert_dim is not None
                        else adapted.shape[-1]
                    )
                
                # Adapt embedding dimension if needed
                if adapted.shape[-1] != target_dim:
                    self.logger.debug(
                        f"SAHub: Adapting channels {adapted.shape[-1]} -> "
                        f"{target_dim} for {target_block.__class__.__name__}"
                    )
                    adapted = self._adapt_channels(adapted, target_dim)
                
                # Adapt sequence length for output adaptation
                if main_path_shape and adapted.shape[1] != main_path_shape[1]:
                    target_seq_len = main_path_shape[1]
                    # Transpose for 1D interpolation: (B, N, C) -> (B, C, N)
                    adapted = adapted.transpose(1, 2)
                    adapted = F.interpolate(
                        adapted,
                        size=target_seq_len,
                        mode='linear',
                        align_corners=False
                    )
                    # Transpose back: (B, C, N) -> (B, N, C)
                    adapted = adapted.transpose(1, 2)

            # Case 2: Transformer -> CNN conversion
            elif input_format == 'transformer' and target_format == 'cnn':
                # Determine target channels and spatial dimensions
                if main_path_shape and len(main_path_shape) == 4:
                    target_channels = main_path_shape[1]
                    target_spatial_dims = main_path_shape[2:]
                else:
                    # Find first Conv2d to infer expected input shape
                    first_conv = None
                    for module in target_block.modules():
                        if isinstance(module, nn.Conv2d):
                            first_conv = module
                            break
                    if not first_conv:
                        raise RuntimeError(
                            f"Cannot find input Conv2d in "
                            f"{target_block.__class__.__name__}"
                        )
                    
                    target_channels = first_conv.in_channels
                    B, N_in, _ = input_tensor.shape
                    H = W = int(N_in**0.5)
                    target_spatial_dims = (H, W)

                adapted = self._adapt_channels(input_tensor, target_channels)
                adapted = self._format_transformer_to_cnn(
                    adapted,
                    target_spatial_dims
                )

            # Case 3: Transformer -> Transformer (different embed_dim/seq_len)
            elif input_format == 'transformer' and target_format == 'transformer':
                # Determine target embedding dimension
                if main_path_shape:
                    target_dim = main_path_shape[-1]
                else:
                    expert_dim = self._get_expert_embed_dim(target_block)
                    target_dim = (
                        expert_dim if expert_dim is not None
                        else adapted.shape[-1]
                    )
                
                # Adapt embedding dimension if needed
                if adapted.shape[-1] != target_dim:
                    self.logger.debug(
                        f"SAHub: Transformer->Transformer channel adaptation "
                        f"{adapted.shape[-1]} -> {target_dim} for "
                        f"{target_block.__class__.__name__}"
                    )
                    adapted = self._adapt_channels(adapted, target_dim)
                
                # Adapt sequence length if needed
                if main_path_shape and adapted.shape[1] != main_path_shape[1]:
                    target_seq_len = main_path_shape[1]
                    self.logger.debug(
                        f"SAHub: Transformer->Transformer seq length adaptation "
                        f"{adapted.shape[1]} -> {target_seq_len}"
                    )
                    adapted = adapted.transpose(1, 2)
                    adapted = F.interpolate(
                        adapted,
                        size=target_seq_len,
                        mode='linear',
                        align_corners=False
                    )
                    adapted = adapted.transpose(1, 2)

            # Case 4: CNN -> CNN (channel/spatial adaptation)
            elif input_format == 'cnn' and target_format == 'cnn':
                # Determine target channels and spatial dimensions
                if main_path_shape:
                    # Output adaptation: match main path shape
                    target_channels = main_path_shape[1]
                    target_spatial_dims = main_path_shape[2:]
                else:
                    # Input adaptation: match expert's expected input
                    first_conv = next(
                        (m for m in target_block.modules()
                         if isinstance(m, nn.Conv2d)),
                        None
                    )
                    if not first_conv:
                        raise RuntimeError(
                            f"Cannot find input Conv2d in "
                            f"{target_block.__class__.__name__}"
                        )
                    
                    target_channels = first_conv.in_channels
                    # Assume expert accepts same spatial dims as input
                    target_spatial_dims = input_tensor.shape[2:]

                # Apply adaptations
                adapted = self._adapt_channels(input_tensor, target_channels)
                if adapted.shape[2:] != target_spatial_dims:
                    adapted = F.interpolate(
                        adapted,
                        size=target_spatial_dims,
                        mode='bilinear',
                        align_corners=False
                    )
            
            return adapted, True

        except Exception as e:
            self.logger.exception(
                f"Unified adaptation failed for {input_tensor.shape} -> "
                f"{target_format}: {e}"
            )
            return input_tensor, False


# ============================================================================
# Factory Functions
# ============================================================================

def create_sa_hub() -> SAHub:
    """
    Factory function to create SAHub instance.
    
    Returns:
        SAHub: A new SAHub instance with empty adapter cache
        
    Example:
        >>> sa_hub = create_sa_hub()
        >>> # Pre-populate adapters during model initialization
        >>> sa_hub.add_adapter("conv_256_to_512", nn.Conv2d(256, 512, 1))
    """
    return SAHub()
