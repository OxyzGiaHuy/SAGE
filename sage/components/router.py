"""
SAGE Router - Hierarchical Expert Selection via SAGE Method

This module implements the SAGE (Shared and Gated Experts) routing mechanism
for the SAGE architecture, enabling dynamic expert selection based on input
characteristics.

Key Components:
    - Semantic Affinity Routing (SAR): Query-key matching for expert selection
    - Shared Expert Gating: Hierarchical control over shared vs. fine-grained experts
    - Load Balancing: Ensures balanced expert utilization during training

References:
    SAGE paper equations:
    - Eq. 6: Shared expert gate (g_s)
    - Eq. 7: Fine-grained routing via SAR
    - Eq. 8: Logit modulation
    - Eq. 9: Unified gating
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SageRouter(nn.Module):
    """
    SAGE Router - Hierarchical Expert Selection via SAGE Method
    
    Implements the SAGE (Shared and Gated Experts) routing mechanism for
    dynamic expert selection in Mixture of Experts architectures.
    
    Architecture:
        1. Feature Aggregation: Reduces spatial dimensions via pooling
        2. Shared Expert Gate (g_s): Controls shared vs. fine-grained expert activation
        3. Semantic Affinity Routing (SAR): Query-key matching for expert selection
        4. Logit Modulation: Hierarchical gating based on g_s
        5. Top-K Selection: Selects most relevant experts per sample
    
    Key Features:
        - Unified top-k selection from entire expert pool
        - Load balancing auxiliary loss for fair expert utilization
        - Adaptive feature projection for input flexibility
        - Exploration noise during training
    
    Attributes:
        in_channels (int): Expected input channel dimension
        expert_pool_size (int): Total number of available experts
        top_k (int): Number of experts to activate per sample
        gating_type (str): Type of gating function ('softmax' or 'sigmoid')
    """
    
    def __init__(
        self,
        in_channels: int,
        expert_pool_size: int,
        expert_infos: List[Dict[str, Any]],
        top_k: int = 1,
        shared_expert_indices: Optional[List[int]] = None,
        gating_type: str = 'softmax',
        router_hidden_dim: int = 256,
        load_balance_factor: float = 0.01,
        logit_modulation: bool = True
    ):
        """
        Initialize SAGE Hierarchical Router.
        
        Args:
            in_channels: Input channel dimension
            expert_pool_size: Total number of experts in the pool
            expert_infos: Metadata for each expert (list of dicts)
            top_k: Number of experts to activate per sample (default: 1)
            shared_expert_indices: Indices of always-active shared experts
            gating_type: Gating function type ('softmax' or 'sigmoid')
            router_hidden_dim: Hidden dimension for router MLP (default: 256)
            load_balance_factor: Weight for load balancing loss (default: 0.01)
            
        Raises:
            ValueError: If gating_type is not 'softmax' or 'sigmoid'
        """
        super().__init__()

        # Configuration
        self.in_channels = in_channels
        self.expert_pool_size = expert_pool_size
        self.expert_infos = expert_infos
        self.top_k = top_k
        self.router_hidden_dim = router_hidden_dim
        self.load_balance_factor = load_balance_factor
        self.shared_expert_indices = shared_expert_indices or []
        self.logit_modulation = bool(logit_modulation)
        
        # Validate gating type
        self.gating_type = gating_type.lower()
        if self.gating_type not in ['softmax', 'sigmoid']:
            raise ValueError(
                f"Unsupported gating_type: {gating_type}. "
                "Must be 'softmax' or 'sigmoid'."
            )
        
        # Feature aggregation and adaptation
        self.feature_aggregator = nn.AdaptiveAvgPool2d((1, 1))
        self.adaptive_projection = None

        # SAGE Routing Components
        # Component 1: Shared Expert Gate (g_s) - Eq. 6
        self.shared_expert_gate = nn.Linear(in_channels, 1)
        # Initialize gate to 0.5 (balanced) at startup: sigmoid(0) = 0.5
        nn.init.zeros_(self.shared_expert_gate.weight)
        nn.init.zeros_(self.shared_expert_gate.bias)

        # Component 2: Fine-grained Router via SAR - Eq. 7
        self.query_projection = nn.Linear(in_channels, router_hidden_dim)
        self.expert_keys = nn.Parameter(
            torch.randn(expert_pool_size, router_hidden_dim)
        )
        nn.init.normal_(
            self.expert_keys,
            mean=0,
            std=1 / np.sqrt(router_hidden_dim)
        )
        self.temperature = np.sqrt(router_hidden_dim)
        
        # Noise projection for exploration during training
        self.noise_projection = nn.Linear(in_channels, expert_pool_size)

        # Component 3: Shared expert mask for logit modulation - Eq. 8
        self.register_buffer(
            'shared_mask',
            torch.zeros(expert_pool_size, dtype=torch.float32)
        )
        self._update_shared_mask()
        
        # Statistics tracking
        self.register_buffer(
            'expert_usage_count',
            torch.zeros(expert_pool_size)
        )
        self.register_buffer('total_calls', torch.tensor(0))
        
        # Logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(
            f"Initialized SAGE Hierarchical Router: "
            f"top_k={self.top_k}, experts={expert_pool_size}, "
            f"gating={gating_type}, logit_modulation={self.logit_modulation}"
        )

    def _update_shared_mask(self):
        """
        Update the binary mask for logit modulation based on shared expert indices.
        
        Creates a binary tensor where positions corresponding to shared experts
        are set to 1.0, and all other positions are 0.0.
        """
        shared_mask_float = torch.zeros(
            self.expert_pool_size,
            dtype=torch.float32,
            device=self.shared_mask.device
        )
        if self.shared_expert_indices:
            shared_mask_float[self.shared_expert_indices] = 1.0
        self.shared_mask.data = shared_mask_float

    def _aggregate_features_with_adaptation(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Aggregate input features and adapt channel dimensions if needed.
        
        Handles different input formats:
        - CNN format (B, C, H, W): Apply adaptive pooling
        - Transformer format (B, N, D): Take mean over sequence dimension
        - Other formats: Flatten to vector
        
        Args:
            x: Input tensor of any supported format
            
        Returns:
            Aggregated feature vector of shape (B, in_channels)
        """
        # Step 1: Aggregate based on input format
        if x.dim() == 4:  # CNN format (B, C, H, W)
            aggregated = self.feature_aggregator(x).squeeze(-1).squeeze(-1)
        elif x.dim() == 3:  # Transformer format (B, N, D)
            aggregated = x.mean(dim=1)
        else:  # Fallback for other formats
            aggregated = x.flatten(1)
        
        # Step 2: Handle channel mismatch with adaptive projection
        actual_channels = aggregated.shape[1]
        if actual_channels != self.in_channels:
            if self.adaptive_projection is None:
                self.adaptive_projection = nn.Linear(
                    actual_channels,
                    self.in_channels
                ).to(x.device)
            aggregated = self.adaptive_projection(aggregated)
        
        return aggregated
    
    def forward(
        self,
        input_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Perform hierarchical expert routing using SAGE method.
        
        Processing Steps:
            1. Aggregate and adapt input features
            2. Compute shared expert gate (g_s) - Eq. 6
            3. Compute base routing logits via SAR - Eq. 7
            4. Apply logit modulation using g_s - Eq. 8
            5. Perform unified top-k selection - Eq. 9
            6. Compute gating weights and load balance loss
        
        Args:
            input_tensor: Input tensor of shape (B, C, H, W) or (B, N, D)
            
        Returns:
            Tuple containing:
                - top_k_indices: Selected expert indices (B, K)
                - gating_weights: Weights for selected experts (B, K)
                - routing_info: Dictionary with routing metadata
        """
        self.total_calls += 1
        B = input_tensor.shape[0]
        
        # Step 1: Aggregate and adapt input features
        aggregated_features = self._aggregate_features_with_adaptation(input_tensor)
        
        # Step 2: Compute shared expert gate (g_s) - SAGE Eq. 6
        g_s = torch.sigmoid(self.shared_expert_gate(aggregated_features))

        # Step 3: Compute base routing logits via SAR - SAGE Eq. 7
        query = self.query_projection(aggregated_features)
        base_logits = torch.matmul(query, self.expert_keys.T) / self.temperature
        
        # Add exploration noise during training
        if self.training:
            noise = torch.randn_like(base_logits) * F.softplus(
                self.noise_projection(aggregated_features)
            )
            base_logits = base_logits + noise

        # Step 4: Logit modulation using hierarchical gating - SAGE Eq. 8
        if self.logit_modulation:
            eps = 1e-9
            log_g_s = torch.log(g_s + eps)
            log_one_minus_g_s = torch.log(1 - g_s + eps)
            modulated_logits = (
                base_logits +
                self.shared_mask * log_g_s +
                (1 - self.shared_mask) * log_one_minus_g_s
            )
        else:
            modulated_logits = base_logits

        # Step 5: Unified top-k selection - SAGE Eq. 9
        top_k_logits, top_k_indices = torch.topk(
            modulated_logits,
            self.top_k,
            dim=-1
        )
        
        # Apply gating function
        if self.gating_type == 'softmax':
            gating_weights = F.softmax(top_k_logits, dim=-1)
        elif self.gating_type == 'sigmoid':
            gating_weights = torch.sigmoid(top_k_logits)
        
        # Step 6: Update statistics and compute auxiliary losses
        if self.training:
            indices_one_hot = F.one_hot(
                top_k_indices,
                num_classes=self.expert_pool_size
            ).sum(dim=1)
            self.expert_usage_count += indices_one_hot.sum(dim=0)
        
        # Compute load balance loss
        full_probs_for_loss = F.softmax(modulated_logits, dim=-1)
        load_balance_loss = self.compute_load_balance_loss(full_probs_for_loss)
        
        # Prepare routing info
        routing_info = {
            'load_balance_loss': load_balance_loss,
            'g_s_score_sample_0': g_s[0].item(),
        }
        
        # Add detailed eval information for debugging (only in eval mode)
        if not self.training and B > 0:
            routing_info['eval_base_logits_sample_0'] = (
                base_logits[0].detach().cpu().float().numpy()
            )
            routing_info['eval_top_k_logits_sample_0'] = (
                top_k_logits[0].detach().cpu().float().numpy()
            )
        
        return top_k_indices, gating_weights, routing_info 
    
    def compute_load_balance_loss(self, expert_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute load balancing auxiliary loss for fair expert utilization.
        
        Implements the Switch Transformer load balancing loss:
            L_balance = N * sum(f_i * P_i)
        where:
            - N is the number of experts
            - f_i is the fraction of samples routed to expert i
            - P_i is the average routing probability for expert i
        
        Args:
            expert_probs: Routing probabilities (B, expert_pool_size)
            
        Returns:
            Scaled load balance loss tensor
        """
        if expert_probs.dim() != 2 or self.expert_pool_size == 0:
            return torch.tensor(0.0, device=expert_probs.device)
        
        # P_i: Average routing probability for expert i over the batch
        P = expert_probs.mean(dim=0)
        
        # f_i: Fraction of samples routed to expert i (soft approximation)
        f = P
        
        # Switch Transformer Loss: N * sum(f_i * P_i)
        load_balance_loss = self.expert_pool_size * (f * P).sum()
        
        return self.load_balance_factor * load_balance_loss
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """
        Get expert usage statistics for monitoring and analysis.
        
        Returns:
            Dictionary containing:
                - total_forward_calls: Number of forward passes executed
                - expert_usage_count: Raw usage count for each expert
                - expert_usage_ratio: Normalized usage ratio for each expert
                - shared_expert_indices: List of shared expert indices
        """
        total_samples_processed = self.total_calls.item()
        usage_counts = self.expert_usage_count.cpu().numpy()
        
        # Compute usage ratios
        total_routed_items = usage_counts.sum()
        if total_routed_items > 0:
            usage_ratios = usage_counts / total_routed_items
        else:
            usage_ratios = np.zeros_like(usage_counts)
        
        return {
            'total_forward_calls': total_samples_processed,
            'expert_usage_count': usage_counts.tolist(),
            'expert_usage_ratio': usage_ratios.tolist(),
            'shared_expert_indices': self.shared_expert_indices,
        }


# ============================================================================
# Factory Functions
# ============================================================================

def create_sage_router(
    in_channels: int,
    expert_pool_size: int,
    expert_infos: List[Dict[str, Any]],
    config: Dict[str, Any],
    **kwargs
) -> SageRouter:
    """
    Factory function to create a SAGE router with configuration.
    
    Args:
        in_channels: Input channel dimension
        expert_pool_size: Total number of experts
        expert_infos: Metadata for each expert
        config: Configuration dictionary with keys:
            - 'top_k' (int): Number of experts to select (default: 1)
            - 'gating_type' (str): Gating function type (default: 'softmax')
            - 'shared_expert_indices' (list): Shared expert indices (default: [])
            - 'router_hidden_dim' (int): Hidden dimension (default: 256)
            - 'load_balance_factor' (float): Load balance weight (default: 0.01)
            - 'logit_modulation' (bool): Enable SAGE logit modulation (default: True)
        **kwargs: Additional keyword arguments (ignored)
    
    Returns:
        Configured SageRouter instance
    """
    return SageRouter(
        in_channels=in_channels,
        expert_pool_size=expert_pool_size,
        expert_infos=expert_infos,
        top_k=config.get('top_k', 1),
        gating_type=config.get('gating_type', 'softmax'),
        shared_expert_indices=config.get('shared_expert_indices', []),
        router_hidden_dim=config.get('router_hidden_dim', 256),
        load_balance_factor=config.get('load_balance_factor', 0.01),
        logit_modulation=config.get('logit_modulation', True)
    )
