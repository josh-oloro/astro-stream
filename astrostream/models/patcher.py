"""
BIOT Patcher: Channel-independent patch tokenizer for biosignals.

Reference: Yang, Westover & Sun (2023). BIOT: Biosignal Foundation Model.
NeurIPS 2023. arXiv:2305.10351.

Implements a learnable tokenization layer that:
- Splits biosignal channels into non-overlapping temporal patches
- Projects patches to d_model dimensions using shared channel-independent weights
- Adds positional encoding combining temporal position + channel identity
- Supports random masking for self-supervised learning (JEPA, masked prediction)

Key insight: Channel-independent projection allows the same patcher to work
on datasets with heterogeneous electrode configurations (208 MEG vs 273 EEG).

Inputs:
  x: (B, C, T) biosignal matrix
     B = batch size
     C = number of channels
     T = total time samples

Outputs:
  - patch_embeddings: (B, C * n_patches, d_model)
  - mask_indices: (B, C * n_patches) BoolTensor or None
"""

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn


class BIOTPatcher(nn.Module):
    """
    Channel-independent patch tokenizer for MEG/EEG biosignals.

    Converts raw biosignal time series into learnable patch tokens.
    The projection layer is shared across all channels and patches,
    enabling transfer across datasets with different electrode counts.
    """

    def __init__(
        self,
        patch_len: int = 50,
        d_model: int = 512,
        mask_token_init: float = 0.0,
    ) -> None:
        """
        Initialize BIOT patcher.

        Args:
            patch_len: length of each temporal patch in samples (default 50 @ 200 Hz = 250 ms)
            d_model: output embedding dimension (default 512)
            mask_token_init: initial value for learnable [MASK] token (default 0.0)
        """
        super().__init__()

        self.patch_len = patch_len  # samples per patch
        self.d_model = d_model  # embedding dimension

        # Shared linear projection: patch_len samples → d_model dimensions
        # Applied channel-independently to every (channel, patch) pair
        self.projection = nn.Linear(patch_len, d_model, bias=True)
        # Shape: (patch_len,) → (d_model,)

        # Learnable [MASK] token
        # Applied when mask_rate > 0 to random patches
        self.mask_token = nn.Parameter(
            torch.full((d_model,), mask_token_init, dtype=torch.float32)
        )  # (d_model,)

        # Learnable channel embeddings
        # Initialized separately; will be concatenated with temporal positional encoding
        # We'll set n_channels dynamically in forward pass
        self.channel_embeddings = None  # Lazy init

    def _init_channel_embeddings(self, n_channels: int) -> None:
        """
        Lazily initialize channel embeddings based on number of channels in input.

        Args:
            n_channels: number of channels in the input (e.g., 208 for Gwilliams MEG)
        """
        if self.channel_embeddings is None or self.channel_embeddings.size(0) != n_channels:
            self.channel_embeddings = nn.Parameter(
                torch.randn(n_channels, self.d_model, dtype=torch.float32) * 0.02
            )  # (n_channels, d_model), initialized with small random values

    def _apply_sinusoidal_positional_encoding(
        self, n_patches: int, n_channels: int, device: torch.device
    ) -> torch.Tensor:
        """
        Generate sinusoidal positional encoding for temporal positions.

        Uses standard transformer PE: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
                                     PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

        Args:
            n_patches: number of patches per channel
            n_channels: number of channels
            device: torch device (cpu/cuda)

        Returns:
            positional_encoding: (n_channels * n_patches, d_model)
        """
        pos = torch.arange(n_patches, dtype=torch.float32, device=device)  # (n_patches,)
        dim = torch.arange(0, self.d_model, 2, dtype=torch.float32, device=device)  # (d_model/2,)

        # Standard sinusoidal PE formula
        # pe[t, 2i] = sin(t / 10000^(2i/d_model))
        # pe[t, 2i+1] = cos(t / 10000^(2i/d_model))
        div_term = 10000 ** (dim / self.d_model)  # (d_model/2,)
        pe_temporal = torch.zeros(n_patches, self.d_model, device=device)  # (n_patches, d_model)

        pe_temporal[:, 0::2] = torch.sin(pos[:, None] / div_term[None, :])  # sin terms
        pe_temporal[:, 1::2] = torch.cos(pos[:, None] / div_term[None, :])  # cos terms

        # Expand to all channels: (n_channels * n_patches, d_model)
        # Repeat each patch PE for all channels
        pe_expanded = pe_temporal.repeat_interleave(n_channels, dim=0)  # (n_channels * n_patches, d_model)

        return pe_expanded  # (n_channels * n_patches, d_model)

    def forward(
        self, x: torch.Tensor, mask: bool = False, mask_rate: float = 0.15
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Tokenize biosignal into patches and generate embeddings.

        Args:
            x: input tensor of shape (B, C, T)
               B = batch size
               C = number of channels
               T = total time samples
            mask: whether to apply random patch masking (for self-supervised learning)
            mask_rate: fraction of patches to mask (default 0.15)

        Returns:
            patch_embeddings: (B, C * n_patches, d_model)
                Learned patch tokens with positional encoding
            mask_indices: (B, C * n_patches) BoolTensor or None
                Boolean mask indicating which patches were masked
                None if mask=False
        """
        batch_size, n_channels, n_samples = x.shape  # (B, C, T)

        # Initialize channel embeddings if needed
        self._init_channel_embeddings(n_channels)

        # === Step 1: Split into patches (channel-independent) ===
        # Verify input length is divisible by patch_len
        if n_samples % self.patch_len != 0:
            raise ValueError(
                f"Input length {n_samples} must be divisible by patch_len {self.patch_len}. "
                f"Got remainder {n_samples % self.patch_len}."
            )

        n_patches = n_samples // self.patch_len  # int

        # Reshape: (B, C, T) → (B, C, n_patches, patch_len)
        x_patches = x.reshape(batch_size, n_channels, n_patches, self.patch_len)
        # (B, C, n_patches, patch_len)

        # === Step 2: Project patches to d_model (shared across channels) ===
        # Flatten: (B, C, n_patches, patch_len) → (B * C * n_patches, patch_len)
        x_flat = x_patches.reshape(-1, self.patch_len)  # (B * C * n_patches, patch_len)

        # Apply shared projection layer
        patch_embeddings_flat = self.projection(x_flat)  # (B * C * n_patches, d_model)

        # Reshape back: (B * C * n_patches, d_model) → (B, C * n_patches, d_model)
        patch_embeddings = patch_embeddings_flat.reshape(
            batch_size, n_channels * n_patches, self.d_model
        )  # (B, C * n_patches, d_model)

        # === Step 3: Add positional encoding (temporal + channel) ===
        # Generate temporal sinusoidal PE
        pe_temporal = self._apply_sinusoidal_positional_encoding(
            n_patches, n_channels, device=x.device
        )  # (n_channels * n_patches, d_model)

        # Add channel embeddings: repeat for each patch
        channel_emb_expanded = self.channel_embeddings.repeat_interleave(
            n_patches, dim=0
        )  # (n_channels * n_patches, d_model)

        # Combine temporal and channel positional information
        positional_encoding = pe_temporal + channel_emb_expanded  # (n_channels * n_patches, d_model)

        # Add to patch embeddings (broadcast over batch)
        patch_embeddings = patch_embeddings + positional_encoding[None, :, :]  # (B, C * n_patches, d_model)

        # === Step 4: Optional masking for self-supervised learning ===
        mask_indices = None

        if mask:
            n_total_patches = n_channels * n_patches  # int
            n_mask = int(n_total_patches * mask_rate)  # number of patches to mask

            # Randomly select patches to mask (same mask across batch)
            mask_indices = torch.bernoulli(
                torch.full(
                    (batch_size, n_total_patches),
                    mask_rate,
                    dtype=torch.float32,
                    device=x.device,
                )
            ).bool()  # (B, C * n_patches)

            # Replace masked patches with learnable [MASK] token
            patch_embeddings = torch.where(
                mask_indices[:, :, None],
                self.mask_token[None, None, :].expand(batch_size, n_total_patches, self.d_model),
                patch_embeddings,
            )  # (B, C * n_patches, d_model)

        return patch_embeddings, mask_indices  # (B, C * n_patches, d_model), (B, C * n_patches) or None
