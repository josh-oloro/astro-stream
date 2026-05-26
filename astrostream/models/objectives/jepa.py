"""
JEPA (Joint-Embedding Predictive Architecture) Training Objective.

Reference: LeCun, Y. (2022). A path towards autonomous machine intelligence.
OpenReview BZ5a1r-kVsf.

Implements a self-supervised learning framework for learning representations
of biosignals without requiring downstream task labels. The architecture consists of:

1. OnlineEncoder: main AstroStream model (Patcher + Mamba + Gate)
2. TargetEncoder: EMA-updated copy of OnlineEncoder (no gradients)
3. Predictor: shallow cross-attention network that predicts masked patches

Training objective: minimize L2 distance between predicted and target
representations in latent space (NOT pixel/signal space).

Key insight: Predicting future/masked latent embeddings is harder than
predicting raw signals, forcing the model to learn meaningful high-level
representations.

Inputs:
  x: (B, C, T) raw MEG/EEG signal or (B, C*n_patches, D) already tokenized
  online_encoder: full AstroStream model
  ema_encoder: EMATargetEncoder wrapping a copy of online_encoder
  predictor: JEPAPredictor network

Outputs:
  {
    'loss': scalar JEPA loss
    'n_masked': number of masked patches
    'context_repr': (B, T_ctx, D) for downstream tasks
  }
"""

import copy
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EMATargetEncoder(nn.Module):
    """
    Exponential Moving Average (EMA) target encoder.

    Maintains a slow copy of the online encoder via EMA updates.
    Provides target representations for contrastive learning.
    Receives NO gradients (frozen during training).
    """

    def __init__(
        self,
        online_encoder: nn.Module,
        momentum_start: float = 0.996,
        momentum_end: float = 0.9999,
    ) -> None:
        """
        Initialize EMA target encoder.

        Args:
            online_encoder: the online AstroStream encoder (Patcher + Mamba + Gate)
            momentum_start: initial EMA momentum (beginning of training)
            momentum_end: final EMA momentum (end of training)
        """
        super().__init__()

        # Deep copy the online encoder
        self.target_encoder = copy.deepcopy(online_encoder)

        # Disable all gradients on target encoder
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # Store momentum schedule parameters
        self.momentum_start = momentum_start  # typically 0.996
        self.momentum_end = momentum_end  # typically 0.9999 (closer to 1.0 = slower)

        # Reference to online encoder for EMA updates
        self.online_encoder = online_encoder

    def update(self, step: int, total_steps: int) -> None:
        """
        Update target encoder parameters via EMA.

        Uses cosine annealing schedule for momentum:
        τ(step) = momentum_end - (momentum_end - momentum_start) * cos(π * step / total_steps) / 2

        This starts with momentum_start and smoothly increases to momentum_end.

        Args:
            step: current training step (0-indexed)
            total_steps: total number of steps in epoch/training
        """
        # Compute current momentum using cosine schedule
        progress = step / max(total_steps, 1)  # [0, 1]
        cos_term = math.cos(math.pi * progress) / 2.0  # [0, -0.5] over progress
        momentum = self.momentum_end - (self.momentum_end - self.momentum_start) * cos_term
        # At step=0: momentum = momentum_start
        # At step=total_steps: momentum = momentum_end

        # Update each target parameter via EMA
        with torch.no_grad():
            for param_t, param_o in zip(
                self.target_encoder.parameters(), self.online_encoder.parameters()
            ):
                # param_t.data = τ * param_t.data + (1-τ) * param_o.data
                param_t.data.mul_(momentum).add_(param_o.data, alpha=1 - momentum)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute target representations.

        Args:
            x: input tensor (B, C*n_patches, D) or (B, C, T)

        Returns:
            target_repr: (B, T, D) target representations (no gradients)
        """
        return self.target_encoder(x)

    def extra_repr(self) -> str:
        return f"momentum_start={self.momentum_start}, momentum_end={self.momentum_end}"


class CrossAttentionBlock(nn.Module):
    """
    Single cross-attention block for JEPA predictor.

    Combines self-attention on queries + cross-attention to context.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        """
        Initialize cross-attention block.

        Args:
            d_model: embedding dimension
            n_heads: number of attention heads
            dropout: dropout rate
        """
        super().__init__()

        # Self-attention on masked (query) embeddings
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # Cross-attention to context
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # Feed-forward network
        ff_dim = 4 * d_model  # standard expansion
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,  # (B, T_mask, D)
        context: torch.Tensor,  # (B, T_ctx, D)
    ) -> torch.Tensor:
        """
        Forward pass with pre-norm residual connections.

        Args:
            query: (B, T_mask, D) masked positions to predict
            context: (B, T_ctx, D) unmasked context

        Returns:
            output: (B, T_mask, D)
        """
        # Self-attention on queries
        query_norm = self.norm1(query)
        self_attn_out, _ = self.self_attn(query_norm, query_norm, query_norm)
        query = query + self.dropout(self_attn_out)

        # Cross-attention to context
        query_norm = self.norm2(query)
        cross_attn_out, _ = self.cross_attn(
            query_norm, context, context
        )  # Q=query, K=V=context
        query = query + self.dropout(cross_attn_out)

        # Feed-forward
        query_norm = self.norm3(query)
        ff_out = self.ff(query_norm)
        query = query + self.dropout(ff_out)

        return query


class JEPAPredictor(nn.Module):
    """
    Shallow cross-attention transformer for JEPA prediction.

    Predicts masked target representations given unmasked context.
    Uses learnable [MASK] query tokens + cross-attention to context.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize JEPA predictor.

        Args:
            d_model: embedding dimension (default 512)
            n_heads: number of attention heads (default 8)
            n_layers: number of cross-attention layers (default 2)
            dropout: dropout rate (default 0.1)
        """
        super().__init__()

        # Stack of cross-attention blocks
        self.blocks = nn.ModuleList(
            [CrossAttentionBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )

        # Learnable [MASK] query token
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)  # (1, 1, D)

    def forward(
        self,
        context: torch.Tensor,  # (B, T_ctx, D)
        mask_indices: torch.Tensor,  # (B, T_total) boolean or (B, n_mask) indices
    ) -> torch.Tensor:  # (B, n_mask, D)
        """
        Predict masked representations using context.

        Args:
            context: (B, T_ctx, D) unmasked online encoder representations
            mask_indices: (B, T_total) boolean mask or (B, n_mask) indices

        Returns:
            predictions: (B, n_mask, D) predicted representations for masked patches
        """
        batch_size = context.shape[0]
        n_mask = mask_indices.sum() if mask_indices.dtype == torch.bool else mask_indices.shape[1]
        n_mask = int(n_mask)

        # Expand [MASK] tokens for each masked position
        # (1, 1, D) → (B, n_mask, D)
        mask_tokens = self.mask_token.expand(batch_size, n_mask, -1)  # (B, n_mask, D)

        # Forward through cross-attention blocks
        query = mask_tokens  # (B, n_mask, D)
        for block in self.blocks:
            query = block(query, context)  # (B, n_mask, D)

        return query  # (B, n_mask, D)


class JEPALoss(nn.Module):
    """
    JEPA loss: L2-normalized MSE in latent space.

    Does NOT reconstruct raw MEG/EEG signals.
    Loss operates entirely on learned representations.
    """

    def __init__(self) -> None:
        """Initialize JEPA loss module."""
        super().__init__()

    def forward(
        self,
        predicted: torch.Tensor,  # (B, n_mask, D)
        target: torch.Tensor,  # (B, n_mask, D)
    ) -> torch.Tensor:  # scalar
        """
        Compute L2-normalized MSE loss.

        Args:
            predicted: (B, n_mask, D) predictions from predictor
            target: (B, n_mask, D) target representations from EMA encoder

        Returns:
            loss: scalar MSE loss on normalized embeddings
        """
        # L2 normalize both predicted and target along embedding dimension
        pred_norm = F.normalize(predicted, dim=-1)  # (B, n_mask, D)
        tgt_norm = F.normalize(target, dim=-1)  # (B, n_mask, D)

        # Compute MSE on normalized representations
        loss = F.mse_loss(pred_norm, tgt_norm)  # scalar

        return loss


class JEPAObjective(nn.Module):
    """
    Top-level JEPA objective module.

    Coordinates:
    1. Online encoder forward + masking (from Patcher)
    2. Target encoder inference
    3. Predictor on unmasked context
    4. Loss computation + EMA update
    """

    def __init__(
        self,
        online_encoder: nn.Module,
        predictor: Optional[JEPAPredictor] = None,
        ema_momentum_start: float = 0.996,
        ema_momentum_end: float = 0.9999,
        normalize_loss: bool = True,
    ) -> None:
        """
        Initialize JEPA objective.

        Args:
            online_encoder: main AstroStream model (Patcher + Mamba + Gate)
            predictor: JEPAPredictor network (created if None)
            ema_momentum_start: initial EMA momentum
            ema_momentum_end: final EMA momentum
            normalize_loss: whether to L2-normalize before MSE loss
        """
        super().__init__()

        self.online_encoder = online_encoder

        # Create predictor if not provided
        if predictor is None:
            d_model = 512  # default
            predictor = JEPAPredictor(d_model, n_heads=8, n_layers=2)
        self.predictor = predictor

        # Create EMA target encoder
        self.ema_encoder = EMATargetEncoder(
            online_encoder,
            momentum_start=ema_momentum_start,
            momentum_end=ema_momentum_end,
        )

        # Loss module
        self.loss_fn = JEPALoss()

    def forward(
        self,
        x: torch.Tensor,  # (B, C*n_patches, D) or (B, C, T)
        mask_rate: float = 0.15,
        step: int = 0,
        total_steps: int = 1000,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute JEPA loss.

        Args:
            x: input tensor (B, C*n_patches, D) already patched or (B, C, T) raw
            mask_rate: fraction of patches to mask (default 0.15)
            step: current training step (for EMA schedule)
            total_steps: total steps in epoch (for EMA schedule)

        Returns:
            dict with keys:
                - 'loss': scalar JEPA loss
                - 'n_masked': int, number of masked patches
                - 'predictions': (B, n_mask, D) model predictions
                - 'targets': (B, n_mask, D) target representations
        """
        # Online encoder forward pass with masking
        online_repr, mask_indices = self.online_encoder(x, mask=True, mask_rate=mask_rate)
        # online_repr: (B, T, D)
        # mask_indices: (B, T) boolean

        # Separate masked and unmasked patches
        batch_size, n_total, d_model = online_repr.shape
        mask_bool = mask_indices.bool()  # ensure boolean
        n_masked = mask_bool.sum().item()

        # Extract unmasked context
        context = online_repr[:, ~mask_bool, :]  # (B, T_ctx, D)

        # Target encoder (EMA) - no gradients
        target_repr = self.ema_encoder(x)  # (B, T, D)
        target_masked = target_repr[:, mask_bool, :]  # (B, n_mask, D)

        # Predictor: predict masked positions from context
        predictions = self.predictor(context, mask_indices)  # (B, n_mask, D)

        # Compute loss
        loss = self.loss_fn(predictions, target_masked)

        # Update EMA encoder
        self.ema_encoder.update(step, total_steps)

        return {
            "loss": loss,
            "n_masked": n_masked,
            "predictions": predictions,
            "targets": target_masked,
            "online_repr": online_repr,
            "context": context,
        }

    def reset_state(self) -> None:
        """Reset encoder states at recording boundaries."""
        if hasattr(self.online_encoder, "reset_state"):
            self.online_encoder.reset_state()
        if hasattr(self.ema_encoder.target_encoder, "reset_state"):
            self.ema_encoder.target_encoder.reset_state()
