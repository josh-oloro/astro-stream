"""
AstroStreamModel: Full neural decoding architecture.

Top-level nn.Module that assembles all components:
  - BIOTPatcher: channel-independent patch tokenization
  - PersistentMambaEncoder: selective state-space backbone with optional persistent state
  - AstrocyteGate: biologically-inspired lateral gating (optional)
  - JEPAObjective: self-supervised contrastive learning
  - Linear decoder: speech decoding classifier

Also includes baseline models (Transformer, LSTM) for ablation studies.

Inputs:
  x: (B, C, T) raw MEG/EEG signal
  recording_ids: list[int] for persistent state reset at boundaries
  mask_for_jepa: bool, whether to mask patches for JEPA objective

Outputs:
  {
    'logits': (B, n_classes) speech decoding predictions
    'repr': (B, D) pooled representation for downstream tasks
    'gate_signal': (B, 1) or None, astrocyte slow state
    'masked_repr': (B, T_ctx, D) context for JEPA loss
    'mask_indices': (B, T) boolean mask
  }
"""

from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from astrostream.models.patcher import BIOTPatcher
from astrostream.models.gates.astrocyte_gate import AstrocyteGate
from astrostream.models.objectives.jepa import JEPAObjective


class PersistentMambaEncoder(nn.Module):
    """
    Placeholder for Mamba encoder with optional persistent state.
    
    Will be replaced with full mamba_ssm implementation.
    For now, returns dummy outputs with correct shapes.
    """

    def __init__(self, d_model: int, n_layers: int, d_state: int = 16, persistent: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_state = d_state
        self.persistent = persistent
        
        # Placeholder: linear projection layers for each Mamba block
        self.layers = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_layers)
        ])
        
        # Hidden state buffer for persistent mode
        if self.persistent:
            self.register_buffer("hidden_state", None)

    def reset_state(self):
        """Reset persistent hidden state at recording boundaries."""
        if self.persistent:
            self.hidden_state = None

    def forward(self, x: torch.Tensor, recording_ids: Optional[List[int]] = None) -> torch.Tensor:
        """
        Forward pass through Mamba encoder.
        
        Args:
            x: (B, T, D) or (B, C*n_patches, D) patch embeddings
            recording_ids: list of recording IDs for state reset
            
        Returns:
            h_t: (B, T, D) encoded representations
        """
        # Handle (B, C*n_patches, D) by treating as (B, T, D)
        if len(x.shape) == 3:
            batch_size, n_patches, d_model = x.shape
        else:
            raise ValueError(f"Expected 3D input, got {x.shape}")
        
        # Process through layers (placeholder)
        h = x
        for layer in self.layers:
            h = F.relu(layer(h))
        
        return h  # (B, T, D)


class AstroStreamModel(nn.Module):
    """
    Full Astro-Stream neural decoding model.
    
    Combines Mamba SSM backbone + optional persistent state + astrocyte gating
    + self-supervised JEPA objective + linear speech decoding classifier.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initialize Astro-Stream model from config.
        
        Args:
            cfg: OmegaConf config with keys:
                - d_model, n_layers, dropout
                - mamba: {d_state, d_conv, expand, persistent_state}
                - astrocyte_gate: {enabled, alpha, init_theta, init_w}
                - jepa: {enabled, ema_momentum_start, ema_momentum_end, ...}
                - decoder: {n_speech_classes}
        """
        super().__init__()
        self.cfg = cfg

        # Extract config values
        self.d_model = cfg.d_model  # 512
        self.n_layers = cfg.n_layers  # 6
        self.dropout = cfg.dropout  # 0.1
        self.patch_len = cfg.patch_length_samples  # 50
        
        # === Component 1: Patcher ===
        self.patcher = BIOTPatcher(
            patch_len=self.patch_len,
            d_model=self.d_model,
            mask_token_init=0.0,
        )
        
        # === Component 2: Mamba Encoder ===
        persistent_state = cfg.mamba.get("persistent_state", True)
        self.encoder = PersistentMambaEncoder(
            d_model=self.d_model,
            n_layers=self.n_layers,
            d_state=cfg.mamba.d_state,
            persistent=persistent_state,
        )
        
        # === Component 3: Astrocyte Gate (optional) ===
        if cfg.astrocyte_gate.enabled:
            self.gate = AstrocyteGate(
                alpha=cfg.astrocyte_gate.alpha,
                init_theta=cfg.astrocyte_gate.init_theta,
                init_w=cfg.astrocyte_gate.init_w,
            )
        else:
            self.gate = nn.Identity()
        
        # === Component 4: Linear Decoder (speech classification) ===
        n_classes = cfg.decoder.n_speech_classes
        self.decoder = nn.Linear(self.d_model, n_classes, bias=True)
        
        # === Component 5: JEPA Objective (optional) ===
        if cfg.jepa.enabled:
            self.jepa = JEPAObjective(
                online_encoder=self,  # Will be set properly in training loop
                predictor=None,  # Created inside JEPAObjective
                ema_momentum_start=cfg.jepa.ema_momentum_start,
                ema_momentum_end=cfg.jepa.ema_momentum_end,
            )
        else:
            self.jepa = None

    def reset_state(self) -> None:
        """Reset persistent state at recording boundaries."""
        self.encoder.reset_state()
        if hasattr(self.gate, "reset_state"):
            self.gate.reset_state()

    def forward(
        self,
        x: torch.Tensor,  # (B, C, T)
        recording_ids: Optional[List[int]] = None,
        mask_for_jepa: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through full model.
        
        Args:
            x: (B, C, T) raw MEG/EEG signal
            recording_ids: list of recording IDs for persistent state reset
            mask_for_jepa: whether to mask patches for JEPA
            
        Returns:
            dict with keys:
                - 'logits': (B, n_classes) speech decoding predictions
                - 'repr': (B, D) pooled representation
                - 'gate_signal': (B, 1) or None, astrocyte slow state
                - 'masked_repr': (B, T_ctx, D) or None, context for JEPA
                - 'mask_indices': (B, T) or None, boolean mask
        """
        batch_size, n_channels, n_samples = x.shape  # (B, C, T)
        
        # Step 1: Patcher - tokenize to patches
        patch_emb, mask_indices = self.patcher(x, mask=mask_for_jepa, mask_rate=self.cfg.mask_rate)
        # patch_emb: (B, C*n_patches, D)
        # mask_indices: (B, C*n_patches) or None
        
        # Step 2: Mamba Encoder - encode patches
        h_encoded = self.encoder(patch_emb, recording_ids=recording_ids)
        # h_encoded: (B, C*n_patches, D)
        
        # Step 3: Astrocyte Gate - modulate encoder output
        if isinstance(self.gate, nn.Identity):
            h_gated = h_encoded
            s_T = None
        else:
            h_gated, s_T = self.gate(h_encoded)
            # h_gated: (B, C*n_patches, D)
            # s_T: (B, 1)
        
        # Step 4: Linear Decoder - compute speech decoding logits
        # Pool over time (mean across patches)
        repr_pooled = h_gated.mean(dim=1)  # (B, D)
        logits = self.decoder(repr_pooled)  # (B, n_classes)
        
        # Step 5: Prepare JEPA context (unmasked patches)
        masked_repr = None
        if mask_for_jepa and mask_indices is not None:
            mask_bool = mask_indices.bool()
            masked_repr = h_encoded[:, ~mask_bool, :]  # (B, T_ctx, D)
        
        return {
            "logits": logits,  # (B, n_classes)
            "repr": repr_pooled,  # (B, D)
            "gate_signal": s_T,  # (B, 1) or None
            "masked_repr": masked_repr,  # (B, T_ctx, D) or None
            "mask_indices": mask_indices,  # (B, C*n_patches) or None
            "encoded": h_encoded,  # (B, C*n_patches, D) for inspection
        }

    def get_n_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_config(cls, cfg_path: str) -> "AstroStreamModel":
        """
        Load model from YAML config file.
        
        Args:
            cfg_path: path to YAML config (e.g., "configs/model/astrostream_base.yaml")
            
        Returns:
            AstroStreamModel instance
        """
        cfg_path = Path(cfg_path)
        cfg = OmegaConf.load(cfg_path)
        return cls(cfg)

    def extra_repr(self) -> str:
        """Return extra module representation."""
        n_params = self.get_n_params()
        gate_status = "enabled" if not isinstance(self.gate, nn.Identity) else "disabled"
        return f"d_model={self.d_model}, n_layers={self.n_layers}, n_params={n_params}, gate={gate_status}"


class TransformerBaselineModel(nn.Module):
    """
    Standard Transformer baseline (episodic, no persistent state, no gating).
    
    Used as comparison baseline in ablation studies.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initialize Transformer baseline.
        
        Args:
            cfg: config dict (same interface as AstroStreamModel)
        """
        super().__init__()
        self.cfg = cfg
        
        self.d_model = cfg.d_model
        self.n_layers = cfg.n_layers
        self.patch_len = cfg.patch_length_samples
        
        # Patcher
        self.patcher = BIOTPatcher(
            patch_len=self.patch_len,
            d_model=self.d_model,
            mask_token_init=0.0,
        )
        
        # Standard Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=8,
            dim_feedforward=4 * self.d_model,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
        
        # Linear decoder
        n_classes = cfg.decoder.n_speech_classes
        self.decoder = nn.Linear(self.d_model, n_classes, bias=True)

    def forward(
        self,
        x: torch.Tensor,  # (B, C, T)
        recording_ids: Optional[List[int]] = None,
        mask_for_jepa: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass (episodic, no masking by default)."""
        # Patcher
        patch_emb, _ = self.patcher(x, mask=False)  # (B, C*n_patches, D)
        
        # Transformer encoder
        h_encoded = self.encoder(patch_emb)  # (B, C*n_patches, D)
        
        # Decoder
        repr_pooled = h_encoded.mean(dim=1)  # (B, D)
        logits = self.decoder(repr_pooled)  # (B, n_classes)
        
        return {
            "logits": logits,
            "repr": repr_pooled,
            "gate_signal": None,
            "encoded": h_encoded,
        }

    def get_n_params(self) -> int:
        """Return trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LSTMBaselineModel(nn.Module):
    """
    LSTM baseline (episodic, no persistent state, no gating).
    
    Used as comparison baseline in ablation studies.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initialize LSTM baseline.
        
        Args:
            cfg: config dict (same interface as AstroStreamModel)
        """
        super().__init__()
        self.cfg = cfg
        
        self.d_model = cfg.d_model
        self.patch_len = cfg.patch_length_samples
        
        # Patcher
        self.patcher = BIOTPatcher(
            patch_len=self.patch_len,
            d_model=self.d_model,
            mask_token_init=0.0,
        )
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=2,
            batch_first=True,
            dropout=cfg.dropout,
        )
        
        # Linear decoder
        n_classes = cfg.decoder.n_speech_classes
        self.decoder = nn.Linear(self.d_model, n_classes, bias=True)

    def forward(
        self,
        x: torch.Tensor,  # (B, C, T)
        recording_ids: Optional[List[int]] = None,
        mask_for_jepa: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass (episodic)."""
        # Patcher
        patch_emb, _ = self.patcher(x, mask=False)  # (B, C*n_patches, D)
        
        # LSTM encoder
        h_encoded, (h_n, c_n) = self.lstm(patch_emb)  # (B, C*n_patches, D)
        
        # Decoder
        repr_pooled = h_encoded.mean(dim=1)  # (B, D)
        logits = self.decoder(repr_pooled)  # (B, n_classes)
        
        return {
            "logits": logits,
            "repr": repr_pooled,
            "gate_signal": None,
            "encoded": h_encoded,
        }

    def get_n_params(self) -> int:
        """Return trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(cfg: DictConfig, model_type: str = "astrostream") -> nn.Module:
    """
    Factory function to build model by type.
    
    Args:
        cfg: OmegaConf config
        model_type: "astrostream", "transformer", or "lstm"
        
    Returns:
        model instance
    """
    if model_type == "astrostream":
        return AstroStreamModel(cfg)
    elif model_type == "transformer":
        return TransformerBaselineModel(cfg)
    elif model_type == "lstm":
        return LSTMBaselineModel(cfg)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
