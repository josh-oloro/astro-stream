"""
Astrocyte-Inspired Gating Module for Neural Signal Modulation.

Reference: Letellier & Goda (2023). Astrocyte calcium signaling shifts the
polarity of presynaptic plasticity. Neuroscience, 525, 38–46.

Implements a biologically-inspired lateral gating mechanism that models
slow Ca²⁺-mediated modulation of synaptic gain at the tripartite synapse.
The gate operates in parallel with the Mamba encoder and modulates its output
based on a slowly-evolving state that tracks input mean activity.

Mathematical formulation:
  s_t = (1 - α) · s_{t-1} + α · mean(h_t, dim=-1, keepdim=True)
  g_t = sigmoid(w · relu(s_t - θ))
  output_t = g_t · h_t

where:
  - α ∈ (0, 1): decay constant (typically 0.02 for ~200ms timescale @ 200 Hz)
  - θ: learned Ca²⁺ threshold
  - w: learned gate weight
  - s_t: slow state scalar (B, 1, 1)
  - g_t: gate signal (B, 1, 1), broadcasts to (B, T, D)
  - h_t: fast input (B, T, D)

Key insight: The gate is LATERAL (not hierarchical), running in parallel
with the Mamba encoder and taking the same input. This implements the
"modulatory" aspect of astrocyte signaling: astrocytes don't compute,
they modulate.

Inputs:
  h_t: (B, T, D) fast encoder hidden states
  config: dict with keys {alpha, init_theta, init_w}

Outputs:
  - gated_h_t: (B, T, D) modulated hidden states
  - s_T: (B, 1) final slow state for inspection
"""

from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class AstrocyteGate(nn.Module):
    """
    Lateral astrocyte-inspired gating layer for slow modulation of neural activity.

    The gate models Ca²⁺-mediated gain modulation at the tripartite synapse.
    It maintains a slow state that tracks input activity and produces a
    multiplicative gating signal applied to the fast Mamba output.

    This is a LATERAL layer: it runs in parallel with the encoder,
    not on top of it.
    """

    def __init__(
        self,
        alpha: float = 0.02,
        init_theta: float = 0.0,
        init_w: float = 1.0,
    ) -> None:
        """
        Initialize astrocyte gate.

        Args:
            alpha: decay constant for slow state (default 0.02 ≈ 200ms @ 200 Hz)
                   Controls timescale of astrocyte Ca²⁺ dynamics
            init_theta: initial Ca²⁺ threshold θ (default 0.0, learned)
            init_w: initial gate weight w (default 1.0, learned)
        """
        super().__init__()

        # Decay constant: α ∈ (0, 1)
        # s_t = (1-α)·s_{t-1} + α·x_t
        # At 200 Hz: α=0.02 → timescale ≈ 1/α / sfreq ≈ 50 / 200 = 250ms
        # Actually: timescale ≈ -1 / ln(1-α) ≈ 1/α for small α
        # So τ ≈ 50 samples ≈ 250ms
        self.alpha = alpha  # Fixed (could make learnable)

        # Learned parameters: threshold θ and gate weight w
        # Both operate on the slow state s_t to compute gate: g_t = sigmoid(w·relu(s_t - θ))
        self.theta = nn.Parameter(torch.tensor(init_theta, dtype=torch.float32))  # scalar
        self.w = nn.Parameter(torch.tensor(init_w, dtype=torch.float32))  # scalar

        # Slow state s_t: initialized to 0, updated sequentially over time steps
        self.register_buffer("slow_state", torch.tensor(0.0, dtype=torch.float32))  # scalar

    def reset_state(self) -> None:
        """
        Reset slow state to zero.

        Called at recording/episode boundaries to avoid information leakage.
        """
        self.slow_state.fill_(0.0)

    def forward(self, h_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply astrocyte-inspired gating to fast neural activity.

        Processes the input sequentially over the time dimension,
        maintaining and updating the slow state at each step.

        Args:
            h_t: fast encoder hidden states, shape (B, T, D)
                 B = batch size
                 T = time steps
                 D = hidden dimension

        Returns:
            gated_h_t: modulated hidden states, shape (B, T, D)
            s_T: final slow state, shape (B, 1) for inspection/logging
        """
        batch_size, n_steps, hidden_dim = h_t.shape  # (B, T, D)

        # Device and dtype
        device = h_t.device
        dtype = h_t.dtype

        # Initialize outputs
        gated_outputs = []  # list of (B, D) tensors
        slow_states = []  # list of scalars for inspection

        # Ensure slow state is on correct device and dtype
        s_prev = self.slow_state.to(device).to(dtype)  # scalar

        # === Sequential processing over time dimension ===
        # This implements the recurrent dynamics: s_t depends on s_{t-1}
        for t in range(n_steps):
            # Extract current hidden state
            h_current = h_t[:, t, :]  # (B, D)

            # Step 1: Update slow state
            # s_t = (1 - α) · s_{t-1} + α · mean(h_t, dim=-1)
            # Compute mean across hidden dimension
            h_mean = torch.mean(h_current, dim=-1, keepdim=True)  # (B, 1)

            # Exponential moving average
            s_current = (1 - self.alpha) * s_prev + self.alpha * h_mean  # (B, 1)

            # Step 2: Compute gate value
            # g_t = sigmoid(w · relu(s_t - θ))
            s_threshold = torch.relu(s_current - self.theta)  # (B, 1), apply ReLU
            gate_logit = self.w * s_threshold  # (B, 1)
            g_t = torch.sigmoid(gate_logit)  # (B, 1), gate in [0, 1]

            # Step 3: Apply gate element-wise to fast stream
            # output_t = g_t · h_t
            # g_t: (B, 1) broadcasts to (B, D)
            gated_h_current = g_t * h_current  # (B, D)

            # Store outputs and state for inspection
            gated_outputs.append(gated_h_current)  # (B, D)
            slow_states.append(s_current)  # (B, 1)

            # Update state for next iteration
            s_prev = s_current  # (B, 1)

        # Stack outputs back into (B, T, D)
        gated_h_t = torch.stack(gated_outputs, dim=1)  # (B, T, D)

        # Final slow state for logging
        s_T = slow_states[-1]  # (B, 1)

        # Update the buffer with the final slow state (for reset_state on next forward)
        # Use squeeze to extract scalar mean for buffer update
        self.slow_state = s_T.mean(dim=0)  # scalar (average over batch)

        return gated_h_t, s_T  # (B, T, D), (B, 1)

    @property
    def gate_statistics(self) -> Dict[str, float]:
        """
        Compute statistics of gate values during forward pass.

        Returns:
            dict with keys:
                - 'slow_state_mean': mean slow state across batch
                - 'slow_state_std': std slow state across batch
                - 'theta': current threshold parameter
                - 'w': current gate weight parameter
        """
        # Slow state statistics
        slow_state_value = self.slow_state.item()  # scalar

        # Gate weight and threshold
        theta_value = self.theta.item()  # scalar
        w_value = self.w.item()  # scalar

        return {
            "slow_state": slow_state_value,
            "theta": theta_value,
            "w": w_value,
            "alpha": self.alpha,
        }

    def extra_repr(self) -> str:
        """Return extra representation for module."""
        return f"alpha={self.alpha}, init_theta={self.theta.item():.4f}, init_w={self.w.item():.4f}"


class AstrocyteGateLateral(nn.Module):
    """
    Wrapper module for applying astrocyte gating as a lateral layer.

    This module is designed to run in parallel with a Mamba encoder.
    Both take the same input, and the gate modulates the encoder output.

    Usage:
        encoder = MambaEncoder(...)
        gate = AstrocyteGateLateral(alpha=0.02, init_theta=0.0, init_w=1.0)

        # In forward pass:
        h_encoded = encoder(patches)  # (B, T, D)
        h_gated, s_T = gate(h_encoded)  # Apply gating
        # h_gated is the final modulated output
    """

    def __init__(
        self,
        alpha: float = 0.02,
        init_theta: float = 0.0,
        init_w: float = 1.0,
    ) -> None:
        """
        Initialize lateral astrocyte gate wrapper.

        Args:
            alpha: decay constant (default 0.02)
            init_theta: initial threshold (default 0.0)
            init_w: initial gate weight (default 1.0)
        """
        super().__init__()
        self.gate = AstrocyteGate(
            alpha=alpha, init_theta=init_theta, init_w=init_w
        )  # Core gate module

    def forward(self, h_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply gate to hidden states.

        Args:
            h_t: (B, T, D) hidden states

        Returns:
            gated_h_t: (B, T, D) modulated states
            s_T: (B, 1) final slow state
        """
        return self.gate(h_t)

    def reset_state(self) -> None:
        """Reset internal state."""
        self.gate.reset_state()

    def get_statistics(self) -> Dict[str, float]:
        """Get gate statistics for monitoring."""
        return self.gate.gate_statistics
