"""
Persistent State Manager for Streaming Neural Decoding.

Coordinates the lifecycle of hidden states across mini-batches during training.
Ensures that:

1. Within a recording session: hidden states are carried forward (never reset)
2. At recording boundaries: hidden states are reset to zeros
3. States are detached from computation graph (no backprop across boundaries)
4. Memory is managed efficiently (prune old recording states when not needed)

The key insight: recording_ids can be shuffled in mini-batches, so we must
track which recordings are "active" in the current batch and which are
starting fresh (segment_idx == 0).
"""

from typing import Dict, List, Set, Optional, Tuple
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PersistentStateManager:
    """
    Manages hidden state lifecycle across mini-batches for persistent-state training.

    Implements the critical invariant:
    - Within recording: state is carried forward
    - At boundaries: state is reset to zeros
    - States are detached from backprop graph
    """

    def __init__(
        self,
        encoder: nn.Module,
        gate: Optional[nn.Module] = None,
        max_recordings_in_memory: int = 1000,
    ) -> None:
        """
        Initialize persistent state manager.

        Args:
            encoder: PersistentMambaEncoder with per-recording state buffers
            gate: AstrocyteGate with resettable slow state (optional)
            max_recordings_in_memory: max recording IDs to track
        """
        self.encoder = encoder
        self.gate = gate
        self.max_recordings_in_memory = max_recordings_in_memory

        # Track active recording sessions
        self.active_recordings: Set[int] = set()
        self.recording_segment_map: Dict[int, int] = {}

        # Statistics for monitoring
        self.reset_count = 0
        self.carried_forward_count = 0

    def on_batch_start(
        self,
        recording_ids: List[int],
        segment_idxs: List[int],
    ) -> None:
        """
        Process batch start: reset states for new recordings.

        Determines which recordings are starting fresh (segment_idx == 0 or new recording)
        and resets their hidden states. States for continuing recordings are preserved.

        Args:
            recording_ids: list of recording IDs in batch, shape (batch_size,)
            segment_idxs: list of segment indices, shape (batch_size,)
        """
        if len(recording_ids) != len(segment_idxs):
            raise ValueError(
                f"recording_ids and segment_idxs must have same length, "
                f"got {len(recording_ids)} and {len(segment_idxs)}"
            )

        # Identify recordings that are starting fresh
        recordings_to_reset: Set[int] = set()

        for rec_id, seg_idx in zip(recording_ids, segment_idxs):
            rec_id_int = int(rec_id)
            seg_idx_int = int(seg_idx)

            # Determine if this recording is starting fresh
            is_new_recording = rec_id_int not in self.active_recordings
            is_first_segment = seg_idx_int == 0

            if is_new_recording or is_first_segment:
                recordings_to_reset.add(rec_id_int)
                self.reset_count += 1
            else:
                # Continuing recording: state is carried forward
                self.carried_forward_count += 1

            # Update tracking
            self.active_recordings.add(rec_id_int)
            self.recording_segment_map[rec_id_int] = seg_idx_int

        # Reset states for new/restarting recordings
        for rec_id in recordings_to_reset:
            self._reset_recording_state(rec_id)

        # Log state resets for debugging
        if recordings_to_reset:
            logger.debug(
                f"Reset states for {len(recordings_to_reset)} recording(s): {recordings_to_reset}"
            )

    def on_batch_end(
        self,
        recording_ids: List[int],
    ) -> None:
        """
        Process batch end: store and log state statistics.

        Args:
            recording_ids: list of recording IDs in batch
        """
        n_active = self.get_n_active_states()
        logger.debug(
            f"Batch end: {n_active} active recordings. "
            f"Resets: {self.reset_count}, Carried: {self.carried_forward_count}"
        )

        # Sanity check
        batch_recs = set(int(r) for r in recording_ids)
        if not batch_recs.issubset(self.active_recordings):
            missing = batch_recs - self.active_recordings
            logger.warning(f"Batch recordings not in active set: {missing}")

    def on_epoch_end(self) -> None:
        """Process epoch end: cleanup and memory management."""
        self._reset_all_states()

        logger.info(
            f"Epoch end: reset all states. "
            f"Resets: {self.reset_count}, Carried: {self.carried_forward_count}"
        )

        # Reset counters
        self.reset_count = 0
        self.carried_forward_count = 0

        # Clear active recordings
        self.active_recordings.clear()
        self.recording_segment_map.clear()

    def _reset_recording_state(self, recording_id: int) -> None:
        """Reset hidden state for a specific recording."""
        if hasattr(self.encoder, "reset_state_for_recording"):
            self.encoder.reset_state_for_recording(recording_id)
        elif hasattr(self.encoder, "reset_state"):
            self.encoder.reset_state()

        if self.gate is not None and hasattr(self.gate, "reset_state"):
            self.gate.reset_state()

    def _reset_all_states(self) -> None:
        """Reset all hidden states for all recordings."""
        if hasattr(self.encoder, "reset_all_states"):
            self.encoder.reset_all_states()
        elif hasattr(self.encoder, "reset_state"):
            self.encoder.reset_state()

        if self.gate is not None and hasattr(self.gate, "reset_state"):
            self.gate.reset_state()

    def get_n_active_states(self) -> int:
        """Return number of recording IDs currently tracked in memory."""
        return len(self.active_recordings)

    def get_state_summary(self) -> Dict[str, int]:
        """Return summary of state tracking statistics."""
        return {
            "n_active_recordings": self.get_n_active_states(),
            "total_resets": self.reset_count,
            "total_carried_forward": self.carried_forward_count,
        }


class EncoderWithPersistentState(nn.Module):
    """Wrapper that adds persistent state management to an encoder."""

    def __init__(self, encoder: nn.Module) -> None:
        """Initialize state-enhanced encoder."""
        super().__init__()
        self.encoder = encoder
        self.hidden_states: Dict[int, torch.Tensor] = {}

    def set_hidden_state(self, recording_id: int, h_t: torch.Tensor) -> None:
        """Store hidden state for a recording (detached from graph)."""
        self.hidden_states[recording_id] = h_t.detach()

    def get_hidden_state(self, recording_id: int) -> Optional[torch.Tensor]:
        """Retrieve hidden state for a recording."""
        return self.hidden_states.get(recording_id, None)

    def reset_state_for_recording(self, recording_id: int) -> None:
        """Reset hidden state for a specific recording."""
        if recording_id in self.hidden_states:
            del self.hidden_states[recording_id]

    def reset_all_states(self) -> None:
        """Reset all hidden states for all recordings."""
        self.hidden_states.clear()

    def forward(self, x: torch.Tensor, recording_ids: List[int]) -> torch.Tensor:
        """Forward pass with state management."""
        return self.encoder(x, recording_ids=recording_ids)


class GateWithPersistentState(nn.Module):
    """Wrapper that adds persistent state to astrocyte gate."""

    def __init__(self, gate: nn.Module) -> None:
        """Initialize state-enhanced gate."""
        super().__init__()
        self.gate = gate
        self.slow_states: Dict[int, torch.Tensor] = {}

    def set_slow_state(self, recording_id: int, s_t: torch.Tensor) -> None:
        """Store slow state for a recording."""
        self.slow_states[recording_id] = s_t.detach()

    def get_slow_state(self, recording_id: int) -> Optional[torch.Tensor]:
        """Retrieve slow state for a recording."""
        return self.slow_states.get(recording_id, None)

    def reset_state_for_recording(self, recording_id: int) -> None:
        """Reset slow state for a recording."""
        if recording_id in self.slow_states:
            del self.slow_states[recording_id]

    def reset_all_states(self) -> None:
        """Reset all slow states."""
        self.slow_states.clear()

    def forward(self, h_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass (delegates to wrapped gate)."""
        return self.gate(h_t)
