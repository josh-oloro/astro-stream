"""
AstroStreamTrainer: PyTorch Lightning training orchestration.

Implements the full training loop for Astro-Stream:
1. Forward pass with masking + persistent state management
2. Multi-objective loss: JEPA + speech decoding + closure regularization
3. EMA target encoder updates
4. Validation with closure metric
5. Checkpoint management + W&B logging

Key features:
- Persistent state across batches (no reset within recording)
- Self-supervised JEPA objective + supervised decoding
- Biologically-inspired closure metric (latent predictability)
- Multi-GPU support via PyTorch Lightning
- Automatic mixed precision (AMP) with gradient scaling
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from astrostream.models.astrostream import AstroStreamModel, build_model
from astrostream.models.objectives.jepa import (
    JEPAObjective,
    EMATargetEncoder,
    JEPAPredictor,
    JEPALoss,
)
from astrostream.training.persistent_state import PersistentStateManager
from astrostream.data.datamodule import AstroStreamDataModule

logger = logging.getLogger(__name__)


class AstroStreamLightningModule(L.LightningModule):
    """
    PyTorch Lightning module for Astro-Stream training.

    Orchestrates:
    - Multi-objective training (JEPA + speech decoding + closure)
    - Persistent state management across batches
    - EMA target encoder updates
    - Validation with closure metric
    """

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initialize lightning module.

        Args:
            cfg: OmegaConf config with model, training, closure, jepa settings
        """
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))

        # === Component 1: Main model ===
        self.model = AstroStreamModel(cfg)
        logger.info(
            f"Built AstroStreamModel with {self.model.get_n_params():,} parameters"
        )

        # === Component 2: JEPA components ===
        if cfg.jepa.enabled:
            # EMA target encoder (deep copy of main encoder)
            self.ema_encoder = EMATargetEncoder(
                self.model.encoder,
                momentum_start=cfg.jepa.ema_momentum_start,
                momentum_end=cfg.jepa.ema_momentum_end,
            )

            # Predictor network
            self.jepa_predictor = JEPAPredictor(
                d_model=cfg.d_model,
                n_heads=cfg.jepa.get("n_heads", 8),
                n_layers=cfg.jepa.get("n_layers", 2),
                dropout=cfg.dropout,
            )

            # JEPA loss
            self.jepa_loss_fn = JEPALoss()
        else:
            self.ema_encoder = None
            self.jepa_predictor = None
            self.jepa_loss_fn = None

        # === Component 3: Persistent state manager ===
        gate = self.model.gate if not isinstance(self.model.gate, nn.Identity) else None
        self.state_manager = PersistentStateManager(
            self.model.encoder,
            gate=gate,
        )

        # === Component 4: Closure metric ===
        if cfg.closure.enabled:
            try:
                from astrostream.evaluation.closure_metric import NTICClosureMetric

                self.closure_metric = NTICClosureMetric(
                    d_model=cfg.d_model,
                    k=cfg.closure.get("k", 10),
                )
            except ImportError:
                logger.warning("NTICClosureMetric not available, disabling closure")
                self.closure_metric = None
        else:
            self.closure_metric = None

        # === Component 5: Training state ===
        self.train_loss_buffer = {"jepa": [], "decoder": [], "closure": []}
        self.val_loss_buffer = {"acc_top1": [], "acc_top10": [], "closure": []}

    def forward(
        self,
        x: torch.Tensor,
        recording_ids: Optional[list] = None,
        mask_for_jepa: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through model."""
        return self.model(x, recording_ids=recording_ids, mask_for_jepa=mask_for_jepa)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step: forward pass + loss computation.

        Args:
            batch: dict with keys 'meg', 'label', 'recording_id', 'segment_idx'
            batch_idx: batch index

        Returns:
            total_loss: scalar tensor for backprop
        """
        # Step 1: Manage persistent state
        self.state_manager.on_batch_start(
            recording_ids=batch["recording_id"].tolist(),
            segment_idxs=batch["segment_idx"].tolist(),
        )

        # Step 2: Forward pass with masking
        outputs = self.forward(
            x=batch["meg"],
            recording_ids=batch["recording_id"].tolist(),
            mask_for_jepa=self.cfg.jepa.enabled,
        )

        # Step 3: Compute losses

        # --- Speech decoding loss ---
        logits = outputs["logits"]  # (B, n_classes)
        labels = batch["label"].long()  # (B,)
        decoder_loss = F.cross_entropy(logits, labels)

        # --- JEPA loss ---
        jepa_loss = torch.tensor(0.0, device=self.device)
        if self.cfg.jepa.enabled and outputs["masked_repr"] is not None:
            with torch.no_grad():
                target_repr = self.ema_encoder(batch["meg"])
                target_masked = target_repr[:, outputs["mask_indices"].bool(), :]

            pred_repr = self.jepa_predictor(
                context=outputs["masked_repr"],
                mask_indices=outputs["mask_indices"],
            )

            jepa_loss = self.jepa_loss_fn(pred_repr, target_masked)

            # Update EMA encoder
            self.ema_encoder.update(self.global_step, self.trainer.estimated_stepping_batches)

        # --- Closure regularization ---
        closure_loss = torch.tensor(0.0, device=self.device)
        if self.cfg.closure.enabled and self.closure_metric is not None:
            repr_batch = outputs["repr"]  # (B, D)
            closure_score = self.closure_metric(repr_batch)
            closure_loss = self.cfg.closure.weight * (1.0 - closure_score)

        # Step 4: Compute total loss
        total_loss = (
            decoder_loss
            + self.cfg.jepa.weight * jepa_loss
            + closure_loss
        )

        # Step 5: State management at batch end
        self.state_manager.on_batch_end(batch["recording_id"].tolist())

        # Step 6: Logging
        self.log_dict(
            {
                "train/decoder_loss": decoder_loss,
                "train/jepa_loss": jepa_loss,
                "train/closure_loss": closure_loss,
                "train/total_loss": total_loss,
                "train/learning_rate": self.trainer.optimizers[0].param_groups[0]["lr"],
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,  # Multi-GPU
        )

        return total_loss

    def on_train_epoch_end(self) -> None:
        """Called at end of training epoch."""
        self.state_manager.on_epoch_end()

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Validation step: compute accuracy and closure metrics.

        Args:
            batch: dict with keys 'meg', 'label', 'recording_id', 'segment_idx'
            batch_idx: batch index

        Returns:
            dict with metrics
        """
        # Reset state for validation
        self.state_manager.on_batch_start(
            recording_ids=batch["recording_id"].tolist(),
            segment_idxs=[0] * len(batch["recording_id"]),  # Always reset for val
        )

        # Forward pass (no masking for validation)
        with torch.no_grad():
            outputs = self.forward(
                x=batch["meg"],
                recording_ids=batch["recording_id"].tolist(),
                mask_for_jepa=False,
            )

        # Compute metrics
        logits = outputs["logits"]  # (B, n_classes)
        labels = batch["label"].long()  # (B,)

        # Top-1 accuracy
        preds_top1 = logits.argmax(dim=-1)
        acc_top1 = (preds_top1 == labels).float().mean()

        # Top-10 accuracy
        top10_preds = torch.topk(logits, k=min(10, logits.shape[1]), dim=-1)[1]
        acc_top10 = (top10_preds == labels.unsqueeze(1)).any(dim=1).float().mean()

        # Closure metric
        closure_score = torch.tensor(0.0, device=self.device)
        if self.cfg.closure.enabled and self.closure_metric is not None:
            repr_batch = outputs["repr"]  # (B, D)
            closure_score = self.closure_metric(repr_batch)

        return {
            "val_acc_top1": acc_top1,
            "val_acc_top10": acc_top10,
            "val_closure": closure_score,
        }

    def on_validation_epoch_end(self) -> None:
        """Called at end of validation epoch."""
        # Average metrics across all validation batches
        if self.val_loss_buffer["acc_top1"]:
            avg_top1 = torch.stack(self.val_loss_buffer["acc_top1"]).mean()
            avg_top10 = torch.stack(self.val_loss_buffer["acc_top10"]).mean()
            avg_closure = torch.stack(self.val_loss_buffer["closure"]).mean()

            self.log_dict(
                {
                    "val/acc_top1": avg_top1,
                    "val/acc_top10": avg_top10,
                    "val/closure": avg_closure,
                },
                sync_dist=True,
            )

            # Clear buffers
            self.val_loss_buffer = {"acc_top1": [], "acc_top10": [], "closure": []}

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and learning rate scheduler."""
        # Separate parameters: gate parameters exclude weight decay
        gate_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "gate" in name and any(x in name for x in ["alpha", "w", "theta"]):
                    gate_params.append(param)
                else:
                    other_params.append(param)

        # Add JEPA parameters if enabled
        if self.cfg.jepa.enabled:
            for param in self.jepa_predictor.parameters():
                if param.requires_grad:
                    other_params.append(param)

        # Optimizer with different weight decay for gate vs. other params
        optimizer = AdamW(
            [
                {"params": other_params, "weight_decay": self.cfg.optimizer.weight_decay},
                {"params": gate_params, "weight_decay": 0.0},  # No weight decay for gate
            ],
            lr=self.cfg.optimizer.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Cosine annealing with warmup
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.trainer.estimated_stepping_batches // self.cfg.optimizer.epochs,
            T_mult=1,
            eta_min=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Called when saving checkpoint."""
        checkpoint["state_manager_stats"] = self.state_manager.get_state_summary()

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Called when loading checkpoint."""
        logger.info(f"Loaded checkpoint stats: {checkpoint.get('state_manager_stats')}")


def train(cfg: DictConfig) -> None:
    """
    Main training function.

    Sets up data, model, trainer, and runs training.

    Args:
        cfg: OmegaConf config from Hydra
    """
    # === Setup logging ===
    logging.basicConfig(level=logging.INFO)
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # === Setup W&B ===
    if HAS_WANDB and cfg.get("wandb", {}).get("enabled", False):
        wandb_config = cfg.get("wandb", {})
        wandb.init(
            project=wandb_config.get("project", "astro-stream"),
            entity=wandb_config.get("entity", None),
            name=wandb_config.get("run_name", None),
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=wandb_config.get("tags", []),
        )
        logger.info(f"W&B run: {wandb.run.url}")

    # === Setup data ===
    logger.info("Building data module...")
    data_module = AstroStreamDataModule(cfg)
    data_module.setup()

    # === Setup model ===
    logger.info("Building lightning module...")
    lightning_module = AstroStreamLightningModule(cfg)

    # === Setup trainer ===
    logger.info("Building trainer...")
    trainer_cfg = cfg.get("trainer", {})

    trainer = L.Trainer(
        max_epochs=cfg.optimizer.get("epochs", 100),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=trainer_cfg.get("num_gpus", 1),
        strategy="ddp" if trainer_cfg.get("num_gpus", 1) > 1 else "auto",
        precision=trainer_cfg.get("precision", "32"),
        gradient_clip_val=trainer_cfg.get("gradient_clip_val", 1.0),
        accumulate_grad_batches=trainer_cfg.get("accumulate_grad_batches", 1),
        log_every_n_steps=trainer_cfg.get("log_every_n_steps", 50),
        val_check_interval=trainer_cfg.get("val_check_interval", 0.5),
        enable_checkpointing=True,
        default_root_dir=Path(cfg.get("ckpt_dir", "./checkpoints")),
        logger=L.pytorch_lightning.loggers.WandbLogger() if HAS_WANDB and cfg.get("wandb", {}).get("enabled") else None,
    )

    # === Run training ===
    logger.info("Starting training...")
    trainer.fit(
        lightning_module,
        train_dataloaders=data_module.train_dataloader(),
        val_dataloaders=data_module.val_dataloader(),
    )

    # === Finish ===
    if HAS_WANDB and cfg.get("wandb", {}).get("enabled", False):
        wandb.finish()

    logger.info("Training complete!")
