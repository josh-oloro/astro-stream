"""
PyTorch Dataset and DataModule for MEG-speech decoding.

Implements a dataset class that loads preprocessed MEG epochs from HDF5 files
and aligns them with Wav2Vec2 speech embeddings. Supports:

- Multi-subject data loading with per-subject HDF5 caching
- Automatic alignment between MEG epochs and speech embeddings
- Subject-level train/val/test splitting
- Batch collation with recording-aware sorting for persistent-state training
- PyTorch Lightning DataModule interface

Inputs:
  - h5_epoch_files: list of Path objects pointing to subject-level epoch .h5 files
  - embeddings_dir: directory containing {subject_id}_wav2vec2_embeddings.npy files
  - config: OmegaConf config with keys {sampling_rate_target, epoch_tmin, epoch_tmax,
            split: {train, val, test}, mask_rate}

Outputs:
  - Dataset: returns dicts with keys {meg, wav2vec, subject_id, recording_id, segment_idx, word_text}
  - DataLoaders: batches sorted by (recording_id, segment_idx) for persistent-state training
"""

from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
from omegaconf import DictConfig


class MEGSpeechDataset(Dataset):
    """
    PyTorch Dataset for MEG-speech alignment.

    Loads preprocessed MEG epochs (HDF5) and aligns with Wav2Vec2 embeddings.
    Supports multi-subject data with subject-level train/val/test splitting.
    """

    def __init__(
        self,
        h5_epoch_files: List[Path],
        embeddings_dir: Path,
        config: Dict[str, Any],
        subject_split: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize MEG-speech dataset.

        Args:
            h5_epoch_files: list of Path objects to HDF5 epoch files (one per subject)
            embeddings_dir: directory containing *_wav2vec2_embeddings.npy files
            config: dict with keys {sampling_rate_target, epoch_tmin, epoch_tmax,
                    split: {train, val, test}, mask_rate}
            subject_split: optional list of subject IDs to include (for train/val/test splits)
        """
        self.h5_epoch_files = [Path(f) for f in h5_epoch_files]
        self.embeddings_dir = Path(embeddings_dir)
        self.config = config
        self.subject_split = subject_split

        self.sfreq_target = config["sampling_rate_target"]  # Hz
        self.epoch_tmin = config["epoch_tmin"]  # s
        self.epoch_tmax = config["epoch_tmax"]  # s
        self.n_samples_per_epoch = int(
            (self.epoch_tmax - self.epoch_tmin) * self.sfreq_target
        )  # samples

        # Build index: list of (h5_file_path, subject_id, epoch_idx, embeddings_array, word_text)
        self.index = []
        self._build_index()

    def _build_index(self) -> None:
        """
        Build mapping from dataset indices to (subject, epoch_idx, embedding).

        Reads metadata from HDF5 files and cross-references with Wav2Vec2 embeddings.
        """
        for h5_file in self.h5_epoch_files:
            if not h5_file.exists():
                print(f"Warning: {h5_file} not found, skipping.")
                continue

            with h5py.File(h5_file, "r") as f:
                subject_id = f.attrs["subject_id"]

                # Filter by subject_split if provided
                if self.subject_split is not None and subject_id not in self.subject_split:
                    continue

                n_epochs = f.attrs["n_epochs"]  # (n_epochs, n_channels, n_samples)

                # Load embeddings for this subject
                embeddings_path = (
                    self.embeddings_dir / f"{subject_id}_wav2vec2_embeddings.npy"
                )
                if not embeddings_path.exists():
                    print(f"Warning: {embeddings_path} not found, skipping subject {subject_id}.")
                    continue

                embeddings = np.load(embeddings_path)  # (T, 1024)

                # For each epoch, find the most recent embedding frame
                # Assume epochs are time-aligned with embeddings
                for epoch_idx in range(n_epochs):
                    # Align epoch to embedding: epoch center time → embedding frame
                    epoch_center_s = self.epoch_tmin + (self.epoch_tmax - self.epoch_tmin) / 2
                    embedding_frame_idx = int(epoch_center_s * 50)  # 50 ms frame rate

                    if 0 <= embedding_frame_idx < len(embeddings):
                        # Simplified: use word text as placeholder (should come from BIDS events)
                        word_text = "word"
                        recording_id = 0  # placeholder: would come from session metadata
                        segment_idx = epoch_idx

                        self.index.append(
                            {
                                "h5_file": h5_file,
                                "subject_id": subject_id,
                                "epoch_idx": epoch_idx,
                                "embedding": embeddings[embedding_frame_idx],  # (1024,)
                                "recording_id": recording_id,
                                "segment_idx": segment_idx,
                                "word_text": word_text,
                            }
                        )

    def __len__(self) -> int:
        """Return total number of valid (epoch, embedding) pairs."""
        return len(self.index)  # int

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single (MEG epoch, speech embedding) pair.

        Args:
            idx: index into the dataset

        Returns:
            dict with keys:
                - 'meg': FloatTensor of shape (n_channels, n_samples)
                - 'wav2vec': FloatTensor of shape (embedding_dim,), typically (1024,)
                - 'subject_id': str, subject identifier
                - 'recording_id': int, continuous recording session ID
                - 'segment_idx': int, position within recording
                - 'word_text': str, spoken word
        """
        item = self.index[idx]

        h5_file = item["h5_file"]
        epoch_idx = item["epoch_idx"]
        subject_id = item["subject_id"]
        recording_id = item["recording_id"]
        segment_idx = item["segment_idx"]
        word_text = item["word_text"]
        embedding = item["embedding"]  # (1024,)

        # Load MEG epoch from HDF5
        with h5py.File(h5_file, "r") as f:
            meg_epoch = f["epochs"][epoch_idx]  # (n_channels, n_samples)
            meg_epoch = torch.from_numpy(meg_epoch).float()  # (n_channels, n_samples)

        # Convert embedding to tensor
        wav2vec_embedding = torch.from_numpy(embedding).float()  # (1024,)

        return {
            "meg": meg_epoch,  # (n_channels, n_samples)
            "wav2vec": wav2vec_embedding,  # (1024,)
            "subject_id": subject_id,  # str
            "recording_id": recording_id,  # int
            "segment_idx": segment_idx,  # int
            "word_text": word_text,  # str
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples.

        Pads/truncates MEG epochs to a common length and sorts by (recording_id, segment_idx)
        for correct persistent-state training (consecutive segments adjacent in batch).

        Args:
            batch: list of dicts from __getitem__

        Returns:
            dict with batched tensors:
                - 'meg': FloatTensor of shape (batch_size, n_channels, max_n_samples)
                - 'wav2vec': FloatTensor of shape (batch_size, 1024)
                - 'subject_id': list of str
                - 'recording_id': LongTensor of shape (batch_size,)
                - 'segment_idx': LongTensor of shape (batch_size,)
                - 'word_text': list of str
                - 'seq_len': LongTensor of shape (batch_size,), true length before padding
        """
        # Sort by (recording_id, segment_idx) for persistent-state contiguity
        batch_sorted = sorted(
            batch, key=lambda x: (x["recording_id"], x["segment_idx"])
        )  # list[(dict)]

        # Extract components
        meg_list = [item["meg"] for item in batch_sorted]  # list[(n_channels, n_samples)]
        wav2vec_list = [item["wav2vec"] for item in batch_sorted]  # list[(1024,)]
        subject_ids = [item["subject_id"] for item in batch_sorted]  # list[str]
        recording_ids = torch.LongTensor(
            [item["recording_id"] for item in batch_sorted]
        )  # (batch_size,)
        segment_idxs = torch.LongTensor(
            [item["segment_idx"] for item in batch_sorted]
        )  # (batch_size,)
        word_texts = [item["word_text"] for item in batch_sorted]  # list[str]

        # Pad/truncate MEG epochs to max length
        max_n_samples = max(meg.shape[1] for meg in meg_list)  # int
        batch_size = len(meg_list)
        n_channels = meg_list[0].shape[0]  # int

        meg_padded = torch.zeros(
            batch_size, n_channels, max_n_samples, dtype=torch.float32
        )  # (batch_size, n_channels, max_n_samples)
        seq_len = torch.zeros(batch_size, dtype=torch.long)  # (batch_size,)

        for i, meg in enumerate(meg_list):
            n_samp = meg.shape[1]
            meg_padded[i, :, :n_samp] = meg  # pad with zeros (implicit)
            seq_len[i] = n_samp

        # Stack Wav2Vec embeddings
        wav2vec_batch = torch.stack(wav2vec_list, dim=0)  # (batch_size, 1024)

        return {
            "meg": meg_padded,  # (batch_size, n_channels, max_n_samples)
            "wav2vec": wav2vec_batch,  # (batch_size, 1024)
            "subject_id": subject_ids,  # list[str]
            "recording_id": recording_ids,  # (batch_size,)
            "segment_idx": segment_idxs,  # (batch_size,)
            "word_text": word_texts,  # list[str]
            "seq_len": seq_len,  # (batch_size,), true lengths before padding
        }

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "MEGSpeechDataset":
        """
        Construct dataset from OmegaConf config.

        Args:
            cfg: OmegaConf config with keys:
                - data: {gwilliams2022 or schoffelen2019}
                - cache_dir: base cache directory
                - embeddings_dir: directory with Wav2Vec2 .npy files

        Returns:
            MEGSpeechDataset instance
        """
        cache_dir = Path(cfg.cache_dir)
        embeddings_dir = Path(cfg.embeddings_dir)

        # Find all HDF5 epoch files
        h5_files = sorted(cache_dir.glob("sub-*_epochs.h5"))

        # Extract config dict from cfg.data
        config_dict = {
            "sampling_rate_target": cfg.sampling_rate_target,
            "epoch_tmin": cfg.epoch_tmin,
            "epoch_tmax": cfg.epoch_tmax,
            "split": cfg.split,
            "mask_rate": cfg.mask_rate,
        }

        return cls(
            h5_epoch_files=h5_files,
            embeddings_dir=embeddings_dir,
            config=config_dict,
        )

    def split_by_subject(
        self, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1
    ) -> Tuple["MEGSpeechDataset", "MEGSpeechDataset", "MEGSpeechDataset"]:
        """
        Split dataset into train/val/test by subject.

        Ensures no subject appears in multiple splits.

        Args:
            train_ratio: fraction of subjects for training
            val_ratio: fraction of subjects for validation
            test_ratio: fraction of subjects for testing

        Returns:
            (train_dataset, val_dataset, test_dataset)
        """
        # Get unique subjects
        unique_subjects = sorted(set(item["subject_id"] for item in self.index))
        n_subjects = len(unique_subjects)

        # Compute split counts
        n_train = int(n_subjects * train_ratio)
        n_val = int(n_subjects * val_ratio)
        n_test = n_subjects - n_train - n_val

        # Split subjects
        train_subjects = unique_subjects[:n_train]
        val_subjects = unique_subjects[n_train : n_train + n_val]
        test_subjects = unique_subjects[n_train + n_val :]

        # Create subset datasets
        train_dataset = MEGSpeechDataset(
            h5_epoch_files=self.h5_epoch_files,
            embeddings_dir=self.embeddings_dir,
            config=self.config,
            subject_split=train_subjects,
        )
        val_dataset = MEGSpeechDataset(
            h5_epoch_files=self.h5_epoch_files,
            embeddings_dir=self.embeddings_dir,
            config=self.config,
            subject_split=val_subjects,
        )
        test_dataset = MEGSpeechDataset(
            h5_epoch_files=self.h5_epoch_files,
            embeddings_dir=self.embeddings_dir,
            config=self.config,
            subject_split=test_subjects,
        )

        return train_dataset, val_dataset, test_dataset


class MEGSpeechDataModule:
    """
    PyTorch DataModule wrapper for MEG-speech dataset.

    Compatible with PyTorch Lightning training loops.
    """

    def __init__(
        self,
        dataset: MEGSpeechDataset,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
    ) -> None:
        """
        Initialize DataModule.

        Args:
            dataset: MEGSpeechDataset instance
            batch_size: number of samples per batch
            num_workers: number of data loader workers
            pin_memory: whether to pin memory for GPU transfer
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Split dataset by subject
        self.train_dataset, self.val_dataset, self.test_dataset = (
            dataset.split_by_subject(
                train_ratio=dataset.config["split"]["train"],
                val_ratio=dataset.config["split"]["val"],
                test_ratio=dataset.config["split"]["test"],
            )
        )

    def train_dataloader(self) -> DataLoader:
        """Return training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=MEGSpeechDataset.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=MEGSpeechDataset.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test DataLoader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=MEGSpeechDataset.collate_fn,
        )
