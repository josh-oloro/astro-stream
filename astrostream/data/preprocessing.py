"""
MNE-based preprocessing module for MEG/EEG data.

This module implements preprocessing pipelines for converting raw BIDS MEG/EEG data
to epoched segments suitable for neural decoding experiments. Supports:

- Loading raw data (FIF, CTF .ds formats)
- Bandpass filtering and resampling
- Event-based epoching with rejection
- Fast HDF5 caching for repeated experiments
- Wav2Vec2 audio embedding extraction for speech alignment

Implements methods from MNE-Python and follows BIDS conventions.

Inputs:
  - config: dict with keys {bandpass_low, bandpass_high, sampling_rate_target,
            epoch_tmin, epoch_tmax, reject_threshold}
  - raw_path: Path to BIDS MEG/EEG raw file (.fif or .ds)
  - audio_path: Path to audio file (.wav, .mp3)

Outputs:
  - mne.Epochs: preprocessed epoched data
  - .h5 files: cached epochs with metadata
  - .npy files: Wav2Vec2 embeddings (T, 1024)
"""

import warnings
from pathlib import Path
from typing import Dict, Any

import h5py
import mne
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
import soundfile as sf

# Suppress MNE verbose output
mne.set_log_level("WARNING")


class MNEPreprocessor:
    """Wraps MNE-Python for preprocessing MEG/EEG data to epochs with caching."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize preprocessor with configuration.

        Args:
            config: dict with keys:
                - bandpass_low (float): Hz, high-pass cutoff
                - bandpass_high (float): Hz, low-pass cutoff
                - sampling_rate_raw (int): original sampling rate
                - sampling_rate_target (int): target sampling rate after downsampling
                - epoch_tmin (float): seconds relative to event onset
                - epoch_tmax (float): seconds relative to event onset
                - reject_threshold (float): peak-to-peak amplitude threshold for rejection
        """
        self.config = config
        self.bandpass_low = config["bandpass_low"]  # Hz
        self.bandpass_high = config["bandpass_high"]  # Hz
        self.sfreq_raw = config["sampling_rate_raw"]  # Hz
        self.sfreq_target = config["sampling_rate_target"]  # Hz
        self.tmin = config["epoch_tmin"]  # s
        self.tmax = config["epoch_tmax"]  # s
        self.reject_threshold = config["reject_threshold"]  # V or T

    def preprocess_subject(self, raw_path: Path) -> mne.Epochs:
        """
        Load, filter, resample, epoch, and reject a raw MEG/EEG file.

        Args:
            raw_path: Path to raw FIF or CTF .ds file

        Returns:
            mne.Epochs: cleaned epochs with shape (n_epochs, n_channels, n_samples)
                where n_samples = int((epoch_tmax - epoch_tmin) * sfreq_target)
        """
        raw_path = Path(raw_path)

        # Load raw data (auto-detect format: .fif, .ds, etc.)
        if raw_path.suffix == ".fif":
            raw = mne.io.read_raw_fif(str(raw_path), preload=True)  # (n_channels, n_samples)
        elif raw_path.suffix == ".ds":
            raw = mne.io.read_raw_ctf(str(raw_path), preload=True)  # (n_channels, n_samples)
        else:
            raise ValueError(f"Unsupported format: {raw_path.suffix}")

        # Apply bandpass filter IIR (causal, low-latency)
        raw.filter(
            l_freq=self.bandpass_low,
            h_freq=self.bandpass_high,
            method="iir",
            phase="forward",  # causal
        )  # (n_channels, n_samples)

        # Resample to target sampling rate
        if raw.info["sfreq"] != self.sfreq_target:
            raw.resample(self.sfreq_target)  # (n_channels, n_samples_resampled)

        # Read events from *_events.tsv BIDS sidecar
        events_tsv_path = raw_path.parent / raw_path.name.replace(".fif", "_events.tsv").replace(
            ".ds", "_events.tsv"
        )

        if not events_tsv_path.exists():
            raise FileNotFoundError(f"Events file not found: {events_tsv_path}")

        # Parse BIDS events file (tab-separated: onset, duration, trial_type)
        events = np.loadtxt(events_tsv_path, skiprows=1, usecols=(0,))  # (n_events,), in seconds
        onset_samples = (events * self.sfreq_target).astype(int)  # convert to samples

        # Create events array for mne.Epochs: (n_events, 3) where col 2 is event ID
        event_id = 1
        mne_events = np.column_stack([onset_samples, np.zeros_like(onset_samples), event_id])

        # Epoch around events
        epochs = mne.Epochs(
            raw,
            mne_events,
            event_id=event_id,
            tmin=self.tmin,
            tmax=self.tmax,
            baseline=None,  # no baseline correction
            preload=True,
        )  # (n_epochs, n_channels, n_samples)

        # Reject epochs exceeding peak-to-peak amplitude threshold
        # Compute peak-to-peak for each epoch and channel
        data = epochs.get_data()  # (n_epochs, n_channels, n_samples)
        pp_amplitude = np.ptp(data, axis=2)  # (n_epochs, n_channels), peak-to-peak
        max_pp = np.max(pp_amplitude, axis=1)  # (n_epochs,), max across channels
        valid_mask = max_pp < self.reject_threshold  # (n_epochs,), boolean

        # Drop bad epochs
        bad_indices = np.where(~valid_mask)[0]
        epochs.drop(indices=bad_indices)  # in-place

        return epochs  # (n_valid_epochs, n_channels, n_samples)

    def cache_subject(self, subject_id: str, epochs: mne.Epochs, cache_dir: Path) -> Path:
        """
        Save epochs to HDF5 file with metadata.

        Args:
            subject_id: unique identifier (e.g., "sub-01")
            epochs: mne.Epochs object
            cache_dir: directory to save .h5 file

        Returns:
            Path to saved .h5 file
        """
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        h5_path = cache_dir / f"{subject_id}_epochs.h5"

        data = epochs.get_data()  # (n_epochs, n_channels, n_samples)
        n_epochs, n_channels, n_samples = data.shape

        with h5py.File(h5_path, "w") as f:
            # Store data as HDF5 dataset (contiguous, row-major for fast row access)
            f.create_dataset("epochs", data=data, dtype="float32", compression="gzip")

            # Store metadata as attributes
            f.attrs["subject_id"] = subject_id
            f.attrs["n_epochs"] = n_epochs
            f.attrs["n_channels"] = n_channels
            f.attrs["n_samples"] = n_samples
            f.attrs["sfreq"] = epochs.info["sfreq"]
            f.attrs["tmin"] = self.tmin
            f.attrs["tmax"] = self.tmax

            # Store channel names as fixed-length string array
            channel_names = np.array(epochs.ch_names, dtype="S50")
            f.create_dataset("channel_names", data=channel_names)

        return h5_path

    def load_or_preprocess(
        self, subject_id: str, raw_path: Path, cache_dir: Path
    ) -> h5py.File:
        """
        Load cached epochs or preprocess and cache on first call.

        Args:
            subject_id: unique identifier
            raw_path: path to raw BIDS file
            cache_dir: directory for cached .h5 files

        Returns:
            h5py.File opened in read mode with keys {"epochs", "channel_names"}
                and attributes {"subject_id", "n_epochs", "n_channels", "n_samples",
                "sfreq", "tmin", "tmax"}
        """
        cache_dir = Path(cache_dir)
        h5_path = cache_dir / f"{subject_id}_epochs.h5"

        # Check if cached version exists
        if h5_path.exists():
            return h5py.File(h5_path, "r")

        # Preprocess and cache
        epochs = self.preprocess_subject(raw_path)  # (n_epochs, n_channels, n_samples)
        self.cache_subject(subject_id, epochs, cache_dir)

        return h5py.File(h5_path, "r")


def extract_wav2vec_embeddings(
    audio_path: Path, model_name: str = "facebook/wav2vec2-large-960h", device: str = "cuda"
) -> np.ndarray:
    """
    Extract Wav2Vec2 frame-level embeddings from audio file.

    Processes audio in 30-second chunks to avoid GPU OOM.

    Args:
        audio_path: path to .wav/.mp3 audio file
        model_name: HuggingFace model ID (default: English Wav2Vec2-large)
        device: "cuda" or "cpu"

    Returns:
        embeddings: (T, 1024) float32 array of frame-level embeddings
            where T is number of frames at 50ms resolution (~20 fps)
    """
    audio_path = Path(audio_path)

    # Check for cached embeddings
    npy_path = audio_path.parent / f"{audio_path.stem}_wav2vec2_embeddings.npy"
    if npy_path.exists():
        return np.load(npy_path)

    # Load model and processor
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    # Load audio
    audio, sr = librosa.load(str(audio_path), sr=processor.feature_extractor.sampling_rate)
    duration_sec = len(audio) / sr

    # Process in 30-second chunks to avoid OOM
    chunk_duration_sec = 30.0
    chunk_samples = int(chunk_duration_sec * sr)
    n_frames_per_chunk = int(chunk_duration_sec * 50)  # 50 ms frame rate

    all_embeddings = []

    for start_idx in range(0, len(audio), chunk_samples):
        end_idx = min(start_idx + chunk_samples, len(audio))
        audio_chunk = audio[start_idx:end_idx]  # (chunk_samples,)

        # Preprocess
        inputs = processor(
            audio_chunk, sampling_rate=sr, return_tensors="pt", padding=True
        ).to(device)

        # Extract embeddings (output of CNN feature extractor before transformer)
        with torch.no_grad():
            outputs = model(
                inputs["input_values"],
                output_hidden_states=True,
                return_dict=True,
            )
            # Use hidden states from the last transformer layer
            embeddings_chunk = outputs.hidden_states[-1]  # (1, T_chunk, 1024)
            embeddings_chunk = embeddings_chunk.squeeze(0).cpu().numpy()  # (T_chunk, 1024)

        all_embeddings.append(embeddings_chunk)

    # Concatenate chunks
    embeddings = np.concatenate(all_embeddings, axis=0)  # (T, 1024)
    embeddings = embeddings.astype("float32")

    # Save to cache
    np.save(npy_path, embeddings)

    return embeddings  # (T, 1024)
