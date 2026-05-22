# Astro-Stream NeuroAI

A research codebase for **streaming neuroai models** that decode neural signals in real-time. Astro-Stream combines **Mamba state-space models** with **astrocyte-inspired gating mechanisms** to achieve efficient, causal neural decoding from MEG/EEG data.

**Language:** 100% Python | **Status:** Active Development

## Quick Start

### Installation

```bash
git clone https://github.com/josh-oloro/astro-stream.git
cd astro-stream
pip install -r requirements.txt
# For CUDA 12.1:
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

### Running Experiments

```bash
# Phase 1: Reproduce Défossez baseline (C0)
python experiments/00_reproduce_defossez.py --config configs/train/phase1_baseline.yaml

# Phase 2: Run ablation study (C0–C7)
python experiments/01_run_ablation.py --config configs/train/phase2_ablation.yaml

# Phase 3: Closure metric sweep
python experiments/02_closure_sweep.py --closure-threshold 0.5
```

## Architecture Overview

### Core Innovation: Astro-Stream

Astro-Stream combines three mechanisms for efficient, causal neural decoding:

1. **Persistent-State Mamba** – Maintains hidden state across patches for temporal context
2. **Astrocyte-Inspired Gating** – Learned gate modulates encoder updates (α=0.02 → 200ms timescale)
3. **Closure Training Objective** – Predicts future latent embeddings with tightness metric

### Component Modules

#### 1. Data Pipeline (`astrostream/data/`)
- **`preprocessing.py`** – `MNEPreprocessor` class for MEG/EEG preprocessing
  - Loads raw FIF/CTF files using MNE-Python
  - IIR bandpass filtering (causal, low-latency)
  - Resampling, epoching, peak-to-peak rejection
  - HDF5 caching with metadata for fast reloading
  - `extract_wav2vec_embeddings()` for speech alignment (30s chunks, GPU-efficient)

- **`dataset.py`** – PyTorch Dataset and DataModule
  - `MEGSpeechDataset`: loads epochs from HDF5, aligns with Wav2Vec2 embeddings
  - `collate_fn`: **sorts batches by (recording_id, segment_idx)** for persistent-state training
  - `split_by_subject()`: subject-level train/val/test splits (no leakage)
  - `MEGSpeechDataModule`: PyTorch Lightning compatible interface

#### 2. Model Architecture (`astrostream/models/`)

**Backbone Encoders** (`backbone/`):
- **`mamba_encoder.py`** – Selective state-space model with optional persistent state
  - d_model=512, n_layers=6, SSM state dim d_state=16
  - Supports episodic (stateless) or persistent (stateful) modes
  - Outputs latent codes: (B, T, d_model)

- **`transformer_encoder.py`** – Standard Transformer baseline
- **`lstm_encoder.py`** – LSTM baseline with peephole connections

**Gating Mechanisms** (`gates/`):
- **`astrocyte_gate.py`** – Calcium-inspired gating layer
  - Slow-state dynamics: s_t = (1-α)·s_{t-1} + α·x_t (α=0.02)
  - Learned threshold θ and weight w per channel
  - Gate output g_t ∈ [0,1] modulates encoder: x_t ← x_t ⊙ g_t

**Objectives** (`objectives/`):
- **`jepa.py`** – Joint-Embedding Predictive Architecture (LeCun 2022)
  - Target encoder (EMA-updated): (B, T, D) → (B, T, D)
  - Context predictor: (B, T, D) → (B, T, D_latent)
  - Loss: contrastive between predictions and target embeddings
  
- **`decoder_head.py`** – Linear probe for downstream evaluation
  - Fine-tuned on frozen representations
  - Speech decoding: 1594 word classes (Gwilliams test set)

#### 3. Training Infrastructure (`astrostream/training/`)
- **`trainer.py`** – Main training loop with multi-GPU support
  - Integrates JEPA loss + closure loss
  - Persistent state management across batches
  - Checkpoint saving with best-val tracking

- **`persistent_state.py`** – State manager for streaming inference
  - Maintains SSM hidden states: h_t ∈ (B, D_state, 1)
  - Resets on segment/recording boundaries
  - Enables causal inference without lookahead

- **`scheduler.py`** – Learning rate schedules
  - Linear warmup + cosine annealing
  - EMA momentum scheduling for target encoder

#### 4. Evaluation (`astrostream/evaluation/`)
- **`speech_decoding.py`** – Fine-tune linear decoder on learned representations
  - Logistic regression on frozen embeddings
  - Reports accuracy, F1, confusion matrices per subject

- **`closure_metric.py`** – Latent predictability metric
  - Computes PCA-based tightness of predictions
  - Reports closure@lag for lags 1–8 timesteps ahead

#### 5. Utilities (`astrostream/utils/`)
- **`config.py`** – OmegaConf config loading and validation
- **`logging.py`** – Weights & Biases integration, experiment tracking

## Datasets & Configurations

### MEG-MASC (Gwilliams et al. 2022)
- **27 subjects**, 208 MEG channels
- Movie-watching with speech transcriptions
- OpenNeuro ID: [ds004352](https://openneuro.org/datasets/ds004352)
- Config: `configs/data/gwilliams2022.yaml`
- Preprocessing: 1000 Hz → 200 Hz, 0.5–30 Hz bandpass, ±0.2–1.0s epochs

### MOUS (Schoffelen et al. 2019)
- **96 subjects**, 273 EEG channels
- Language production (Dutch)
- Donders repository: [10.34973/tg9s-5831](https://data.donders.radboudumc.nl/collections/di/dccn/DSC_3015033.02_268)
- Config: `configs/data/schoffelen2019.yaml`
- Same preprocessing as Gwilliams (200 Hz target)

## Ablation Study: Causal Mechanisms

The proposed Astro-Stream model (C7) combines three mechanisms. We isolate each to measure contribution:

| Config | Persistent State | Astrocyte Gate | Closure Training | Baseline | Description |
|--------|------------------|----------------|------------------|----------|-------------|
| **C0** | ✗ | ✗ | ✗ | ✓ | Episodic Mamba (Défossez 2023 replicate) |
| **C1** | ✓ | ✗ | ✗ | ✓ | Memory alone |
| **C2** | ✗ | ✓ | ✗ | ✓ | Gating alone |
| **C3** | ✓ | ✓ | ✗ | ✓ | Memory + Gate |
| **C4** | ✗ | ✗ | ✓ | ✓ | Closure training alone |
| **C5** | ✓ | ✗ | ✓ | ✓ | Memory + Closure |
| **C6** | ✗ | ✓ | ✓ | ✓ | Gate + Closure |
| **C7** | ✓ | ✓ | ✓ | ✓ | **Full Astro-Stream** |

All configurations use JEPA as the core training objective. Ablation configs in `configs/model/ablation/`.

## Model Configurations

### Base Architecture
- **d_model**: 512 (hidden dimension)
- **n_layers**: 6 Mamba blocks
- **Mamba SSM**: d_state=16, d_conv=4, expand=2
- **Astrocyte gate**: α=0.02 (200ms decay), per-channel learnable θ, w
- **JEPA**: EMA momentum 0.996→0.9999, 2-layer predictor
- **Closure metric**: max_lag=8, PCA dims [2, 4, 8, 16, 32, 64, 128]

See `configs/model/astrostream_base.yaml` for full specification.

### Baseline Models
- **Transformer**: 6-layer, 8 heads, d_model=512 (`configs/model/baselines/transformer.yaml`)
- **LSTM**: 2-layer, 512 hidden dim (`configs/model/baselines/lstm.yaml`)

## Training Workflows

### Phase 1: Baseline Training
```bash
python experiments/00_reproduce_defossez.py \
  --config configs/train/phase1_baseline.yaml \
  --dataset gwilliams2022
```
Trains C0 baseline on Gwilliams MEG-MASC data. Expected: ~0.65 speech decoding accuracy.

### Phase 2: Ablation Study
```bash
python experiments/01_run_ablation.py \
  --config configs/train/phase2_ablation.yaml \
  --sweep-configs C0,C1,C2,C3,C4,C5,C6,C7
```
Runs all 8 ablation conditions in sequence. Logs to Weights & Biases.

### Phase 3: Closure Threshold Sweep
```bash
python experiments/02_closure_sweep.py \
  --dataset schoffelen2019 \
  --closure-threshold 0.5 0.6 0.7 0.8
```
Measures closure metric @ different thresholds on MOUS EEG data.

### SLURM Array Job
Submit full ablation to HPC cluster:
```bash
sbatch scripts/submit_slurm_array.sh  # submits 8 parallel jobs (C0–C7)
```

## Training Pipeline Details

### Data Flow
```
Raw MEG/EEG files (FIF, CTF)
    ↓
MNEPreprocessor (filter, resample, epoch, reject)
    ↓
HDF5 cache (per subject, fast reload)
    ↓
MEGSpeechDataset (align with Wav2Vec2 embeddings)
    ↓
DataLoader (sorted by recording_id, segment_idx for persistent state)
    ↓
Mamba Encoder + Astrocyte Gate
    ↓
JEPA Head + Closure Objective
    ↓
Loss: L_JEPA + λ·L_closure
```

### Persistent State in Training
- Each batch is sorted by (recording_id, segment_idx)
- Hidden state h_t ∈ (B, D_state, 1) is maintained across consecutive segments
- Resets on segment boundaries (group_id change)
- Enables causal, real-time inference without lookahead

### Loss Function
```
L_total = L_JEPA(z_pred, z_target) + λ_closure · L_closure(predictions, targets)

L_JEPA: contrastive loss on joint embeddings
L_closure: PCA-based tightness metric on latent predictions
λ_closure: 0.1 (ablation-dependent, see config)
```

## Evaluation Metrics

### 1. Speech Decoding Accuracy
Fine-tune a linear decoder on frozen learned representations:
```python
from astrostream.evaluation import SpeechDecoder

decoder = SpeechDecoder(model=model, config=config)
train_acc, val_acc, test_acc = decoder.evaluate(train_set, val_set, test_set)
```
Reported per subject and overall.

### 2. Closure Metric
Measures predictive tightness—how well the model clusters future embeddings:
```python
from astrostream.evaluation import ClosureMetric

closure = ClosureMetric(max_lag=8, pca_dims=[2, 4, 8, 16, 32, 64, 128])
closure_scores = closure.compute(predictions, targets)
# Returns: {lag_1: 0.85, lag_2: 0.79, ..., lag_8: 0.45}
```

### 3. Real-Time Latency
Streaming inference latency per patch (10 ms @ 200 Hz):
```bash
python scripts/benchmark_latency.py --model-config configs/model/astrostream_base.yaml
# Expected: ~2–5 ms per 50-sample patch on GPU
```

## Testing

### Run All Tests
```bash
pytest tests/ -v --cov=astrostream --cov-report=term-missing
```

### Test Modules
- `test_patcher.py` – Patch embedding and masking
- `test_mamba_encoder.py` – SSM forward pass, persistent state
- `test_astrocyte_gate.py` – Gate dynamics, gradient flow
- `test_jepa.py` – Contrastive loss, EMA updates
- `test_closure_metric.py` – PCA-based tightness computation
- `test_trainer_persistent_state.py` – State management across batches

## Development

### Code Quality
```bash
# Lint
ruff check astrostream/

# Format
ruff format astrostream/

# Type check
mypy astrostream/ --ignore-missing-imports
```

### Project Structure
```
astro-stream/
├── astrostream/              # Core library (100% Python)
│   ├── data/
│   │   ├── preprocessing.py  # MNEPreprocessor, Wav2Vec2 extraction
│   │   ├── dataset.py        # MEGSpeechDataset, DataModule
│   │   └── utils.py
│   ├── models/
│   │   ├── patcher.py        # Patch embedding layer
│   │   ├── astrostream.py    # Main model class
│   │   ├── backbone/
│   │   │   ├── mamba_encoder.py
│   │   │   ├── transformer_encoder.py
│   │   │   └── lstm_encoder.py
│   │   ├── gates/
│   │   │   └── astrocyte_gate.py
│   │   └── objectives/
│   │       ├── jepa.py
│   │       └── decoder_head.py
│   ├── training/
│   │   ├── trainer.py
│   │   ├── persistent_state.py
│   │   └── scheduler.py
│   ├── evaluation/
│   │   ├── speech_decoding.py
│   │   └── closure_metric.py
│   └── utils/
│       ├── config.py
│       └── logging.py
├── configs/                  # YAML configurations
│   ├── data/                 # Dataset configs
│   ├── model/                # Model architectures + ablations
│   └── train/                # Training hyperparameters
├── experiments/              # High-level experiment runners
├── scripts/                  # Download, preprocess, SLURM utilities
├── tests/                    # Pytest suite
├── README.md
├── LICENSE                   # MIT
├── pyproject.toml            # PEP 621 packaging
└── requirements.txt          # Pinned dependencies (CUDA 12.1 compatible)
```

## Contributing

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Implement changes with full type annotations and docstrings
3. Add tests in `tests/`
4. Run linting: `ruff check astrostream/`
5. Run type checking: `mypy astrostream/`
6. Run tests: `pytest tests/`
7. Submit PR to `main`

### Code Standards
- **No placeholders**: every function has real implementation
- **Shape comments**: `# (B, T, C) → (B, T, D)` on tensor operations
- **PyTorch only**: no TensorFlow/JAX
- **Dataclass hyperparameters**: no magic numbers
- **Module docstrings**: explain I/O and research context
- **Full type annotations**: no bare `Any`

## License

MIT License. See `LICENSE` file for details.

## Citation

If you use Astro-Stream in your research, please cite:

```bibtex
@article{astrostream2024,
  title={Astro-Stream: Streaming Neuroai Models with Astrocyte-Inspired Gating},
  author={Oloro, Josh and others},
  journal={Preprint},
  year={2024},
  note={Available at https://github.com/josh-oloro/astro-stream}
}
```

## Authors

- **Joshua Philippe Olorocisimo, Ph.D.** – Project Advisor
- **Paul R. Regonia, Ph.D.** – Project Leader
- **Jeric C. Briones, Ph.D.** – Project Leader

## Contact & Support

- **Issues**: [GitHub Issues](https://github.com/josh-oloro/astro-stream/issues)
- **Discussions**: [GitHub Discussions](https://github.com/josh-oloro/astro-stream/discussions)
- **Email**: josholorocisimo@gmail.com ; pdregonia@up.edu.ph ; jbriones@ateneo.edu

---

**Last Updated**: May 2024 | **Status**: Active Development
