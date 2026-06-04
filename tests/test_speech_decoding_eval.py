"""
Unit tests for speech_decoding_eval module.

Tests the Défossez et al. 2023 protocol implementation:
- Top-k accuracy computation
- Cross-session stability evaluation
- Full evaluation pipeline

Run with: pytest tests/test_speech_decoding_eval.py -v
"""

import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from astrostream.evaluation.speech_decoding_eval import (
    compute_topk_accuracy,
    evaluate_cross_session,
    run_full_evaluation,
)


class DummyDataset:
    """Mock dataset for testing."""

    def __init__(self, n_samples: int = 100, n_subjects: int = 3, seed: int = 42):
        np.random.seed(seed)
        self.n_samples = n_samples
        self.n_subjects = n_subjects
        self.samples = []

        for i in range(n_samples):
            subject_id = i % n_subjects
            session_id = i // (n_samples // 3)
            self.samples.append({
                "meg": np.random.randn(208, 1000),  # (C, T)
                "audio_embedding": np.random.randn(256),  # (D,)
                "label": i % 50,  # 50 candidate clips
                "subject_id": subject_id,
                "session_id": session_id,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class DummyModel(nn.Module):
    """Mock frozen encoder for testing."""

    def __init__(self, output_dim: int = 256):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, x):
        """Return fixed-size representation."""
        batch_size = x.shape[0]
        repr_batch = torch.randn(batch_size, self.output_dim)
        return {"repr": repr_batch}


# ==================== Tests for compute_topk_accuracy ====================


class TestComputeTopkAccuracy:
    """Tests for compute_topk_accuracy function."""

    def test_perfect_accuracy(self):
        """Test case where model perfectly matches true labels."""
        # Perfect case: MEG embeddings exactly match their corresponding audio embeddings
        N_test, N_candidates, D = 10, 50, 256

        # Create embeddings where first N_test audio embeddings are "correct"
        audio_embeddings = np.random.randn(N_candidates, D)
        audio_embeddings = audio_embeddings / (
            np.linalg.norm(audio_embeddings, axis=1, keepdims=True) + 1e-8
        )

        # MEG representations are copies of correct audio embeddings
        meg_repr = audio_embeddings[:N_test].copy()

        # True labels are 0, 1, 2, ..., N_test-1
        labels = np.arange(N_test)

        acc = compute_topk_accuracy(meg_repr, audio_embeddings, labels, k_values=[1, 5, 10])

        assert acc[1] == pytest.approx(1.0, abs=1e-6)
        assert acc[5] == pytest.approx(1.0, abs=1e-6)
        assert acc[10] == pytest.approx(1.0, abs=1e-6)

    def test_random_chance(self):
        """Test random case (expected accuracy ≈ k / N_candidates)."""
        N_test, N_candidates, D = 100, 500, 256

        meg_repr = np.random.randn(N_test, D)
        audio_embeddings = np.random.randn(N_candidates, D)
        labels = np.random.randint(0, N_candidates, size=N_test)

        acc = compute_topk_accuracy(meg_repr, audio_embeddings, labels, k_values=[1, 10])

        # Expected accuracy ≈ k / N_candidates
        # Top-1: 1/500 = 0.002, Top-10: 10/500 = 0.02
        # Allow large tolerance for random case
        assert 0.0 <= acc[1] <= 0.05
        assert 0.0 <= acc[10] <= 0.1

    def test_decreasing_with_k(self):
        """Test that accuracy is non-increasing as k increases."""
        N_test, N_candidates, D = 50, 100, 256

        meg_repr = np.random.randn(N_test, D)
        audio_embeddings = np.random.randn(N_candidates, D)
        labels = np.random.randint(0, N_candidates, size=N_test)

        acc = compute_topk_accuracy(
            meg_repr, audio_embeddings, labels, k_values=[1, 5, 10, 20]
        )

        # Accuracy should be monotonically non-decreasing with k
        assert acc[1] <= acc[5]
        assert acc[5] <= acc[10]
        assert acc[10] <= acc[20]

    def test_invalid_k(self):
        """Test that invalid k raises ValueError."""
        meg_repr = np.random.randn(10, 256)
        audio_embeddings = np.random.randn(50, 256)
        labels = np.arange(10)

        with pytest.raises(ValueError, match="max k=100 exceeds"):
            compute_topk_accuracy(meg_repr, audio_embeddings, labels, k_values=[100])

    def test_dimension_mismatch(self):
        """Test that dimension mismatch raises ValueError."""
        meg_repr = np.random.randn(10, 256)
        audio_embeddings = np.random.randn(50, 128)  # Different dimension
        labels = np.arange(10)

        with pytest.raises(ValueError, match="Dimension mismatch"):
            compute_topk_accuracy(meg_repr, audio_embeddings, labels)

    def test_shape_validation(self):
        """Test that invalid shapes raise ValueError."""
        audio_embeddings = np.random.randn(50, 256)
        labels = np.arange(10)

        # Wrong MEG shape (1D instead of 2D)
        with pytest.raises(ValueError, match="meg_repr must be 2D"):
            compute_topk_accuracy(
                np.random.randn(256),
                audio_embeddings,
                labels,
            )

        # Wrong labels shape (2D instead of 1D)
        with pytest.raises(ValueError, match="labels must be 1D"):
            compute_topk_accuracy(
                np.random.randn(10, 256),
                audio_embeddings,
                np.random.randint(0, 50, size=(10, 1)),
            )

    def test_type_validation(self):
        """Test that non-numpy inputs raise TypeError."""
        audio_embeddings = np.random.randn(50, 256)
        labels = np.arange(10)

        # List instead of numpy array
        with pytest.raises(TypeError):
            compute_topk_accuracy(
                [[1, 2, 3]],
                audio_embeddings,
                labels,
            )

    def test_length_mismatch(self):
        """Test that mismatched lengths raise ValueError."""
        meg_repr = np.random.randn(10, 256)
        audio_embeddings = np.random.randn(50, 256)
        labels = np.arange(15)  # Wrong length

        with pytest.raises(ValueError, match="labels length"):
            compute_topk_accuracy(meg_repr, audio_embeddings, labels)


# ==================== Tests for evaluate_cross_session ====================


class TestEvaluateCrossSession:
    """Tests for evaluate_cross_session function."""

    def test_basic_cross_session(self):
        """Test basic cross-session evaluation."""
        dataset = DummyDataset(n_samples=100, n_subjects=4)
        model = DummyModel(output_dim=256)

        results = evaluate_cross_session(
            model,
            train_subjects=[0, 1],
            test_subjects=[2, 3],
            dataset=dataset,
            persistent=False,
            device="cpu",
        )

        # Check return dict keys
        assert "train_top1" in results
        assert "train_top10" in results
        assert "test_top1" in results
        assert "test_top10" in results
        assert "session_gap_degradation" in results
        assert "n_train" in results
        assert "n_test" in results

        # Check value ranges
        assert 0.0 <= results["train_top1"] <= 1.0
        assert 0.0 <= results["test_top1"] <= 1.0
        assert results["session_gap_degradation"] >= 0.0  # Usually gap > 0

    def test_identical_subjects_error(self):
        """Test that identical train/test subjects raise ValueError."""
        dataset = DummyDataset()
        model = DummyModel()

        with pytest.raises(ValueError, match="must be different"):
            evaluate_cross_session(
                model,
                train_subjects=[0, 1],
                test_subjects=[0, 1],  # Same!
                dataset=dataset,
                device="cpu",
            )

    def test_missing_subjects_error(self):
        """Test that missing subjects raise RuntimeError."""
        dataset = DummyDataset(n_subjects=2)
        model = DummyModel()

        with pytest.raises(RuntimeError, match="No training samples"):
            evaluate_cross_session(
                model,
                train_subjects=[99],  # Subject doesn't exist
                test_subjects=[0],
                dataset=dataset,
                device="cpu",
            )


# ==================== Tests for run_full_evaluation ====================


class TestRunFullEvaluation:
    """Tests for run_full_evaluation function."""

    def test_full_evaluation_pipeline(self):
        """Test complete evaluation pipeline."""
        dataset = DummyDataset(n_samples=50, n_subjects=3)
        model = DummyModel(output_dim=256)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            cfg = OmegaConf.create({
                "eval": {
                    "test_split": 0.2,
                    "random_seed": 42,
                    "persistent": False,
                }
            })

            results = run_full_evaluation(
                model,
                dataset,
                cfg,
                output_dir,
                device="cpu",
            )

            # Check return dict structure
            assert "top1" in results
            assert "top5" in results
            assert "top10" in results
            assert "n_samples" in results
            assert "embedding_dim" in results
            assert "per_subject_top1" in results
            assert "per_subject_top10" in results
            assert "per_subject_n_samples" in results
            assert "cross_session" in results

            # Check value ranges
            assert 0.0 <= results["top1"] <= 1.0
            assert 0.0 <= results["top10"] <= 1.0
            assert results["top1"] <= results["top10"]  # top-1 ≤ top-10

            # Check that results were saved
            results_file = output_dir / "eval_results.json"
            assert results_file.exists()

    def test_per_subject_breakdown(self):
        """Test per-subject accuracy breakdown."""
        dataset = DummyDataset(n_samples=60, n_subjects=3)
        model = DummyModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = OmegaConf.create({"eval": {}})
            results = run_full_evaluation(model, dataset, cfg, Path(tmpdir), device="cpu")

            per_subject_acc = results["per_subject_top1"]
            per_subject_n = results["per_subject_n_samples"]

            # Should have 3 subjects
            assert len(per_subject_acc) == 3
            assert len(per_subject_n) == 3

            # Each subject should have samples
            for subj_id in per_subject_n:
                assert per_subject_n[subj_id] > 0
                assert 0.0 <= per_subject_acc[subj_id] <= 1.0


# ==================== Integration tests ====================


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_defossez_benchmark_comparable(self):
        """Test that implementation matches Défossez et al. 2023 protocol."""
        # Create a scenario with perfect accuracy to verify protocol
        N_test, N_candidates, D = 79, 500, 256  # Like Défossez paper

        # Perfect embeddings: MEG matches true audio
        audio_embeddings = np.random.randn(N_candidates, D)
        audio_embeddings = audio_embeddings / (
            np.linalg.norm(audio_embeddings, axis=1, keepdims=True) + 1e-8
        )

        meg_repr = audio_embeddings[:N_test].copy()
        labels = np.arange(N_test)

        acc = compute_topk_accuracy(meg_repr, audio_embeddings, labels, k_values=[1, 10])

        # Should achieve 100% accuracy
        assert acc[1] == pytest.approx(1.0)
        assert acc[10] == pytest.approx(1.0)

    def test_reproducibility(self):
        """Test that results are reproducible with same seed."""
        dataset = DummyDataset(n_samples=50, n_subjects=3, seed=42)
        model = DummyModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = OmegaConf.create({"eval": {}})

            # Run twice
            results1 = run_full_evaluation(model, dataset, cfg, Path(tmpdir), device="cpu")
            results2 = run_full_evaluation(model, dataset, cfg, Path(tmpdir), device="cpu")

            # Same model, same data should give same results
            assert results1["top1"] == pytest.approx(results2["top1"])
            assert results1["top10"] == pytest.approx(results2["top10"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
