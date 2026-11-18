"""Functional alignment tests for the ARPNet public release.

These tests do not require a GPU, patient data, or model checkpoints.  They
cover the paper-alignment logic that can be verified independently of training.
"""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from arpnet.data.clinical import load_clinical_csv
from arpnet.data.dataset import PairedCBCTDataset
from arpnet.losses import multiclass_brier
from arpnet.metrics import (
    brier_skill_score,
    calibration_curve,
    classification_metrics,
    expected_calibration_error,
)
from arpnet.utils import EarlyStopping, validate_encoder_initialization


class ClinicalEncodingTests(unittest.TestCase):
    def test_gender_is_binary_and_age_is_zscored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "clinical.csv")
            pd.DataFrame(
                [
                    {
                        "case_id": "a",
                        "tooth_type": "molar",
                        "jaw": "mand",
                        "age": 50,
                        "gender": 1,
                    },
                    {
                        "case_id": "b",
                        "tooth_type": "premolar",
                        "jaw": "max",
                        "age": 40,
                        "gender": 0,
                    },
                ]
            ).to_csv(path, index=False)

            data, mean, std = load_clinical_csv(path)
            vector = data["a"]
            self.assertEqual(vector.shape, (8,))
            self.assertEqual(vector[0], 1.0)   # molar
            self.assertEqual(vector[5], 1.0)   # mandible
            self.assertEqual(vector[6], 1.0)   # gender (binary)
            self.assertAlmostEqual(vector[7], (50 - mean) / std)
            self.assertEqual(int((vector[4:6] > 0).sum()), 1)

    def test_missing_training_ids_raise_instead_of_using_full_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "clinical.csv")
            pd.DataFrame(
                [
                    {
                        "case_id": "other",
                        "tooth_type": "molar",
                        "jaw": "mand",
                        "age": 50,
                        "gender": 1,
                    }
                ]
            ).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_clinical_csv(path, fit_case_ids=["not_present"])


class OversamplingTests(unittest.TestCase):
    def test_paper_counts_become_420_entries(self):
        rows = []
        for label, count in enumerate((140, 62, 22)):
            for index in range(count):
                rows.append(
                    {
                        "case_id": f"c{label}_{index}",
                        "pre_path": "pre.nii.gz",
                        "post_path": "post.nii.gz",
                        "label": label,
                        "split": "train",
                    }
                )
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "manifest.csv")
            pd.DataFrame(rows).to_csv(path, index=False)
            dataset = PairedCBCTDataset(
                path,
                split="train",
                balance=True,
                seed=42,
            )
            self.assertEqual(len(dataset), 420)
            self.assertEqual(
                dataset.selected_class_counts(),
                {0: 140, 1: 140, 2: 140},
            )

    def test_validation_is_not_oversampled(self):
        rows = []
        for label, count in enumerate((47, 17, 11)):
            for index in range(count):
                rows.append(
                    {
                        "case_id": f"v{label}_{index}",
                        "pre_path": "pre.nii.gz",
                        "post_path": "post.nii.gz",
                        "label": label,
                        "split": "val",
                    }
                )
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "manifest.csv")
            pd.DataFrame(rows).to_csv(path, index=False)
            dataset = PairedCBCTDataset(
                path,
                split="val",
                balance=False,
                seed=42,
            )
            self.assertEqual(len(dataset), 75)
            self.assertEqual(dataset.selected_class_counts(), {0: 47, 1: 17, 2: 11})


class MetricTests(unittest.TestCase):
    def test_weighted_recall_equals_accuracy(self):
        y_true = np.array([0] * 47 + [1] * 20 + [2] * 8)
        probabilities = np.full((len(y_true), 3), 1.0 / 3.0)
        predictions = probabilities.argmax(axis=1)
        metrics = classification_metrics(
            y_true, predictions, probabilities, train_priors=(140 / 224, 62 / 224, 22 / 224)
        )
        self.assertAlmostEqual(metrics["accuracy"], metrics["recall_weighted"])

    def test_brier_is_averaged_over_cases_and_classes(self):
        y_true = np.array([0, 1, 2, 0])
        probabilities = np.array(
            [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5], [0.6, 0.3, 0.1]]
        )
        predictions = probabilities.argmax(axis=1)
        metrics = classification_metrics(y_true, predictions, probabilities)
        one_hot = np.eye(3)[y_true]
        expected = np.mean((probabilities - one_hot) ** 2)
        self.assertAlmostEqual(metrics["brier"], expected)
        old_definition = np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))
        self.assertAlmostEqual(old_definition, 3 * expected)

    def test_reference_brier_matches_paper_values(self):
        priors = (140 / 224, 62 / 224, 22 / 224)
        internal = np.array([0] * 47 + [1] * 20 + [2] * 8)
        external = np.array([0] * 20 + [1] * 8 + [2] * 2)
        internal_reference = brier_skill_score(
            internal, np.full((75, 3), 1 / 3), priors
        )["brier_reference"]
        external_reference = brier_skill_score(
            external, np.full((30, 3), 1 / 3), priors
        )["brier_reference"]
        self.assertAlmostEqual(internal_reference, 0.17499220521541956)
        self.assertAlmostEqual(external_reference, 0.16094458616780047)

    def test_calibration_has_five_bins_and_perfect_ece_is_zero(self):
        y_true = np.array([0, 1, 2])
        probabilities = np.eye(3)
        rows = calibration_curve(y_true, probabilities, n_bins=5)
        self.assertGreaterEqual(len(rows), 1)
        self.assertTrue(all(row["bin"] < 5 for row in rows))
        self.assertAlmostEqual(
            expected_calibration_error(y_true, probabilities, n_bins=5),
            0.0,
        )

    def test_multiclass_brier_loss_matches_manual_definition(self):
        logits = torch.tensor(
            [[2.0, 0.5, -1.0], [0.1, 1.2, -0.3]], requires_grad=True
        )
        labels = torch.tensor([0, 1])
        loss = multiclass_brier(logits, labels)
        probabilities = torch.softmax(logits, dim=1)
        one_hot = torch.nn.functional.one_hot(labels, 3).float()
        expected = ((probabilities - one_hot) ** 2).mean()
        self.assertTrue(torch.allclose(loss, expected))


class TrainingUtilityTests(unittest.TestCase):
    def test_early_stopping(self):
        stopping = EarlyStopping(patience=2, mode="min")
        self.assertTrue(stopping.step(1.0))
        self.assertFalse(stopping.step(1.1))
        self.assertFalse(stopping.step(1.2))
        self.assertTrue(stopping.stopped)

    def test_encoder_validation_requires_both_encoders(self):
        with self.assertRaises(ValueError):
            validate_encoder_initialization(
                encoder1_ckpt="encoder1.pth",
                encoder2_ckpt=None,
                init_stage2_ckpt=None,
                freeze_encoders=False,
                unfreeze_last_block=True,
            )

    def test_encoder_validation_accepts_full_stage2_checkpoint(self):
        validate_encoder_initialization(
            encoder1_ckpt=None,
            encoder2_ckpt=None,
            init_stage2_ckpt="stage2.pth",
            freeze_encoders=True,
            unfreeze_last_block=False,
        )

    def test_encoder_validation_allows_from_scratch_training(self):
        validate_encoder_initialization(
            encoder1_ckpt=None,
            encoder2_ckpt=None,
            init_stage2_ckpt=None,
            freeze_encoders=False,
            unfreeze_last_block=False,
        )


if __name__ == "__main__":
    unittest.main()
