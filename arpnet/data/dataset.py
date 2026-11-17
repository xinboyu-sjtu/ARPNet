"""Paired preoperative / postoperative CBCT dataset.

The dataset reads a CSV manifest with the following required columns:

    case_id, pre_path, post_path

and the following optional columns:

    label, split

``label`` is an integer NBA class (0, 1 or 2).  ``split`` is a free-form
string such as ``train``, ``val`` or ``test``.  Paths may be absolute or
relative to the manifest file.

Each NIfTI volume is expected to be a binary mask of shape
``(image_size, image_size, image_size)`` in ``(D, H, W)`` order, i.e. the
array returned by ``SimpleITK.GetArrayFromImage``.
"""

import os
import random
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset


class PairedCBCTDataset(Dataset):
    """Load paired preoperative and postoperative binary masks.

    Parameters
    ----------
    manifest:
        Path to the CSV manifest.
    split:
        If given, only rows whose ``split`` column equals this value are used.
    image_size:
        Expected cubic size of each volume.
    balance:
        If True, replicate minority classes in the selected split so that all
        classes have the same number of samples.  This mirrors the class
        balancing used in the original training pipeline.
    seed:
        Random seed for class balancing.
    require_label:
        If True, raise an error when the ``label`` column is missing.
    load_post:
        If False, the postoperative volume is not read and a zero volume is
        returned instead.  This is useful for inference on unlabelled test
        cases.
    """

    def __init__(
        self,
        manifest: str,
        split: Optional[str] = None,
        image_size: int = 192,
        balance: bool = False,
        seed: int = 42,
        require_label: bool = True,
        load_post: bool = True,
    ) -> None:
        self.manifest = manifest
        self.image_size = int(image_size)
        self.load_post = bool(load_post)

        frame = pd.read_csv(manifest)
        required_columns = ["case_id", "pre_path"]
        if load_post:
            required_columns.append("post_path")
        for column in required_columns:
            if column not in frame.columns:
                raise ValueError(
                    f"Manifest is missing required column '{column}': {manifest}"
                )

        if require_label and "label" not in frame.columns:
            raise ValueError(
                f"Manifest is missing required column 'label': {manifest}"
            )

        if split is not None:
            if "split" not in frame.columns:
                raise ValueError(
                    "A split was requested but the manifest has no 'split' "
                    "column."
                )
            frame = frame[frame["split"].astype(str) == str(split)]

        if len(frame) == 0:
            raise ValueError(
                f"No samples found for split '{split}' in {manifest}."
            )

        self.frame = frame.reset_index(drop=True)
        self.base_dir = os.path.dirname(os.path.abspath(manifest))
        self.indices = list(range(len(self.frame)))

        if balance:
            self.indices = self._balanced_indices(seed)

    def _resolve_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(self.base_dir, path))

    def _balanced_indices(self, seed: int) -> List[int]:
        if "label" not in self.frame.columns:
            raise ValueError("Class balancing requires a 'label' column.")

        labels = self.frame["label"].astype(int).tolist()
        by_class: Dict[int, List[int]] = {}
        for index, label in enumerate(labels):
            by_class.setdefault(label, []).append(index)

        max_count = max(len(indices) for indices in by_class.values())
        rng = random.Random(seed)
        balanced: List[int] = []
        for label in sorted(by_class):
            indices = by_class[label]
            balanced.extend(indices)
            if len(indices) < max_count:
                balanced.extend(
                    rng.choices(indices, k=max_count - len(indices))
                )
        rng.shuffle(balanced)
        return balanced

    def __len__(self) -> int:
        return len(self.indices)

    def pre_path_at(self, index: int) -> str:
        """Return the resolved preoperative path for ``index``."""
        row = self.frame.iloc[self.indices[index]]
        return self._resolve_path(str(row["pre_path"]))

    def __getitem__(self, index: int):
        row = self.frame.iloc[self.indices[index]]

        pre = sitk.GetArrayFromImage(
            sitk.ReadImage(self._resolve_path(str(row["pre_path"])))
        ).astype(np.float32)
        expected = (self.image_size, self.image_size, self.image_size)
        if pre.shape != expected:
            raise ValueError(
                f"Case '{row['case_id']}' has pre shape {pre.shape}; "
                f"expected {expected}. Run scripts/prepare_crops.py or "
                "check the manifest."
            )

        if self.load_post and "post_path" in row and not pd.isna(row["post_path"]):
            post_path = self._resolve_path(str(row["post_path"]))
            if os.path.isfile(post_path):
                post = sitk.GetArrayFromImage(sitk.ReadImage(post_path)).astype(
                    np.float32
                )
                if post.shape != expected:
                    raise ValueError(
                        f"Case '{row['case_id']}' has post shape {post.shape}; "
                        f"expected {expected}."
                    )
            else:
                post = np.zeros_like(pre)
        else:
            post = np.zeros_like(pre)

        # Input is mapped to {-1, +1}; target is binary {0, 1}.
        pre = np.where(pre >= 0.5, 1.0, -1.0).astype(np.float32)
        post = (post >= 0.5).astype(np.float32)

        label = -1
        if "label" in row and not pd.isna(row["label"]):
            label = int(row["label"])

        # Add a channel dimension: [1, D, H, W].
        return (
            torch.from_numpy(pre[None]),
            torch.from_numpy(post[None]),
            torch.tensor(label, dtype=torch.long),
            str(row["case_id"]),
        )

    def class_counts(self) -> Dict[int, int]:
        """Return the number of samples per class in the selected split."""
        if "label" not in self.frame.columns:
            return {}
        labels = self.frame["label"].astype(int).tolist()
        return {label: labels.count(label) for label in sorted(set(labels))}

    def unique_case_ids(self):
        """Return the unique case IDs in the selected split."""
        return self.frame["case_id"].astype(str).tolist()
