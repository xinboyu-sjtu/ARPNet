"""Clinical-variable encoding for the multimodal Stage II model.

The multimodal Stage II model uses four clinical variables in addition to the
preoperative and Stage I-predicted volumes:

    tooth type : molar / premolar / incisor / canine   -> 4 one-hot values
    jaw        : maxilla / mandible                    -> 2 one-hot values
    gender     : 0 / 1                                 -> 1 value
    age        : standardized using training statistics -> 1 value

The resulting vector has 8 dimensions.
"""

from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

TOOTH_TYPE_MAP = {"molar": 0, "premolar": 1, "incisor": 2, "canine": 3}
JAW_MAP = {"max": 0, "mand": 1}
CLINICAL_DIM = 8


def load_clinical_csv(
    csv_path: str,
    fit_case_ids: Optional[Iterable[str]] = None,
    age_mean: Optional[float] = None,
    age_std: Optional[float] = None,
) -> Tuple[Dict[str, np.ndarray], float, float]:
    """Load and encode clinical variables from a CSV file.

    Parameters
    ----------
    csv_path:
        CSV with columns ``case_id, tooth_type, jaw, age, gender``.
    fit_case_ids:
        Case IDs used to compute the age mean and standard deviation.  Pass the
        training-split IDs to avoid using validation/test statistics.  If not
        given, all rows are used.
    age_mean, age_std:
        Pre-computed standardization statistics.  When loading a trained
        checkpoint for inference, pass the values stored in the checkpoint so
        that the encoding matches training exactly.

    Returns
    -------
    clinical_data:
        Mapping ``case_id -> float32 vector of length 8``.
    age_mean, age_std:
        The standardized-age statistics that were used.
    """
    frame = pd.read_csv(csv_path)
    required = ["case_id", "tooth_type", "jaw", "age", "gender"]
    for column in required:
        if column not in frame.columns:
            raise ValueError(
                f"Clinical CSV is missing required column '{column}': {csv_path}"
            )

    if age_mean is None or age_std is None:
        fit_frame = frame
        if fit_case_ids is not None:
            fit_ids = {str(value) for value in fit_case_ids}
            fit_frame = frame[frame["case_id"].astype(str).isin(fit_ids)]
            if len(fit_frame) == 0:
                fit_frame = frame
        age_mean = float(fit_frame["age"].astype(float).mean())
        age_std = float(fit_frame["age"].astype(float).std(ddof=0))
        if not np.isfinite(age_std) or age_std == 0:
            age_std = 1.0

    clinical_data: Dict[str, np.ndarray] = {}
    for _, row in frame.iterrows():
        vector = np.zeros(CLINICAL_DIM, dtype=np.float32)

        tooth_type = str(row["tooth_type"]).strip().lower()
        jaw = str(row["jaw"]).strip().lower()
        if tooth_type not in TOOTH_TYPE_MAP:
            raise ValueError(f"Unknown tooth_type '{tooth_type}'.")
        if jaw not in JAW_MAP:
            raise ValueError(f"Unknown jaw '{jaw}'.")

        vector[TOOTH_TYPE_MAP[tooth_type]] = 1.0
        vector[4 + JAW_MAP[jaw]] = 1.0
        vector[6] = float(row["gender"])
        vector[7] = (float(row["age"]) - age_mean) / age_std

        clinical_data[str(row["case_id"]).strip()] = vector

    return clinical_data, age_mean, age_std


class MultimodalDataset(Dataset):
    """Wrap a paired-image dataset and append a clinical vector.

    ``PairedCBCTDataset`` returns ``(pre, post, label, case_id)``.  This
    wrapper returns ``(pre, post, label, clinical, case_id)``.  Cases without a
    clinical record receive an all-zero vector, which is the same fallback used
    by the original implementation.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        clinical_data: Dict[str, np.ndarray],
    ) -> None:
        self.base_dataset = base_dataset
        self.clinical_data = clinical_data

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        pre, post, label, case_id = self.base_dataset[index]
        clinical = self.clinical_data.get(
            str(case_id),
            np.zeros(CLINICAL_DIM, dtype=np.float32),
        )
        return (
            pre,
            post,
            label,
            torch.from_numpy(clinical),
            case_id,
        )
