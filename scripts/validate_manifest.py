#!/usr/bin/env python
"""Validate a patient-level ARPNet manifest.

Checks:

- required columns are present;
- each ``case_id`` occurs only once in the manifest;
- a case never appears in more than one split;
- optional expected split sizes and class counts match the paper's split.

Example
-------
python scripts/validate_manifest.py \
    --manifest data/manifest.csv \
    --expected_train 224 --expected_val 75 --expected_test 75 \
    --expected_train_classes 140 62 22 \
    --expected_val_classes 47 17 11 \
    --expected_test_classes 47 20 8
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an ARPNet manifest.")
    parser.add_argument("--manifest", required=True, help="CSV manifest.")
    parser.add_argument("--case_id_col", default="case_id", help="Case ID column.")
    parser.add_argument("--split_col", default="split", help="Split column.")
    parser.add_argument("--label_col", default="label", help="Label column.")
    parser.add_argument("--expected_train", type=int, default=None)
    parser.add_argument("--expected_val", type=int, default=None)
    parser.add_argument("--expected_test", type=int, default=None)
    parser.add_argument(
        "--expected_train_classes",
        type=int,
        nargs=3,
        default=None,
        help="Expected class counts in the training split.",
    )
    parser.add_argument(
        "--expected_val_classes",
        type=int,
        nargs=3,
        default=None,
        help="Expected class counts in the validation split.",
    )
    parser.add_argument(
        "--expected_test_classes",
        type=int,
        nargs=3,
        default=None,
        help="Expected class counts in the test split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.manifest)

    for column in (args.case_id_col, args.split_col):
        if column not in frame.columns:
            raise ValueError(f"Manifest is missing required column '{column}'.")

    if frame[args.case_id_col].duplicated().any():
        duplicates = frame.loc[
            frame[args.case_id_col].duplicated(), args.case_id_col
        ].tolist()
        raise ValueError(f"Duplicate case_id values: {duplicates[:10]}")

    split_per_case = (
        frame.groupby(args.case_id_col)[args.split_col].nunique()
    )
    if (split_per_case > 1).any():
        leaking = split_per_case[split_per_case > 1].index.tolist()
        raise ValueError(
            "Cases assigned to more than one split: " + ", ".join(leaking[:10])
        )

    print("Split sizes:")
    print(frame[args.split_col].value_counts().sort_index().to_string())

    if args.label_col in frame.columns:
        print("\nClass counts per split:")
        table = pd.crosstab(frame[args.split_col], frame[args.label_col])
        print(table.to_string())

    expected = {
        "train": (args.expected_train, args.expected_train_classes),
        "val": (args.expected_val, args.expected_val_classes),
        "test": (args.expected_test, args.expected_test_classes),
    }
    errors = []
    for split, (size, classes) in expected.items():
        subset = frame[frame[args.split_col].astype(str) == split]
        if size is not None and len(subset) != size:
            errors.append(
                f"{split}: expected {size} cases, found {len(subset)}"
            )
        if classes is not None:
            counts = (
                subset[args.label_col].astype(int).value_counts().sort_index()
            )
            observed = [int(counts.get(i, 0)) for i in range(3)]
            if observed != list(classes):
                errors.append(
                    f"{split}: expected class counts {list(classes)}, "
                    f"found {observed}"
                )

    if errors:
        print("\nERRORS:")
        for error in errors:
            print(" -", error)
        raise SystemExit(1)

    print("\nManifest validation passed.")


if __name__ == "__main__":
    main()
