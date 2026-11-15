#!/usr/bin/env python
"""Crop registered preoperative/postoperative mask pairs to 192^3.

This is a lightweight helper for the final preprocessing step.  The full
pipeline (jaw segmentation with DentalSegmentator, virtual tooth extraction
and surface registration) is performed in 3D Slicer; this script starts from
the aligned binary masks and produces the cubic volumes used by ARPNet.

Input manifest columns: ``case_id, pre_path, post_path, label, split``.
Output: ``pre_<case_id>.nii.gz``, ``post_<case_id>.nii.gz`` and a new
``manifest.csv`` inside ``--out_dir``.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import SimpleITK as sitk

from arpnet.data.preprocessing import (
    binarize,
    crop_pair,
    read_image,
    resample_image,
    write_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop ARP mask pairs.")
    parser.add_argument("--manifest", required=True, help="Input CSV manifest.")
    parser.add_argument("--out_dir", required=True, help="Output directory.")
    parser.add_argument("--size", type=int, default=192, help="Cubic crop size.")
    parser.add_argument(
        "--spacing",
        type=float,
        default=None,
        help="Optional isotropic spacing in mm (e.g. 0.2).",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Mask threshold.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    frame = pd.read_csv(args.manifest)
    for column in ("case_id", "pre_path", "post_path"):
        if column not in frame.columns:
            raise ValueError(f"Manifest is missing required column '{column}'.")

    base_dir = os.path.dirname(os.path.abspath(args.manifest))
    output_rows = []

    for _, row in frame.iterrows():
        case_id = str(row["case_id"])
        pre_path = row["pre_path"]
        post_path = row["post_path"]
        if not os.path.isabs(pre_path):
            pre_path = os.path.join(base_dir, pre_path)
        if not os.path.isabs(post_path):
            post_path = os.path.join(base_dir, post_path)

        pre_array, pre_image = read_image(pre_path)
        post_array, post_image = read_image(post_path)

        if args.spacing is not None:
            spacing = (args.spacing, args.spacing, args.spacing)
            pre_image = resample_image(pre_image, spacing)
            post_image = resample_image(post_image, spacing)
            pre_array = binarize(
                sitk.GetArrayFromImage(pre_image).astype("float32"),
                args.threshold,
            )
            post_array = binarize(
                sitk.GetArrayFromImage(post_image).astype("float32"),
                args.threshold,
            )
        else:
            pre_array = binarize(pre_array, args.threshold)
            post_array = binarize(post_array, args.threshold)

        pre_crop, post_crop = crop_pair(pre_array, post_array, args.size)

        pre_name = f"pre_{case_id}.nii.gz"
        post_name = f"post_{case_id}.nii.gz"
        write_image(pre_crop, pre_image, os.path.join(args.out_dir, pre_name))
        write_image(post_crop, post_image, os.path.join(args.out_dir, post_name))

        output_rows.append(
            {
                "case_id": case_id,
                "pre_path": pre_name,
                "post_path": post_name,
                "label": row.get("label", -1),
                "split": row.get("split", ""),
            }
        )
        print(f"Prepared {case_id}: {pre_crop.shape}")

    output_manifest = os.path.join(args.out_dir, "manifest.csv")
    pd.DataFrame(output_rows).to_csv(output_manifest, index=False)
    print(f"Saved manifest -> {output_manifest}")


if __name__ == "__main__":
    main()
