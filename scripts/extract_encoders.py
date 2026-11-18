#!/usr/bin/env python
"""Extract the two 3D ResNet-18 encoder state dicts from a Stage II checkpoint.

The multimodal Stage II checkpoint contains the pretrained encoder weights
used by the classifier.  This utility writes them as standalone ``net.*``
state dicts so they can be passed to ``train_stage2.py`` via
``--encoder1_ckpt`` and ``--encoder2_ckpt``.

Example
-------
python scripts/extract_encoders.py \
    --stage2_ckpt checkpoints/ARPNetStageII_multimodal.pth \
    --out_dir checkpoints/encoders
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Stage II encoders.")
    parser.add_argument("--stage2_ckpt", required=True)
    parser.add_argument("--out_dir", default="checkpoints/encoders")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.stage2_ckpt, map_location="cpu")
    state_dict = payload["state_dict"] if "state_dict" in payload else payload

    os.makedirs(args.out_dir, exist_ok=True)
    for encoder_name in ("encoder1", "encoder2"):
        prefix = f"{encoder_name}."
        encoder_state = {
            key[len(prefix):]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if not encoder_state:
            raise ValueError(
                f"No keys starting with '{prefix}' in {args.stage2_ckpt}."
            )
        output_path = os.path.join(args.out_dir, f"{encoder_name}.pth")
        torch.save(encoder_state, output_path)
        print(f"Saved {encoder_name} -> {output_path}")


if __name__ == "__main__":
    main()
