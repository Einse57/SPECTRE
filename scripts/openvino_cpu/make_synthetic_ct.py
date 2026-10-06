#!/usr/bin/env python3
"""Write a one-crop synthetic CT (.nii.gz) for CPU smoke tests.

Volume shape matches SPECTRE's default crop (128, 128, 64) so tiling yields N=1.
Values are synthetic Hounsfield Units — not clinical data.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-o", "--output", type=Path, default=Path("artifacts/sample/synthetic_ct.nii.gz"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    shape = (128, 128, 64)
    rng = np.random.default_rng(args.seed)
    vol = rng.normal(loc=40.0, scale=120.0, size=shape).astype(np.float32)
    vol = np.clip(vol, -1000, 1000)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(vol, np.eye(4)), args.output)
    print(f"wrote {args.output} shape={shape} HU[{vol.min():.1f},{vol.max():.1f}]")


if __name__ == "__main__":
    main()
