#!/usr/bin/env python3
"""Cosine agreement: torch backbone tokens vs OpenVINO CPU (FP32 hint).

Also reports onnxruntime vs torch (should be ~1.0) for the ONNX path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import openvino as ov
import torch
import torch.nn as nn
from spectre import SpectreImageFeatureExtractor
from spectre.io import load_and_window


class BackboneFeatures(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, crops: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(crops)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", type=Path, default=Path("artifacts/sample/synthetic_ct.nii.gz"))
    ap.add_argument("--ir", type=Path, default=Path("artifacts/ov/spectre_vit_backbone_torch.xml"))
    ap.add_argument("--onnx", type=Path, default=Path("artifacts/ov/spectre_vit_backbone.onnx"))
    ap.add_argument("-o", "--output", type=Path, default=Path("artifacts/ov/cosine_report.json"))
    ap.add_argument("--model", default="spectre-large")
    args = ap.parse_args()

    extractor = SpectreImageFeatureExtractor.from_pretrained(args.model).eval()
    wrapper = BackboneFeatures(extractor.backbone).eval()

    crops, grid, _ = load_and_window(args.scan, crop_size=(128, 128, 64))
    if not isinstance(crops, torch.Tensor):
        crops = torch.as_tensor(crops)
    crops = crops.float()
    xnp = crops.numpy()

    with torch.inference_mode():
        torch_tokens = wrapper(crops).cpu().numpy()
        full = extractor(crops, grid_size=grid, max_crops_per_forward=1).cpu().numpy()

    report: dict = {
        "label": "box-only / non-target (not Xeon/Arc iron)",
        "scan": str(args.scan),
        "grid": list(grid),
        "torch_tokens_shape": list(torch_tokens.shape),
        "full_model_features_shape": list(full.shape),
        "note": "Agreement is on ViT backbone.forward_features tokens; full-model cls is torch reference only.",
    }

    if args.onnx.exists():
        import onnxruntime as ort

        sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
        ort_out = sess.run(None, {sess.get_inputs()[0].name: xnp})[0]
        report["ort_vs_torch"] = {
            "cosine_all": cosine(torch_tokens, ort_out),
            "cosine_cls": cosine(torch_tokens[:, 0], ort_out[:, 0]),
            "max_abs": float(np.max(np.abs(torch_tokens - ort_out))),
        }

    core = ov.Core()
    compiled = core.compile_model(
        str(args.ir), "CPU", {"INFERENCE_PRECISION_HINT": "FP32"}
    )
    ov_out = compiled({compiled.inputs[0]: xnp})[compiled.outputs[0]]
    report["openvino_cpu_fp32"] = {
        "ir": str(args.ir),
        "devices": core.available_devices,
        "cosine_all_tokens": cosine(torch_tokens, ov_out),
        "cosine_cls_token": cosine(torch_tokens[:, 0], ov_out[:, 0]),
        "cosine_mean_patch_tokens": cosine(
            torch_tokens[:, 1:].mean(1), ov_out[:, 1:].mean(1)
        ),
        "max_abs_diff": float(np.max(np.abs(torch_tokens - ov_out))),
        "mse": float(np.mean((torch_tokens - ov_out) ** 2)),
    }

    # Document default (often BF16) degradation so Xeon runners know to force FP32.
    compiled_default = core.compile_model(str(args.ir), "CPU")
    ov_def = compiled_default({compiled_default.inputs[0]: xnp})[compiled_default.outputs[0]]
    report["openvino_cpu_default_precision"] = {
        "cosine_all_tokens": cosine(torch_tokens, ov_def),
        "cosine_cls_token": cosine(torch_tokens[:, 0], ov_def[:, 0]),
        "max_abs_diff": float(np.max(np.abs(torch_tokens - ov_def))),
        "warning": "Default OV CPU precision may use BF16 and drop cosine << 1; use FP32 hint.",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
