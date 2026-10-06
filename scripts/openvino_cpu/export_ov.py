#!/usr/bin/env python3
"""Export SPECTRE ViT backbone -> ONNX -> OpenVINO IR (CPU / FP32).

Exports backbone.forward_features for a fixed crop (1, 1, 128, 128, 64).
Requires INFERENCE_PRECISION_HINT=FP32 at compile time for torch agreement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import openvino as ov
import torch
import torch.nn as nn
from spectre import SpectreImageFeatureExtractor


class BackboneFeatures(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, crops: torch.Tensor) -> torch.Tensor:
        # crops: (N, C, 128, 128, 64) -> (N, T, F)
        return self.backbone.forward_features(crops)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--outdir", type=Path, default=Path("artifacts/ov"))
    ap.add_argument("--model", default="spectre-large")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    extractor = SpectreImageFeatureExtractor.from_pretrained(args.model)
    extractor.eval()
    wrapper = BackboneFeatures(extractor.backbone).eval()
    dummy = torch.randn(1, 1, 128, 128, 64)

    onnx_path = args.outdir / "spectre_vit_backbone.onnx"
    print(f"exporting ONNX (dynamo) -> {onnx_path}")
    torch.onnx.export(
        wrapper,
        dummy,
        str(onnx_path),
        input_names=["crops"],
        output_names=["tokens"],
        dynamo=True,
        opset_version=18,
    )

    ir_xml = args.outdir / "spectre_vit_backbone.xml"
    print(f"converting OpenVINO IR (FP32 weights) -> {ir_xml}")
    ov_model = ov.convert_model(str(onnx_path))
    ov.save_model(ov_model, str(ir_xml), compress_to_fp16=False)

    # Also keep a torch-direct IR (often matches FP32 path cleanly).
    ir_torch = args.outdir / "spectre_vit_backbone_torch.xml"
    print(f"converting torch -> OpenVINO IR -> {ir_torch}")
    ov_torch = ov.convert_model(wrapper, example_input=dummy)
    ov.save_model(ov_torch, str(ir_torch), compress_to_fp16=False)

    meta = {
        "onnx": str(onnx_path),
        "ir_onnx": str(ir_xml),
        "ir_torch": str(ir_torch),
        "input": "(N, 1, 128, 128, 64)",
        "output": "backbone.forward_features -> (N, T, F)",
        "compile_hint": "INFERENCE_PRECISION_HINT=FP32 required for torch agreement",
        "label": "box-only / non-target until run on Xeon/Arc",
    }
    (args.outdir / "export_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
