# OpenVINO CPU path for SPECTRE ViT (embeddings only)

**License / NC:** Code is MIT; pretrained weights are **CC-BY-NC-SA**. Use weights for **internal benchmark only** — no product binary, no commercial redistribution of weight-derived artifacts.

## What this does

1. Synthetic one-crop CT (`128×128×64`) smoke input.
2. `spectre embed --device cpu` (torch CPU reference).
3. Export ViT **backbone** `forward_features` → ONNX → OpenVINO IR.
4. Cosine agreement vs torch (require `INFERENCE_PRECISION_HINT=FP32`).
5. Latency + peak RSS bench for `spectre embed` (**box-only / non-target**).

Arc / Xeon iron is **out of scope for this box**; re-run the same scripts on target hardware.

## One command

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip wheel
pip install torch==2.10.0+cpu torchvision==0.25.0+cpu \
  --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-cpu-openvino.txt
pip install -e ".[inference]"
bash scripts/openvino_cpu/run_all.sh
```

## Precision note

OpenVINO CPU default precision can use BF16 and drop cosine well below 1.0 on this RoPE ViT.
Always compile with `INFERENCE_PRECISION_HINT=FP32` for numerical agreement (see `agree_cosine.py`).
