#!/usr/bin/env bash
# One-command CPU smoke + OpenVINO export + cosine + embed bench.
# Labels: box-only / non-target (not Xeon/Arc iron).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

echo "== [1/4] synthetic CT =="
python scripts/openvino_cpu/make_synthetic_ct.py

echo "== [2/4] torch CPU spectre embed =="
spectre embed artifacts/sample/synthetic_ct.nii.gz \
  -o artifacts/embeddings --device cpu --max-crops-per-forward 1 --overwrite

echo "== [3/4] ONNX + OpenVINO export =="
python scripts/openvino_cpu/export_ov.py

echo "== [4/4] cosine agreement + embed bench =="
python scripts/openvino_cpu/agree_cosine.py
python scripts/openvino_cpu/bench_embed.py

echo "Done. See artifacts/ov/cosine_report.json and artifacts/bench/embed_bench.json"
echo "LABEL: box-only / non-target"
