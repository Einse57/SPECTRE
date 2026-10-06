#!/usr/bin/env python3
"""One-command-style bench: spectre embed latency + peak RSS (box-only / non-target).

Usage:
  python scripts/openvino_cpu/bench_embed.py
  # or via run_all.sh
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def peak_rss_kb(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except OSError:
        return 0
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", type=Path, default=Path("artifacts/sample/synthetic_ct.nii.gz"))
    ap.add_argument("-o", "--output", type=Path, default=Path("artifacts/embeddings"))
    ap.add_argument("--report", type=Path, default=Path("artifacts/bench/embed_bench.json"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    # Warmup / weight download not counted in timed repeats after first successful run.
    runs = []
    for i in range(args.repeats):
        cmd = [
            "spectre", "embed", str(args.scan),
            "-o", str(args.output),
            "--device", args.device,
            "--max-crops-per-forward", "1",
            "--overwrite",
            "--quiet",
        ]
        t0 = time.perf_counter()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        peak = 0
        while p.poll() is None:
            peak = max(peak, peak_rss_kb(p.pid))
            time.sleep(0.05)
        # final sample
        peak = max(peak, peak_rss_kb(p.pid) if p.pid else 0)
        out = p.stdout.read() if p.stdout else ""
        elapsed = time.perf_counter() - t0
        runs.append({
            "repeat": i,
            "exit": p.returncode,
            "wall_s": elapsed,
            "peak_rss_mb": peak / 1024.0,
            "stdout_tail": out[-500:],
        })
        if p.returncode != 0:
            break

    ok = [r for r in runs if r["exit"] == 0]
    report = {
        "label": "box-only / non-target (not Xeon/Arc iron)",
        "command": " ".join(cmd),
        "device": args.device,
        "scan": str(args.scan),
        "runs": runs,
        "summary": {
            "n_ok": len(ok),
            "latency_s_mean": sum(r["wall_s"] for r in ok) / len(ok) if ok else None,
            "latency_s_min": min((r["wall_s"] for r in ok), default=None),
            "peak_rss_mb_max": max((r["peak_rss_mb"] for r in ok), default=None),
        },
        "host": {
            "platform": sys.platform,
            "cpu_count": os.cpu_count(),
        },
    }
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
