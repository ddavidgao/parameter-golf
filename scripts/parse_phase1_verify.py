#!/usr/bin/env python3
"""Parse Phase 1 same-mode verifier output and make a gate decision."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", help="Phase 1 verifier log")
    parser.add_argument("--out", default="", help="Optional JSON output path")
    parser.add_argument(
        "--min-gap",
        type=float,
        default=0.012,
        help="Minimum standard-vs-best quant-delta improvement in BPB.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for line in Path(args.log).read_text().splitlines():
        if not line or line.startswith("label,"):
            continue
        if line.startswith("quant_error "):
            continue
        parts = next(csv.reader([line]))
        if len(parts) == 6 and parts[1] not in ("variant",):
            label, variant, state, mode, loss, bpb = parts
            if state in ("fp", "rdquant") and mode in ("normal", "sliding"):
                rows.append(
                    {
                        "label": label,
                        "variant": variant,
                        "state": state,
                        "mode": mode,
                        "loss": float(loss),
                        "bpb": float(bpb),
                    }
                )

    by_key = {(r["label"], r["mode"], r["state"]): r for r in rows}
    labels = sorted({r["label"] for r in rows})
    deltas = []
    for label in labels:
        for mode in ("normal", "sliding"):
            fp = by_key.get((label, mode, "fp"))
            rdq = by_key.get((label, mode, "rdquant"))
            if fp and rdq:
                deltas.append(
                    {
                        "label": label,
                        "variant": fp["variant"],
                        "mode": mode,
                        "fp_bpb": fp["bpb"],
                        "rdq_bpb": rdq["bpb"],
                        "delta_bpb": rdq["bpb"] - fp["bpb"],
                    }
                )

    normal = [d for d in deltas if d["mode"] == "normal"]
    sliding = [d for d in deltas if d["mode"] == "sliding"]
    std_normal = next((d for d in normal if d["variant"] == "standard"), None)
    std_sliding = next((d for d in sliding if d["variant"] == "standard"), None)
    best_normal = min(normal, key=lambda d: d["delta_bpb"]) if normal else None
    best_sliding = min(sliding, key=lambda d: d["delta_bpb"]) if sliding else None

    normal_gap = (
        std_normal["delta_bpb"] - best_normal["delta_bpb"]
        if std_normal and best_normal
        else None
    )
    sliding_gap = (
        std_sliding["delta_bpb"] - best_sliding["delta_bpb"]
        if std_sliding and best_sliding
        else None
    )

    # Gate requires a same-mode normal quant advantage and non-opposite sliding evidence.
    # Sliding can be noisy/stricter, but it must not say standard is better by the same scale.
    positive = bool(
        normal_gap is not None
        and normal_gap >= args.min_gap
        and best_normal is not None
        and best_normal["variant"] in {"dg", "vshift_dginit", "kvshift_zero", "kshift_zero"}
        and (
            sliding_gap is None
            or sliding_gap >= -0.004
        )
    )

    result = {
        "positive": positive,
        "reason": (
            "quant_delta_gate_passed"
            if positive
            else "quant_delta_gate_failed_or_inconclusive"
        ),
        "min_gap": args.min_gap,
        "normal_gap_vs_standard": normal_gap,
        "sliding_gap_vs_standard": sliding_gap,
        "best_normal": best_normal,
        "best_sliding": best_sliding,
        "deltas": deltas,
    }

    print(json.dumps(result, indent=2, sort_keys=True))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
