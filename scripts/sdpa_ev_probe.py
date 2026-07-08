#!/usr/bin/env python3
"""Probe SDPA support for Q/K dim != V dim.

DG-LP only matters if a narrow payload can stay on a fused SDPA path instead of
materializing full [B, H, T, T] attention weights. This script gives a quick
yes/no plus rough timing for the installed torch/CUDA stack.
"""

from __future__ import annotations

import argparse
import contextlib
import time

import torch
import torch.nn.functional as F


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def backend_context(name: str):
    if name == "default":
        return contextlib.nullcontext()
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except Exception:
        return contextlib.nullcontext()
    mapping = {
        "flash": SDPBackend.FLASH_ATTENTION,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
        "math": SDPBackend.MATH,
    }
    return sdpa_kernel(mapping[name])


def run_case(args: argparse.Namespace, payload_dim: int, backend: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    q = torch.randn(args.batch, args.heads, args.seq_len, args.head_dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(args.batch, args.kv_heads, args.seq_len, args.head_dim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(args.batch, args.kv_heads, args.seq_len, payload_dim, device=device, dtype=dtype, requires_grad=True)
    try:
        with backend_context(backend):
            for _ in range(args.warmup):
                for t in (q, k, v):
                    t.grad = None
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=args.heads != args.kv_heads)
                y.float().square().mean().backward()
            sync()
            torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
            start = time.perf_counter()
            for _ in range(args.iters):
                for t in (q, k, v):
                    t.grad = None
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=args.heads != args.kv_heads)
                y.float().square().mean().backward()
            sync()
        peak = torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0
        print(
            f"ok backend={backend} payload_dim={payload_dim} output={tuple(y.shape)} "
            f"ms={(time.perf_counter() - start) * 1000 / args.iters:.2f} peak_mib={peak:.1f}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"fail backend={backend} payload_dim={payload_dim} error={type(exc).__name__}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--payload-dims", default="64,48,32,16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()
    print(f"torch={torch.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    for backend in ("default", "flash", "efficient", "math"):
        for payload_dim in [int(x) for x in args.payload_dims.split(",") if x]:
            run_case(args, payload_dim, backend)


if __name__ == "__main__":
    main()

