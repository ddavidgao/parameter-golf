#!/usr/bin/env python3
"""Re-evaluate saved Parameter Golf checkpoints in one clean process.

This script is designed for the DG Attention audit. It imports a training script
by path, constructs the matching model variant, evaluates the raw checkpoint,
then applies the training script's rdquant roundtrip and evaluates again.

It reports both normal validation and sliding-window validation because the
training logs use normal eval at checkpoints but sliding eval after rdquant.
Comparing those two directly is a confound.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-script",
        default="records/track_non_record_16mb/2026-03-23_DGAttention_DavidGao/train_gpt.py",
        help="Training script whose model/eval/rdquant functions should be reused.",
    )
    parser.add_argument("--checkpoint", action="append", required=True, help="Raw state_dict checkpoint path.")
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        choices=("standard", "dg"),
        help="Attention variant for the corresponding --checkpoint. Repeat in the same order.",
    )
    parser.add_argument("--label", action="append", help="Optional label for the corresponding checkpoint.")
    parser.add_argument("--data-path", default="./data/datasets/fineweb10B_sp1024")
    parser.add_argument("--tokenizer-path", default="./data/tokenizers/fineweb_1024_bpe.model")
    parser.add_argument("--train-seq-len", type=int, default=1024)
    parser.add_argument("--val-batch-size", type=int, default=524288)
    parser.add_argument("--eval-stride", type=int, default=64)
    parser.add_argument("--eval-batch-seqs", type=int, default=32)
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=11)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--mlp-mult", type=float, default=3.0)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-normal", action="store_true")
    parser.add_argument("--skip-sliding", action="store_true")
    return parser.parse_args()


def set_import_env(args: argparse.Namespace) -> None:
    # Hyperparameters are read at import time in the training script.
    env = {
        "DATA_PATH": args.data_path,
        "TOKENIZER_PATH": args.tokenizer_path,
        "TRAIN_SEQ_LEN": str(args.train_seq_len),
        "VAL_BATCH_SIZE": str(args.val_batch_size),
        "EVAL_STRIDE": str(args.eval_stride),
        "EVAL_BATCH_SEQS": str(args.eval_batch_seqs),
        "MODEL_DIM": str(args.model_dim),
        "NUM_LAYERS": str(args.num_layers),
        "NUM_HEADS": str(args.num_heads),
        "NUM_KV_HEADS": str(args.num_kv_heads),
        "MLP_MULT": str(args.mlp_mult),
        "VOCAB_SIZE": str(args.vocab_size),
        "TIE_EMBEDDINGS": "1",
    }
    os.environ.update(env)


def import_train_script(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("pg_train_for_reeval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import training script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def clean_state_dict(module: ModuleType, checkpoint_path: Path, device: str) -> dict[str, object]:
    import torch

    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint did not contain a state_dict: {checkpoint_path}")
    cleaned = {}
    for name, tensor in state.items():
        clean_name = name.removeprefix("module.")
        cleaned[clean_name] = tensor
    return cleaned


def build_model(module: ModuleType, args: argparse.Namespace, variant: str):
    import torch

    model = module.GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
        bigram_vocab_size=getattr(module.Hyperparameters, "bigram_vocab_size", 0),
        bigram_dim=getattr(module.Hyperparameters, "bigram_dim", 128),
        attn_variant=variant,
        latent_kv_dim=getattr(module.Hyperparameters, "latent_kv_dim", 64),
        block_type=getattr(module.Hyperparameters, "block_type", "standard"),
        num_light_blocks=getattr(module.Hyperparameters, "num_light_blocks", 3),
    ).to(torch.device(args.device)).bfloat16()
    for submodule in model.modules():
        if hasattr(module, "CastedLinear") and isinstance(submodule, module.CastedLinear):
            submodule.float()
    return model


def quant_error_summary(fp_state: dict[str, object], deq_state: dict[str, object]) -> list[tuple[str, float, float, float]]:
    import torch

    groups: dict[str, list[tuple[float, float, float]]] = {}
    for name, fp in fp_state.items():
        deq = deq_state[name]
        if not getattr(fp, "is_floating_point", lambda: False)():
            continue
        diff = deq.float() - fp.float()
        mse = float(diff.square().mean().item()) if diff.numel() else 0.0
        max_abs = float(diff.abs().max().item()) if diff.numel() else 0.0
        denom = float(fp.float().square().mean().sqrt().item()) if fp.numel() else 0.0
        rel_rmse = (mse**0.5 / denom) if denom > 0 else 0.0
        if ".attn." in name:
            group = "attn"
        elif ".mlp." in name:
            group = "mlp"
        elif "tok_emb" in name:
            group = "tok_emb"
        else:
            group = "other"
        groups.setdefault(group, []).append((mse, max_abs, rel_rmse))

    rows = []
    for group, vals in sorted(groups.items()):
        mse = sum(v[0] for v in vals) / len(vals)
        max_abs = max(v[1] for v in vals)
        rel_rmse = sum(v[2] for v in vals) / len(vals)
        rows.append((group, mse, max_abs, rel_rmse))
    return rows


def main() -> None:
    args = parse_args()
    if len(args.checkpoint) != len(args.variant):
        raise SystemExit("--checkpoint and --variant must be repeated the same number of times")
    labels = args.label or []
    if labels and len(labels) != len(args.checkpoint):
        raise SystemExit("--label, if provided, must match --checkpoint count")

    set_import_env(args)
    train_script = Path(args.train_script).resolve()
    module = import_train_script(train_script)

    import sentencepiece as spm
    import torch

    device = torch.device(args.device)
    rank = 0
    world_size = 1
    grad_accum_steps = 8
    hargs = module.Hyperparameters()
    hargs.train_seq_len = args.train_seq_len
    hargs.val_batch_size = args.val_batch_size
    hargs.eval_stride = args.eval_stride
    hargs.eval_batch_seqs = args.eval_batch_seqs
    hargs.tokenizer_path = args.tokenizer_path
    hargs.data_path = args.data_path
    hargs.val_files = os.path.join(args.data_path, "fineweb_val_*.bin")

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = module.load_validation_tokens(hargs.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = module.build_sentencepiece_luts(
        sp, args.vocab_size, device
    )

    print(f"train_script={train_script}")
    print(f"val_tokens={val_tokens.numel() - 1} seq_len={args.train_seq_len}")
    print("label,variant,state,eval_mode,val_loss,val_bpb")

    for i, raw_ckpt in enumerate(args.checkpoint):
        checkpoint = Path(raw_ckpt).resolve()
        variant = args.variant[i]
        label = labels[i] if labels else checkpoint.stem

        model = build_model(module, args, variant)
        fp_state = clean_state_dict(module, checkpoint, args.device)
        model.load_state_dict(fp_state, strict=True)

        if not args.skip_normal:
            loss, bpb = module.eval_val(
                hargs,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            print(f"{label},{variant},fp,normal,{loss:.8f},{bpb:.8f}", flush=True)

        if not args.skip_sliding:
            loss, bpb = module.eval_val_sliding(
                hargs,
                model,
                rank,
                world_size,
                device,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
                stride=args.eval_stride,
                batch_seqs=args.eval_batch_seqs,
            )
            print(f"{label},{variant},fp,sliding,{loss:.8f},{bpb:.8f}", flush=True)

        q_result, q_meta = module.mixed_quantize_rdquant(fp_state, module.LAYER_QUANT_CONFIG, {"mlp", "attn"})
        deq_state = module.dequantize_rdquant(q_result, q_meta, fp_state)
        model.load_state_dict(deq_state, strict=True)

        if not args.skip_normal:
            loss, bpb = module.eval_val(
                hargs,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            print(f"{label},{variant},rdquant,normal,{loss:.8f},{bpb:.8f}", flush=True)

        if not args.skip_sliding:
            loss, bpb = module.eval_val_sliding(
                hargs,
                model,
                rank,
                world_size,
                device,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
                stride=args.eval_stride,
                batch_seqs=args.eval_batch_seqs,
            )
            print(f"{label},{variant},rdquant,sliding,{loss:.8f},{bpb:.8f}", flush=True)

        for group, mse, max_abs, rel_rmse in quant_error_summary(fp_state, deq_state):
            print(
                f"quant_error label={label} variant={variant} group={group} "
                f"mse={mse:.8e} max_abs={max_abs:.8e} rel_rmse={rel_rmse:.8e}",
                flush=True,
            )

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

