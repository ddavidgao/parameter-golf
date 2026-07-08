#!/usr/bin/env bash
set -euo pipefail

# Run inside a RunPod shell from the parameter-golf repo root.
#
# Modes:
#   1. Validation-data bootstrap only:
#      bash scripts/runpod_verify.sh
#
#   2. Re-evaluate two uploaded checkpoints:
#      bash scripts/runpod_verify.sh \
#        /workspace/checkpoints/final_model_standard_seed1337_5k.pt \
#        /workspace/checkpoints/final_model_dg_linear_5k.pt
#
# This script intentionally downloads only validation shards by using
# --train-shards 0. Training runs should use their own setup command.

STD_CKPT="${1:-}"
DG_CKPT="${2:-}"
SKIP_SETUP="${SKIP_SETUP:-0}"
SKIP_NORMAL="${SKIP_NORMAL:-0}"
SKIP_SLIDING="${SKIP_SLIDING:-0}"

if [[ ! -f "data/cached_challenge_fineweb.py" ]]; then
  echo "error: run from parameter-golf repo root" >&2
  exit 2
fi

if [[ "${SKIP_SETUP}" != "1" ]]; then
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt

  echo "downloading sp1024 tokenizer + full validation split only"
  python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 0
fi

if [[ -z "${STD_CKPT}" || -z "${DG_CKPT}" ]]; then
  cat <<'MSG'
No checkpoint paths provided, so setup is complete but re-eval was skipped.

Upload/copy these raw checkpoints to the pod, then rerun:
  bash scripts/runpod_verify.sh /path/to/final_model_standard_seed1337_5k.pt /path/to/final_model_dg_linear_5k.pt
MSG
  exit 0
fi

if [[ ! -f "${STD_CKPT}" ]]; then
  echo "error: standard checkpoint not found: ${STD_CKPT}" >&2
  exit 2
fi
if [[ ! -f "${DG_CKPT}" ]]; then
  echo "error: DG checkpoint not found: ${DG_CKPT}" >&2
  exit 2
fi

extra_args=()
if [[ "${SKIP_NORMAL}" == "1" ]]; then
  extra_args+=(--skip-normal)
fi
if [[ "${SKIP_SLIDING}" == "1" ]]; then
  extra_args+=(--skip-sliding)
fi

python scripts/reeval_roundtrip.py \
  --train-script records/track_non_record_16mb/2026-03-23_DGAttention_DavidGao/train_gpt.py \
  --checkpoint "${STD_CKPT}" --variant standard --label std_5k_seed1337 \
  --checkpoint "${DG_CKPT}" --variant dg --label dg_linear_5k_seed1337 \
  --data-path ./data/datasets/fineweb10B_sp1024 \
  --tokenizer-path ./data/tokenizers/fineweb_1024_bpe.model \
  --train-seq-len 1024 \
  --val-batch-size 524288 \
  --eval-stride 64 \
  --eval-batch-seqs 32 \
  "${extra_args[@]}"
