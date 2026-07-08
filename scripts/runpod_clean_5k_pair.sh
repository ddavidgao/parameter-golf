#!/usr/bin/env bash
set -euo pipefail

# Run inside a RunPod shell from the parameter-golf repo root.
#
# Clean matched rerun of the standard vs DG pair. This is the fallback when
# the old Windows checkpoints are unavailable or too confounded. It deliberately
# disables wallclock-based LR scheduling with MAX_WALLCLOCK_SECONDS=0.

if [[ ! -f "records/track_non_record_16mb/2026-03-23_DGAttention_DavidGao/train_gpt.py" ]]; then
  echo "error: run from parameter-golf repo root" >&2
  exit 2
fi

TRAIN_SHARDS="${TRAIN_SHARDS:-80}"
SEED_VALUE="${SEED_VALUE:-1337}"
ITERATIONS_VALUE="${ITERATIONS_VALUE:-5000}"
TRAIN_BATCH_TOKENS_VALUE="${TRAIN_BATCH_TOKENS_VALUE:-393216}"
TRAIN_SEQ_LEN_VALUE="${TRAIN_SEQ_LEN_VALUE:-1024}"
EVAL_STRIDE_VALUE="${EVAL_STRIDE_VALUE:-0}"
RUN_SUFFIX="${RUN_SUFFIX:-clean${ITERATIONS_VALUE}_seed${SEED_VALUE}}"
SKIP_SETUP="${SKIP_SETUP:-0}"

if [[ "${SKIP_SETUP}" != "1" ]]; then
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt

  echo "downloading sp1024 tokenizer + ${TRAIN_SHARDS} train shards + full validation split"
  python data/cached_challenge_fineweb.py --variant sp1024 --train-shards "${TRAIN_SHARDS}"
fi

mkdir -p checkpoints logs

run_one() {
  local variant="$1"
  local run_id="$2"
  echo "starting ${run_id}"
  RUN_ID="${run_id}" \
  ATTN_VARIANT="${variant}" \
  SEED="${SEED_VALUE}" \
  TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS_VALUE}" \
  TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN_VALUE}" \
  ITERATIONS="${ITERATIONS_VALUE}" \
  MAX_WALLCLOCK_SECONDS=0 \
  VAL_LOSS_EVERY=500 \
  TRAIN_LOG_EVERY=100 \
  EVAL_STRIDE="${EVAL_STRIDE_VALUE}" \
  DATA_PATH=./data/datasets/fineweb10B_sp1024 \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
  torchrun --standalone --nproc_per_node=1 records/track_non_record_16mb/2026-03-23_DGAttention_DavidGao/train_gpt.py

  cp final_model.pt "checkpoints/${run_id}.pt"
  cp final_model.int8.ptz "checkpoints/${run_id}.rdquant.ptz"
  echo "saved checkpoints/${run_id}.pt"
}

run_one standard "std_${RUN_SUFFIX}"
run_one dg "dg_${RUN_SUFFIX}"

echo "clean pair complete"
echo "raw checkpoints:"
ls -lh "checkpoints/std_${RUN_SUFFIX}.pt" "checkpoints/dg_${RUN_SUFFIX}.pt"

echo "run clean re-eval:"
echo "bash scripts/runpod_verify.sh checkpoints/std_${RUN_SUFFIX}.pt checkpoints/dg_${RUN_SUFFIX}.pt"
