#!/usr/bin/env bash
set -euo pipefail

# Run inside a RunPod shell from the parameter-golf repo root.
# Short smoke matrix for Flash-preserving previous-token tap variants.

if [[ ! -f "records/track_non_record_16mb/2026-03-23_DGAttention_DavidGao/train_gpt.py" ]]; then
  echo "error: run from parameter-golf repo root" >&2
  exit 2
fi

SEED_VALUE="${SEED_VALUE:-1337}"
ITERATIONS_VALUE="${ITERATIONS_VALUE:-1000}"
TRAIN_BATCH_TOKENS_VALUE="${TRAIN_BATCH_TOKENS_VALUE:-393216}"
TRAIN_SEQ_LEN_VALUE="${TRAIN_SEQ_LEN_VALUE:-1024}"
RUN_SUFFIX="${RUN_SUFFIX:-smoke${ITERATIONS_VALUE}_seed${SEED_VALUE}}"
SKIP_SETUP="${SKIP_SETUP:-1}"

if [[ "${SKIP_SETUP}" != "1" ]]; then
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  python data/cached_challenge_fineweb.py --variant sp1024 --train-shards "${TRAIN_SHARDS:-25}"
fi

mkdir -p checkpoints logs queue_results

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
  EVAL_STRIDE=0 \
  DATA_PATH=./data/datasets/fineweb10B_sp1024 \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
  torchrun --standalone --nproc_per_node=1 records/track_non_record_16mb/2026-03-23_DGAttention_DavidGao/train_gpt.py

  cp final_model.pt "checkpoints/${run_id}.pt"
  cp final_model.int8.ptz "checkpoints/${run_id}.rdquant.ptz"
  echo "saved checkpoints/${run_id}.pt"
}

run_one standard "std_${RUN_SUFFIX}"
run_one dg "dg_${RUN_SUFFIX}"
run_one vshift_zero "vshift_zero_${RUN_SUFFIX}"
run_one vshift_dginit "vshift_dginit_${RUN_SUFFIX}"
run_one vshift_posinit "vshift_posinit_${RUN_SUFFIX}"
run_one kshift_zero "kshift_zero_${RUN_SUFFIX}"
run_one kvshift_zero "kvshift_zero_${RUN_SUFFIX}"

echo "smoke matrix complete"
grep -hE "attention_mode:|model_params:|step:500/|step:${ITERATIONS_VALUE}/|peak memory|Total submission size|final_rdquant_roundtrip_exact" \
  "logs/std_${RUN_SUFFIX}.txt" \
  "logs/dg_${RUN_SUFFIX}.txt" \
  "logs/vshift_zero_${RUN_SUFFIX}.txt" \
  "logs/vshift_dginit_${RUN_SUFFIX}.txt" \
  "logs/vshift_posinit_${RUN_SUFFIX}.txt" \
  "logs/kshift_zero_${RUN_SUFFIX}.txt" \
  "logs/kvshift_zero_${RUN_SUFFIX}.txt" | tee "queue_results/next_variant_${RUN_SUFFIX}_summary.txt"
