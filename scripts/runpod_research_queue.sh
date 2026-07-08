#!/usr/bin/env bash
set -euo pipefail

# One-pod DG Attention research queue.
#
# Intended use:
#   bash scripts/runpod_research_queue.sh
#
# It runs, in order:
#   1. dependency + dataset setup
#   2. attention microbench if available
#   3. 1K clean standard/DG sanity pair
#   4. same-mode FP/rdquant re-eval of the 1K pair
#   5. optional 5K clean standard/DG pair
#   6. same-mode FP/rdquant re-eval of the 5K pair
#
# Cost control:
#   RUN_FULL_5K=0 skips the expensive 5K pair.
#   TRAIN_SHARDS controls dataset train shard count. Validation is always full.

RUN_FULL_5K="${RUN_FULL_5K:-1}"
TRAIN_SHARDS="${TRAIN_SHARDS:-80}"
SEED_VALUE="${SEED_VALUE:-1337}"
TRAIN_BATCH_TOKENS_VALUE="${TRAIN_BATCH_TOKENS_VALUE:-393216}"
TRAIN_SEQ_LEN_VALUE="${TRAIN_SEQ_LEN_VALUE:-1024}"
VAL_BATCH_SIZE_VALUE="${VAL_BATCH_SIZE_VALUE:-524288}"

if [[ ! -f "data/cached_challenge_fineweb.py" ]]; then
  echo "error: run from parameter-golf repo root" >&2
  exit 2
fi

mkdir -p logs checkpoints queue_results

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log_section() {
  echo
  echo "================================================================================"
  echo "$(timestamp) $*"
  echo "================================================================================"
}

log_section "system"
nvidia-smi || true
python --version

log_section "dependencies"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

log_section "data"
python data/cached_challenge_fineweb.py --variant sp1024 --train-shards "${TRAIN_SHARDS}"

log_section "sdpa E/Ev probe"
python scripts/sdpa_ev_probe.py --iters 30 --warmup 5 | tee queue_results/sdpa_ev_probe.txt

if [[ -f "scripts/attention_microbench.py" ]]; then
  log_section "attention microbench"
  python scripts/attention_microbench.py --payload-head-dim 32 --iters 30 --warmup 5 | tee queue_results/microbench_p32.txt
  python scripts/attention_microbench.py --payload-head-dim 16 --iters 30 --warmup 5 | tee queue_results/microbench_p16.txt
  python scripts/attention_microbench.py --payload-head-dim 32 --compile --iters 30 --warmup 5 | tee queue_results/microbench_p32_compile.txt || true
  python scripts/attention_microbench.py --payload-head-dim 16 --compile --iters 30 --warmup 5 | tee queue_results/microbench_p16_compile.txt || true
else
  echo "scripts/attention_microbench.py not present; skipping microbench"
fi

log_section "clean 1K pair"
RUN_SUFFIX="clean1000_seed${SEED_VALUE}" \
ITERATIONS_VALUE=1000 \
SKIP_SETUP=1 \
TRAIN_SHARDS="${TRAIN_SHARDS}" \
SEED_VALUE="${SEED_VALUE}" \
TRAIN_BATCH_TOKENS_VALUE="${TRAIN_BATCH_TOKENS_VALUE}" \
TRAIN_SEQ_LEN_VALUE="${TRAIN_SEQ_LEN_VALUE}" \
bash scripts/runpod_clean_5k_pair.sh | tee queue_results/clean_1k_pair.log

log_section "verify 1K pair"
SKIP_SETUP=1 SKIP_SLIDING=1 bash scripts/runpod_verify.sh \
  "checkpoints/std_clean1000_seed${SEED_VALUE}.pt" \
  "checkpoints/dg_clean1000_seed${SEED_VALUE}.pt" | tee queue_results/verify_1k.log

if [[ "${RUN_FULL_5K}" != "1" ]]; then
  log_section "done after 1K because RUN_FULL_5K=${RUN_FULL_5K}"
  exit 0
fi

log_section "clean 5K pair"
RUN_SUFFIX="clean5000_seed${SEED_VALUE}" \
ITERATIONS_VALUE=5000 \
SKIP_SETUP=1 \
TRAIN_SHARDS="${TRAIN_SHARDS}" \
SEED_VALUE="${SEED_VALUE}" \
TRAIN_BATCH_TOKENS_VALUE="${TRAIN_BATCH_TOKENS_VALUE}" \
TRAIN_SEQ_LEN_VALUE="${TRAIN_SEQ_LEN_VALUE}" \
bash scripts/runpod_clean_5k_pair.sh | tee queue_results/clean_5k_pair.log

log_section "verify 5K pair"
SKIP_SETUP=1 bash scripts/runpod_verify.sh \
  "checkpoints/std_clean5000_seed${SEED_VALUE}.pt" \
  "checkpoints/dg_clean5000_seed${SEED_VALUE}.pt" | tee queue_results/verify_5k.log

log_section "queue complete"
find queue_results -maxdepth 1 -type f -print -exec tail -n 20 {} \;
