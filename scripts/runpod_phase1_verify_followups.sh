#!/usr/bin/env bash
set -euo pipefail

# Run inside the parameter-golf repo root on the RunPod.
# Phase 1 diagnostic gate: same-process, same-mode fp/rdquant evals for all
# saved 5K checkpoints from the local-shift factorization batch.

if [[ ! -f "records/track_non_record_16mb/2026-03-23_DGAttention_DavidGao/train_gpt.py" ]]; then
  echo "error: run from parameter-golf repo root" >&2
  exit 2
fi

mkdir -p queue_results

require_ckpt() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "error: checkpoint missing: ${path}" >&2
    exit 2
  fi
}

ckpts=(
  "checkpoints/std_clean5000_seed1337.pt"
  "checkpoints/dg_clean5000_seed1337.pt"
  "checkpoints/vshift_zero_followup5000_seed1337_factorized.pt"
  "checkpoints/vshift_dginit_followup5000_seed1337_factorized.pt"
  "checkpoints/kshift_zero_followup5000_seed1337_factorized.pt"
  "checkpoints/kvshift_zero_followup5000_seed1337_factorized.pt"
)
for ckpt in "${ckpts[@]}"; do
  require_ckpt "${ckpt}"
done

out="queue_results/phase1_same_mode_verify_$(date +%Y%m%d_%H%M%S).log"

python scripts/reeval_roundtrip.py \
  --train-script records/track_non_record_16mb/2026-03-23_DGAttention_DavidGao/train_gpt.py \
  --checkpoint checkpoints/std_clean5000_seed1337.pt \
  --variant standard \
  --label standard_5k_seed1337 \
  --checkpoint checkpoints/dg_clean5000_seed1337.pt \
  --variant dg \
  --label dg_fixed_5k_seed1337 \
  --checkpoint checkpoints/vshift_zero_followup5000_seed1337_factorized.pt \
  --variant vshift_zero \
  --label vshift_zero_5k_seed1337 \
  --checkpoint checkpoints/vshift_dginit_followup5000_seed1337_factorized.pt \
  --variant vshift_dginit \
  --label vshift_dginit_5k_seed1337 \
  --checkpoint checkpoints/kshift_zero_followup5000_seed1337_factorized.pt \
  --variant kshift_zero \
  --label kshift_zero_5k_seed1337 \
  --checkpoint checkpoints/kvshift_zero_followup5000_seed1337_factorized.pt \
  --variant kvshift_zero \
  --label kvshift_zero_5k_seed1337 \
  --data-path ./data/datasets/fineweb10B_sp1024 \
  --tokenizer-path ./data/tokenizers/fineweb_1024_bpe.model \
  --train-seq-len 1024 \
  --val-batch-size 524288 \
  --eval-stride 64 \
  --eval-batch-seqs 32 \
  | tee "${out}"

echo "phase1_verify_log=${out}"
