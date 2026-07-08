#!/usr/bin/env bash
set -euo pipefail

# Run on the RunPod from /workspace/parameter-golf.
# This is intentionally diagnostic-only: it runs same-mode verification,
# parses the quant-delta gate, writes markers, then stops the pod if possible.

cd /workspace/parameter-golf

mkdir -p queue_results
rm -f /workspace/dg_queue_done /workspace/dg_queue_failed /workspace/phase1_positive /workspace/phase1_negative

git fetch origin feat/dg-conv2-smokes
git checkout feat/dg-conv2-smokes
git reset --hard origin/feat/dg-conv2-smokes

log="queue_results/phase1_supervised_$(date +%Y%m%d_%H%M%S).log"
decision="queue_results/phase1_decision_$(date +%Y%m%d_%H%M%S).json"

{
  echo "phase1_supervised_start $(date -Is)"
  bash scripts/runpod_phase1_verify_followups.sh
  latest_verify="$(ls -t queue_results/phase1_same_mode_verify_*.log | head -n 1)"
  echo "phase1_verify_log=${latest_verify}"
  python scripts/parse_phase1_verify.py "${latest_verify}" --out "${decision}"
  echo "phase1_decision_json=${decision}"
  if python - "${decision}" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
raise SystemExit(0 if data.get("positive") else 1)
PY
  then
    touch /workspace/phase1_positive
    echo "phase1_gate=positive"
  else
    touch /workspace/phase1_negative
    echo "phase1_gate=negative_or_inconclusive"
  fi
  touch /workspace/dg_queue_done
  echo "phase1_supervised_done $(date -Is)"
} 2>&1 | tee "${log}"

if command -v runpodctl >/dev/null 2>&1 && [[ -n "${RUNPOD_API_KEY:-}" ]]; then
  echo "stopping pod through runpodctl"
  runpodctl pod stop "${RUNPOD_POD_ID:-h547aj1ed1r6oz}" || true
fi

echo "phase1_supervised_log=${log}"
