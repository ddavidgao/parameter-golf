#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PANDOC_BIN="${PANDOC_BIN:-$(command -v pandoc || true)}"
TECTONIC_BIN="${TECTONIC_BIN:-$(command -v tectonic || true)}"

if [[ -z "$PANDOC_BIN" ]]; then
  echo "pandoc not found; set PANDOC_BIN=/path/to/pandoc" >&2
  exit 1
fi
if [[ -z "$TECTONIC_BIN" ]]; then
  echo "tectonic not found; set TECTONIC_BIN=/path/to/tectonic" >&2
  exit 1
fi

if python3 -c 'import matplotlib' >/dev/null 2>&1; then
  python3 "$ROOT/scripts/generate_int4_paper_figures.py"
elif command -v uv >/dev/null 2>&1; then
  uv run --with matplotlib python "$ROOT/scripts/generate_int4_paper_figures.py"
else
  echo "matplotlib is unavailable and uv was not found; cannot regenerate figures" >&2
  exit 1
fi

python3 "$ROOT/scripts/generate_int4_latex.py" --pandoc "$PANDOC_BIN"
mkdir -p "$ROOT/output/pdf"
(
  cd "$ROOT/paper"
  "$TECTONIC_BIN" int4_attention_manuscript.tex \
    --outdir "$ROOT/output/pdf"
)

echo "$ROOT/output/pdf/int4_attention_manuscript.pdf"
