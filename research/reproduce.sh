#!/usr/bin/env bash
# One command to reproduce the offline results: embeddings -> experiments 1, 2, 5, 6 -> figures + summary.
# Run from the repo root:  bash research/reproduce.sh
# The first run downloads both datasets and the model weights, and extracts embeddings for
# every image, which takes a while. Pass --limit N to extract_embeddings.py by hand for a quick subset instead.
set -euo pipefail
PY="${PYTHON:-python3}"
$PY research/extract_embeddings.py
$PY research/exp1_shots.py
$PY research/exp2_heads.py
$PY research/exp5_forgetting.py
$PY research/exp6_unknown.py
$PY research/make_figures.py
echo "Done: see results/summary.md and results/figures/"
