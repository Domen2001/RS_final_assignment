#!/usr/bin/env bash

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1
export RS_SCORES_DIR="${RS_SCORES_DIR:-/local/$USER/rs_scores}"

MODELS="ease als itemknn popularity bpr multvae content lightgcn bert4rec recency"

# Ensure scores are cached for both folds.
echo "STAGE 1: cache Fold B + Fold A scores"
# python src/train_all.py --fold b --models $MODELS
# python src/train_all.py --fold a --models $MODELS

# Generate + rank candidates (subsets × RRF k × tuned weights).
echo ""; echo "STAGE 2: sweep candidates"
python src/sweep.py --models $MODELS --rrf_ks 20 40 60 100 --n_trials 200

echo ""
echo "Done. Candidate CSVs in data/sweep/. Ranking in data/sweep/_ranking.csv"

