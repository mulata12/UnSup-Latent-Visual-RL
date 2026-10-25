#!/usr/bin/env bash
# Copy ONLY the small, publishable artefacts of a finished run into the repo layout used by the README and the demo.
# Heavy files (data/*.npz, SB3 model zips, the big output folders, zip archives) stay local and are git-ignored.
#
# Usage (from the repo root):  bash scripts/prepare_github_release.sh [RUN_DIR] [EXPORT_DIR]
#   RUN_DIR     folder produced by the notebook (default: full)
#   EXPORT_DIR  folder created by notebook section 11 (default: submission_results_full)
set -euo pipefail
RUN_DIR=${1:-full}
EXPORT_DIR=${2:-submission_results_full}

[ -d "$RUN_DIR/tables" ] || { echo "ERROR: $RUN_DIR/tables not found - run from the repo root and pass the run folder"; exit 1; }
mkdir -p results/tables results/figures results/raw demo_assets

cp -v "$RUN_DIR"/tables/*  results/tables/
cp -v "$RUN_DIR"/figures/* results/figures/
cp -v "$RUN_DIR"/results/*_seed*.csv results/raw/            # raw learning curves (used by the demo dashboard)
cp -v "$RUN_DIR"/run_info.json results/
[ -f "$EXPORT_DIR/RESULTS.md" ] && cp -v "$EXPORT_DIR/RESULTS.md" results/ || echo "note: no RESULTS.md found in $EXPORT_DIR"
cp -v "$RUN_DIR"/autoencoders/cae_z*.pt demo_assets/         # small trained encoders for the demo

echo; echo "Size of what will be committed:"; du -sh results demo_assets
echo "Next:  git add -A && git status   (check that full/, *.npz and big zip files are NOT listed)"
echo "Optional: python scripts/train_demo_agent.py   # adds a trained agent to the demo's 'Live episode' tab"
