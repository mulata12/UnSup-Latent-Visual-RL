#!/usr/bin/env bash
# Full reproduction: main experiment + latent-size ablation + analysis.
set -euo pipefail
CONFIG=${1:-configs/default.yaml}
python -m latent_visual_rl train-autoencoder --config "$CONFIG"
python -m latent_visual_rl train-agents      --config "$CONFIG"
python -m latent_visual_rl ablation          --config "$CONFIG"
python -m latent_visual_rl analyze           --config "$CONFIG"
