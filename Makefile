CONFIG ?= configs/default.yaml

.PHONY: install install-dev test smoke autoencoder agents ablation analyze all demo demo-agent clean
install:      ; pip install -e .
install-dev:  ; pip install -r requirements-dev.txt && pip install -e .
test:         ; pytest -m "not slow"
smoke:        ; python -m latent_visual_rl run-all --config configs/smoke_test.yaml
autoencoder:  ; python -m latent_visual_rl train-autoencoder --config $(CONFIG)
agents:       ; python -m latent_visual_rl train-agents --config $(CONFIG)
ablation:     ; python -m latent_visual_rl ablation --config $(CONFIG)
analyze:      ; python -m latent_visual_rl analyze --config $(CONFIG)
all:          ; python -m latent_visual_rl run-all --config $(CONFIG)
demo:         ; pip install -r requirements-demo.txt && python demo.py
demo-agent:   ; python scripts/train_demo_agent.py --latent-dim 16 --steps 50000 --seed 0
clean:        ; rm -rf outputs .pytest_cache
