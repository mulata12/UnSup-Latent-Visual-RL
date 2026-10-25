#!/usr/bin/env python
"""(Optional) train the latent DQN agent shown in the demo's 'Live episode' tab.

Uses the same autoencoder checkpoint, observation pipeline and DQN settings as the experiment.
    python scripts/train_demo_agent.py --latent-dim 16 --steps 50000 --seed 0     # ~5-6 minutes
Writes demo_assets/latent_agent_z16.zip (a few hundred KB).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from latent_visual_rl.agents.dqn_training import build_dqn                      # noqa: E402
from latent_visual_rl.config import Config                                      # noqa: E402
from latent_visual_rl.environments.factory import make_vec_env                  # noqa: E402
from latent_visual_rl.representation.checkpoint import load_autoencoder         # noqa: E402
from latent_visual_rl.reproducibility import resolve_device, seed_everything    # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--latent-dim", type=int, default=16)
p.add_argument("--steps", type=int, default=50_000)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--assets", default="demo_assets")
a = p.parse_args()

cfg = Config()
device = resolve_device(cfg.experiment.device)
seed_everything(a.seed)
ae = load_autoencoder(Path(a.assets) / f"cae_z{a.latent_dim}.pt", device)
env = make_vec_env(cfg, "latent", a.seed, ae)
model = build_dqn(cfg, "latent", env, a.seed)
model.learn(a.steps)
out = Path(a.assets) / f"latent_agent_z{a.latent_dim}.zip"
model.save(out)
print("saved", out)
