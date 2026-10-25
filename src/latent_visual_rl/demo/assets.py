"""Locate and load the pretrained artefacts used by the demo."""
from dataclasses import dataclass, field
from pathlib import Path

from latent_visual_rl.representation.checkpoint import TrainedAutoencoder, load_autoencoder


@dataclass
class DemoAssets:
    models: dict                       # latent_dim -> TrainedAutoencoder (CPU, frozen)
    agents: dict = field(default_factory=dict)   # latent_dim -> path of saved SB3 DQN agent (optional)
    results_dir: Path = Path("results")


def load_assets(assets_dir="demo_assets", results_dir="results") -> DemoAssets:
    assets_dir, results_dir = Path(assets_dir), Path(results_dir)
    ckpts = sorted(assets_dir.glob("cae_z*.pt"))
    if not ckpts:
        raise FileNotFoundError(
            f"No autoencoder checkpoints (cae_z*.pt) found in '{assets_dir}'.\n"
            "Run:  bash scripts/prepare_github_release.sh   (copies them from your finished run)")
    models = {int(p.stem.split("_z")[1]): load_autoencoder(p, "cpu") for p in ckpts}
    agents = {int(p.stem.split("_z")[1]): p for p in sorted(assets_dir.glob("latent_agent_z*.zip"))}
    return DemoAssets(models=models, agents=agents, results_dir=results_dir)
