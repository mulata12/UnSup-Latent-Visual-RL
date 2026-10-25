"""Pipeline stages. Each stage is cached on disk, idempotent and individually callable."""
import json

import numpy as np

from latent_visual_rl.agents.dqn_training import train_agent_run
from latent_visual_rl.data.frame_collection import collect_random_policy_frames
from latent_visual_rl.reproducibility import collect_environment_info, resolve_device
from latent_visual_rl.representation.checkpoint import load_autoencoder, save_autoencoder
from latent_visual_rl.representation.training import train_autoencoder


def prepare_run_directory(cfg):
    """Create the run folder and persist the exact config + environment info."""
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(cfg.run_dir / "config.yaml")
    with open(cfg.run_dir / "environment.json", "w") as f:
        json.dump(collect_environment_info(), f, indent=2)


def ensure_frame_dataset(cfg):
    """Stage 1a: collect (or load cached) unlabelled frames. Returns (frames, states, seconds)."""
    path = cfg.frame_dataset_path
    if path.exists():
        d = np.load(path)
        return d["frames"], d["states"], float(d["seconds"])
    path.parent.mkdir(parents=True, exist_ok=True)
    frames, states, seconds = collect_random_policy_frames(cfg)
    np.savez_compressed(path, frames=frames, states=states, seconds=seconds)
    return frames, states, seconds


def ensure_autoencoder(cfg):
    """Stage 1b: train (or load cached) autoencoder. Returns (TrainedAutoencoder, metadata dict)."""
    device = resolve_device(cfg.experiment.device)
    directory = cfg.autoencoder_dir()
    ckpt, meta_path = directory / "autoencoder.pt", directory / "metadata.json"
    if ckpt.exists():
        return load_autoencoder(ckpt, device), json.load(open(meta_path))
    frames, _, collect_s = ensure_frame_dataset(cfg)
    trained, history, train_s = train_autoencoder(frames, cfg, device)
    save_autoencoder(trained, ckpt)
    meta = dict(collect_seconds=collect_s, train_seconds=train_s, pretrain_seconds=collect_s + train_s,
                final_val_loss=history["val"][-1], history=history)
    json.dump(meta, open(meta_path, "w"), indent=2)
    return trained, meta


def run_agent_stage(cfg, methods=("pixel", "latent")):
    """Stage 2: train every requested method for every seed."""
    for method in methods:
        if method == "pixel":
            for seed in cfg.experiment.seeds:
                train_agent_run(cfg, cfg.pixel_method, seed)
        elif method == "latent":
            trained, meta = ensure_autoencoder(cfg)
            for seed in cfg.experiment.seeds:
                train_agent_run(cfg, cfg.latent_method(), seed, trained, meta["pretrain_seconds"])
        else:
            raise ValueError(f"unknown method '{method}' (use 'pixel' or 'latent')")


def run_latent_dim_ablation(cfg, latent_dims=(4, 8, 32, 64)):
    """Ablation over bottleneck size; results are written next to the main runs."""
    for dim in latent_dims:
        c = cfg.with_latent_dim(dim)
        trained, meta = ensure_autoencoder(c)
        for seed in c.experiment.seeds:
            train_agent_run(c, c.latent_method(), seed, trained, meta["pretrain_seconds"])
