"""Tests for the demo's core logic (no gradio required)."""
import numpy as np
import pandas as pd
import pytest
import torch

from latent_visual_rl.config import Config
from latent_visual_rl.demo import episodes, figures, results
from latent_visual_rl.demo.assets import load_assets
from latent_visual_rl.demo.rendering import StateRenderer, encode_decode
from latent_visual_rl.representation.autoencoder import ConvAutoencoder


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    d = tmp_path_factory.mktemp("assets")
    for dim in (4, 8):      # notebook checkpoint format (z_mean / z_std)
        torch.manual_seed(dim)
        torch.save(dict(state_dict=ConvAutoencoder(dim).state_dict(), z_mean=np.zeros(dim), z_std=np.ones(dim),
                        latent_dim=dim, history={}, pretrain_seconds=1.0), d / f"cae_z{dim}.pt")
    return load_assets(d, d / "no_results")


@pytest.fixture(scope="module")
def renderer():
    return StateRenderer(Config())


def test_load_assets_and_missing_dir(assets, tmp_path):
    assert sorted(assets.models) == [4, 8] and assets.agents == {}
    with pytest.raises(FileNotFoundError, match="prepare_github_release"):
        load_assets(tmp_path)


def test_renderer_and_codec(assets, renderer):
    f0, f1 = renderer.observe(0, 0), renderer.observe(1.5, 8)
    assert f0.shape == (64, 64, 1) and f0.dtype == np.uint8 and not np.array_equal(f0, f1)
    recon, z = encode_decode(assets.models[8], f0)
    assert recon.shape == (64, 64) and z.shape == (8,)


def test_figures_build(assets, renderer):
    fig, mse = figures.lens_figure(renderer, assets.models[4], 0.5, 3.0)
    assert mse >= 0 and fig is not None
    assert figures.sweep_figure(renderer, assets.models[4], "pole angle", 0.0) is not None
    assert figures.sweep_figure(renderer, assets.models[4], "cart position", 2.0) is not None
    assert figures.bottleneck_figure(renderer, assets.models, 0.0, 0.0) is not None


def test_random_episode(assets):
    cfg = Config()
    cfg.experiment.device = "cpu"
    frames, Z, total = episodes.run_episode(cfg, assets.models[4], episodes.random_policy, seed=0)
    assert len(frames) == len(Z) and Z.shape[1] == 4 and total >= 1
    assert episodes.frames_to_gif(frames).endswith(".gif")


def test_results_readers(tmp_path):
    (tmp_path / "raw").mkdir(); (tmp_path / "tables").mkdir()
    rows = [dict(method="pixel_dqn", seed=0, timesteps=t, mean_reward=float(t), total_time_s=float(t)) for t in (1, 2)]
    pd.DataFrame(rows).to_csv(tmp_path / "raw" / "pixel_dqn_seed0.csv", index=False)
    curves = results.load_curves(tmp_path)
    assert "1 seed" in results.seed_note(curves) and figures.curves_figure(curves, ["pixel_dqn"], "Environment steps")
    # same layout the notebook writes for the ablation table (2-row column header + index name row)
    abl = pd.DataFrame(dict(latent_dim=[4, 4, 8, 8], final_reward=[1., 3., 2., 4.], auc_reward=[1., 1., 2., 2.]))
    abl.groupby("latent_dim")[["final_reward", "auc_reward"]].agg(["mean", "std"]).round(1).to_csv(
        tmp_path / "tables" / "ablation_latent_dim.csv")
    out = results.read_ablation(tmp_path)
    assert list(out.latent_dim) == [4, 8] and "final_reward_mean" in out.columns
    assert results.load_curves(tmp_path / "missing") is None and results.read_summary(tmp_path / "missing") is None
