"""Figures: learning curves, autoencoder loss, reconstructions."""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_learning_curves(df, x="timesteps", threshold=None, title=None, out_path=None):
    """Mean +/- std across seeds. x in {'timesteps', 'total_time_s'}."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method, g in df.groupby("method"):
        curves = []
        if x == "timesteps":
            grid = np.sort(g.timesteps.unique())
            for _, gs in g.groupby("seed"):
                curves.append(np.interp(grid, gs.timesteps, gs.mean_reward))
        else:
            grid = np.linspace(0, g.groupby("seed").total_time_s.max().min(), 200)
            for _, gs in g.groupby("seed"):
                curves.append(np.interp(grid, gs.total_time_s, gs.mean_reward, left=np.nan))
        arr = np.array(curves)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu, sd = np.nanmean(arr, 0), np.nanstd(arr, 0)
        ax.plot(grid, mu, label=method)
        ax.fill_between(grid, mu - sd, mu + sd, alpha=0.2)
    if threshold:
        ax.axhline(threshold, ls="--", c="gray", lw=0.8)
    ax.set_xlabel("Environment steps" if x == "timesteps" else "Wall-clock seconds (incl. autoencoder pretraining)")
    ax.set_ylabel("Evaluation return")
    ax.set_title(title or ("Sample efficiency" if x == "timesteps" else "Wall-clock efficiency"))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return fig


def plot_autoencoder_loss(history, out_path=None):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(history["train"], label="train")
    ax.plot(history["val"], label="validation")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("weighted reconstruction loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return fig


@torch.no_grad()
def plot_reconstructions(trained, frames, device, n=8, out_path=None):
    x = torch.from_numpy(frames[:n]).permute(0, 3, 1, 2).float().div(255).to(device)
    recon, _ = trained.model.eval()(x)
    fig, axes = plt.subplots(2, n, figsize=(1.6 * n, 3.4))
    for i in range(n):
        axes[0, i].imshow(x[i, 0].cpu(), cmap="gray", vmin=0, vmax=1)
        axes[1, i].imshow(recon[i, 0].cpu(), cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    axes[0, 0].set_title("input", fontsize=8, loc="left")
    axes[1, 0].set_title("reconstruction", fontsize=8, loc="left")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return fig
