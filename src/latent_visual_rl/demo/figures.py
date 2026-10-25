"""Matplotlib figure builders (use Figure directly, not pyplot, so nothing leaks in a long-running server)."""
import numpy as np
from matplotlib.figure import Figure

from latent_visual_rl.demo.rendering import LIMIT_CART_POS, LIMIT_POLE_DEG, encode_decode


def lens_figure(renderer, ae, cart_pos, pole_deg):
    frame = renderer.observe(cart_pos, pole_deg)
    recon, z = encode_decode(ae, frame)
    x = frame[..., 0] / 255.0
    mse = float(np.mean((x - recon) ** 2))
    fig = Figure(figsize=(15, 3.4))
    axs = fig.subplots(1, 4, gridspec_kw=dict(width_ratios=[1, 1, 1, 1.8]))
    for ax, img, title in zip(axs[:3], [x, recon, np.abs(x - recon)],
                              ["input (what the agent sees)", f"reconstruction from z ({ae.latent_dim} numbers)", "absolute error"]):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    axs[3].bar(range(ae.latent_dim), z, color=np.where(z >= 0, "#2a7fbb", "#d9534f"))
    axs[3].set_title("latent vector z (standardised)", fontsize=9)
    axs[3].set_xlabel("latent dimension")
    axs[3].set_ylim(-4, 4)
    axs[3].grid(alpha=0.3)
    fig.tight_layout()
    return fig, mse


def sweep_figure(renderer, ae, vary, other_value):
    """Sweep the pole angle ('pole angle') or the cart position ('cart position') and plot every latent dimension."""
    if vary == "pole angle":
        grid = np.linspace(-LIMIT_POLE_DEG, LIMIT_POLE_DEG, 49)
        Z = [encode_decode(ae, renderer.observe(other_value, g))[1] for g in grid]
        xlabel = "pole angle (degrees)"
    else:
        grid = np.linspace(-LIMIT_CART_POS, LIMIT_CART_POS, 49)
        Z = [encode_decode(ae, renderer.observe(g, other_value))[1] for g in grid]
        xlabel = "cart position"
    Z = np.array(Z)
    fig = Figure(figsize=(9, 3.8))
    ax = fig.subplots()
    im = ax.imshow(Z.T, aspect="auto", cmap="coolwarm", vmin=-3, vmax=3,
                   extent=[grid[0], grid[-1], ae.latent_dim - 0.5, -0.5])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("latent dimension")
    ax.set_title(f"How z responds when the {vary} changes (z size {ae.latent_dim})")
    fig.colorbar(im, ax=ax, label="standardised z")
    fig.tight_layout()
    return fig


def bottleneck_figure(renderer, models, cart_pos, pole_deg):
    frame = renderer.observe(cart_pos, pole_deg)
    x = frame[..., 0] / 255.0
    dims = sorted(models)
    fig = Figure(figsize=(2.6 * (len(dims) + 1), 3.2))
    axs = fig.subplots(1, len(dims) + 1)
    axs[0].imshow(x, cmap="gray", vmin=0, vmax=1)
    axs[0].set_title("input\n(4096 pixels)", fontsize=9)
    axs[0].axis("off")
    for ax, d in zip(axs[1:], dims):
        recon, _ = encode_decode(models[d], frame)
        ax.imshow(recon, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"z = {d}\nMSE {np.mean((x - recon) ** 2):.4f}", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    return fig


def latent_trajectory_figure(Z, title="Latent vector over the episode"):
    fig = Figure(figsize=(9, 3.4))
    ax = fig.subplots()
    im = ax.imshow(Z.T, aspect="auto", cmap="coolwarm", vmin=-3, vmax=3)
    ax.set_xlabel("time step")
    ax.set_ylabel("latent dimension")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="standardised z")
    fig.tight_layout()
    return fig


def curves_figure(curves, methods, x_axis):
    """Learning curves (mean +/- std over seeds when more than one seed exists)."""
    col = "timesteps" if x_axis.startswith("Environment") else "total_time_s"
    fig = Figure(figsize=(8.5, 4.4))
    ax = fig.subplots()
    for m in methods:
        g = curves[curves.method == m]
        agg = g.groupby("timesteps").agg(x=(col, "mean"), mu=("mean_reward", "mean"),
                                         sd=("mean_reward", "std")).fillna(0)
        ax.plot(agg.x, agg.mu, marker="o", ms=3, label=m)
        if g.seed.nunique() > 1:
            ax.fill_between(agg.x, agg.mu - agg.sd, agg.mu + agg.sd, alpha=0.2)
    ax.set_xlabel("environment steps" if col == "timesteps" else "wall-clock seconds (incl. pre-training)")
    ax.set_ylabel("evaluation return")
    ax.grid(alpha=0.3)
    if methods:
        ax.legend()
    fig.tight_layout()
    return fig
