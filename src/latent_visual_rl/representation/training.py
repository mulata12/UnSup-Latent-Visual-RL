"""Unsupervised autoencoder training loop."""
import time

import numpy as np
import torch

from latent_visual_rl.reproducibility import seed_everything
from latent_visual_rl.representation.autoencoder import ConvAutoencoder
from latent_visual_rl.representation.checkpoint import TrainedAutoencoder, frames_to_tensor


def weighted_reconstruction_loss(recon, target, foreground_weight):
    """MSE with extra weight on non-background pixels so the thin pole is not ignored."""
    weights = 1.0 + foreground_weight * (target < 0.95).float()
    return (weights * (recon - target) ** 2).mean()


def train_autoencoder(frames, cfg, device, log=print):
    """Train on unlabelled frames. Returns (TrainedAutoencoder, history dict, seconds)."""
    a = cfg.autoencoder
    seed_everything(a.data_seed)
    t0 = time.time()
    data = frames_to_tensor(frames, device)
    order = torch.randperm(len(data), device=device)
    n_val = int(len(data) * a.validation_fraction)
    val, train = data[order[:n_val]], data[order[n_val:]]
    model = ConvAutoencoder(a.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=a.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, a.epochs)
    history = {"train": [], "val": []}
    for epoch in range(a.epochs):
        model.train()
        perm = torch.randperm(len(train), device=device)
        batch_losses = []
        for i in range(0, len(train), a.batch_size):
            x = train[perm[i:i + a.batch_size]].float() / 255.0
            recon, _ = model(x)
            loss = weighted_reconstruction_loss(recon, x, a.foreground_loss_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        scheduler.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(np.mean([weighted_reconstruction_loss(model(v.float() / 255.0)[0], v.float() / 255.0,
                                                                   a.foreground_loss_weight).item()
                                      for v in val.split(a.batch_size)]))
        history["train"].append(float(np.mean(batch_losses)))
        history["val"].append(val_loss)
        log(f"[autoencoder z={a.latent_dim}] epoch {epoch + 1:02d}/{a.epochs} "
            f"train={history['train'][-1]:.5f} val={val_loss:.5f}")
    model.eval()
    with torch.no_grad():
        z = torch.cat([model.encoder(t.float() / 255.0) for t in train.split(1024)]).cpu().numpy()
    trained = TrainedAutoencoder(model, z.mean(0), z.std(0) + 1e-6, a.latent_dim).freeze()
    return trained, history, time.time() - t0
