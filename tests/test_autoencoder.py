import numpy as np
import torch

from latent_visual_rl.representation.autoencoder import ConvAutoencoder
from latent_visual_rl.representation.training import weighted_reconstruction_loss


def test_shapes():
    model = ConvAutoencoder(latent_dim=12)
    recon, z = model(torch.rand(5, 1, 64, 64))
    assert recon.shape == (5, 1, 64, 64) and z.shape == (5, 12)
    assert recon.min() >= 0 and recon.max() <= 1


def test_foreground_weighting_penalises_foreground_errors_more():
    target = torch.ones(1, 1, 8, 8)
    target[..., 0, 0] = 0.0                       # one foreground pixel
    err_fg, err_bg = target.clone(), target.clone()
    err_fg[..., 0, 0] = 0.5
    err_bg[..., 5, 5] = 0.5
    assert weighted_reconstruction_loss(err_fg, target, 10) > weighted_reconstruction_loss(err_bg, target, 10)
