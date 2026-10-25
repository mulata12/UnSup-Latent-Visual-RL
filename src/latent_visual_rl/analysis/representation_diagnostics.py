"""Quality checks of the learned latent space."""
from latent_visual_rl.representation.checkpoint import encode_frames

STATE_NAMES = ["cart_pos", "cart_vel", "pole_angle", "pole_ang_vel"]


def linear_probe_r2(trained, frames, states, device, seed=0) -> dict:
    """R^2 of a ridge regression from SINGLE-frame z to the true physical state (diagnostic only).

    Velocities are not observable from one frame, so low R^2 there is expected; this motivates
    frame stacking in both pipelines.
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    z = encode_frames(trained, frames, device)
    z_tr, z_te, s_tr, s_te = train_test_split(z, states, test_size=0.2, random_state=seed)
    return {name: round(float(r2_score(s_te[:, i], Ridge(1.0).fit(z_tr, s_tr[:, i]).predict(z_te))), 3)
            for i, name in enumerate(STATE_NAMES)}
