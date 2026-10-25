"""Widened initial-state distribution used ONLY when collecting autoencoder training data."""
import numpy as np

# CartPole state = [cart_pos, cart_vel, pole_angle, pole_ang_vel]
STATE_LOW = np.array([-2.0, -1.0, -0.15, -1.0])
STATE_HIGH = np.array([2.0, 1.0, 0.15, 1.0])


def sample_initial_state(rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(STATE_LOW, STATE_HIGH)
