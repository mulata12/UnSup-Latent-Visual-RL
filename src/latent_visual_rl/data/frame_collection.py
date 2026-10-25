"""Unlabelled frame collection for autoencoder training (stage 1 data)."""
import time

import gymnasium as gym
import numpy as np

from latent_visual_rl.environments.observation_wrappers import preprocess_frame
from latent_visual_rl.environments.state_sampling import sample_initial_state


def collect_random_policy_frames(cfg):
    """Return (frames uint8 [N,H,W,1], states float32 [N,4], elapsed_seconds).

    Frames are the ONLY training signal for the autoencoder. Physical states are stored purely
    for post-hoc linear probing of the latent space and are never used for training.
    """
    a, e = cfg.autoencoder, cfg.env
    rng = np.random.default_rng(a.data_seed)
    env = gym.make(e.env_id, render_mode="rgb_array")
    env.reset(seed=a.data_seed)
    env.action_space.seed(a.data_seed)
    has_state = hasattr(env.unwrapped, "state")
    widen = a.diverse_initial_states and has_state
    frames, states, done = [], [], True
    t0 = time.time()
    while len(frames) < a.num_frames:
        if done:
            env.reset()
            if widen:
                env.unwrapped.state = sample_initial_state(rng)
        frames.append(preprocess_frame(env.render(), e.image_size, e.crop_rows))
        states.append(np.array(env.unwrapped.state, dtype=np.float32) if has_state else np.zeros(4, np.float32))
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        done = terminated or truncated
    env.close()
    return np.stack(frames), np.stack(states), time.time() - t0
