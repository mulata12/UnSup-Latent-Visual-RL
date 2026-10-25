"""Roll out an episode with the same observation pipeline used in training (frozen encoder + frame stack)."""
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def run_episode(cfg, ae, policy, seed):
    """policy(obs, env) -> action array. Returns (rgb frames, per-step latent z (T, d), return)."""
    from latent_visual_rl.environments.factory import make_vec_env   # lazy: needs stable-baselines3
    env = make_vec_env(cfg, "latent", seed, ae)
    env.action_space.seed(seed)
    d = ae.latent_dim
    obs = env.reset()
    frames, zs, total = [env.get_images()[0]], [obs[0, -d:]], 0.0   # newest latent sits at the end of the stack
    for _ in range(500):                                             # CartPole-v1 time limit
        obs, reward, done, _ = env.step(policy(obs, env))
        total += float(reward[0])
        if done[0]:
            break
        frames.append(env.get_images()[0])
        zs.append(obs[0, -d:])
    env.close()
    return frames, np.array(zs), total


def random_policy(obs, env):
    return np.array([env.action_space.sample()])


def agent_policy(path):
    from stable_baselines3 import DQN
    model = DQN.load(str(path), device="cpu")
    return lambda obs, env: model.predict(obs, deterministic=True)[0]


def frames_to_gif(frames, size=(300, 200), ms_per_frame=40) -> str:
    imgs = [Image.fromarray(f).resize(size) for f in frames]
    out = Path(tempfile.mkdtemp()) / "episode.gif"
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=ms_per_frame, loop=0)
    return str(out)
