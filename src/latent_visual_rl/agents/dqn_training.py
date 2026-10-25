"""Stage 2: DQN training (pixel baseline or latent) with wall-clock aware evaluation logging."""
import time
from pathlib import Path

import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from latent_visual_rl.environments.factory import make_vec_env
from latent_visual_rl.reproducibility import seed_everything


def build_dqn(cfg, observation_mode, env, seed):
    """Create a Stable-Baselines3 DQN using hyper-parameters from the config."""
    if observation_mode == "pixel":
        params = dict(cfg.agent.dqn_pixel)
        policy, policy_kwargs = "CnnPolicy", None
    else:
        params = dict(cfg.agent.dqn_latent)
        policy = "MlpPolicy"
        policy_kwargs = dict(net_arch=list(params.pop("net_arch", [256, 256])))
    return DQN(policy, env, policy_kwargs=policy_kwargs, seed=seed, device=cfg.experiment.device,
               verbose=0, **params)


class TimedEvaluationCallback(BaseCallback):
    """Periodic greedy evaluation. Logs steps, reward and training wall-clock (evaluation time is
    excluded). `pretrain_seconds` is added for the latent pipeline so total time is comparable."""

    def __init__(self, eval_env, cfg, method, seed, csv_path, pretrain_seconds=0.0, verbose=1):
        super().__init__(verbose)
        self.eval_env, self.cfg, self.method, self.seed = eval_env, cfg, method, seed
        self.csv_path, self.pretrain_seconds = Path(csv_path), pretrain_seconds
        self.rows, self.eval_seconds = [], 0.0

    def _on_training_start(self):
        self.t0 = time.time()

    def _on_step(self):
        if self.num_timesteps % self.cfg.agent.eval_every_steps == 0:
            t = time.time()
            mean, std = evaluate_policy(self.model, self.eval_env, n_eval_episodes=self.cfg.agent.eval_episodes,
                                        deterministic=True)
            self.eval_seconds += time.time() - t
            rl_time = time.time() - self.t0 - self.eval_seconds
            self.rows.append(dict(method=self.method, seed=self.seed, timesteps=self.num_timesteps,
                                  mean_reward=mean, std_reward=std, rl_time_s=rl_time,
                                  pretrain_time_s=self.pretrain_seconds, total_time_s=rl_time + self.pretrain_seconds))
            pd.DataFrame(self.rows).to_csv(self.csv_path, index=False)  # incremental save
            if self.verbose:
                print(f"  {self.method} seed={self.seed} step={self.num_timesteps:>7d} "
                      f"reward={mean:7.1f} rl_time={rl_time:6.0f}s")
        return True


def train_agent_run(cfg, method, seed, trained_autoencoder=None, pretrain_seconds=0.0, force=False):
    """Train one (method, seed). Resumable: skipped if its result CSV already exists."""
    mode = "pixel" if method.startswith("pixel") else "latent"
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    csv = cfg.results_dir / f"{method}_seed{seed}.csv"
    if csv.exists() and not force:
        print(f"[skip] {csv.name} already exists")
        return pd.read_csv(csv)
    seed_everything(seed)
    env = make_vec_env(cfg, mode, seed, trained_autoencoder)
    eval_env = make_vec_env(cfg, mode, seed + 10_000, trained_autoencoder)
    model = build_dqn(cfg, mode, env, seed)
    callback = TimedEvaluationCallback(eval_env, cfg, method, seed, csv,
                                       pretrain_seconds if mode == "latent" else 0.0)
    model.learn(cfg.agent.total_timesteps, callback=callback)
    model.save(cfg.results_dir / f"{method}_seed{seed}_model")
    env.close()
    eval_env.close()
    return pd.DataFrame(callback.rows)
