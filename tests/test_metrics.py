import numpy as np
import pandas as pd

from latent_visual_rl.analysis.metrics import compute_per_run_metrics, summarize_methods, welch_t_tests


def _fake(method, seed, rewards, pre=0.0):
    steps = np.arange(1, len(rewards) + 1) * 1000
    return pd.DataFrame(dict(method=method, seed=seed, timesteps=steps, mean_reward=rewards,
                             rl_time_s=steps / 100.0, total_time_s=steps / 100.0 + pre))


def test_threshold_and_time_metrics():
    df = pd.concat([_fake("a", 0, [10, 50, 100, 100], pre=30), _fake("b", 0, [10, 20, 30, 40])])
    m = compute_per_run_metrics(df, success_threshold=100).set_index("method")
    assert m.loc["a", "steps_to_threshold"] == 3000
    assert m.loc["a", "time_to_threshold_s"] == 60.0            # 30 s RL + 30 s pretraining
    assert np.isnan(m.loc["b", "steps_to_threshold"])
    assert m.loc["a", "pipeline_time_total_s"] == 70.0


def test_summary_and_tests():
    df = pd.concat([_fake("a", s, [10, 50 + s, 100, 120]) for s in range(3)] +
                   [_fake("b", s, [10, 20, 30 + s, 40]) for s in range(3)])
    per_run = compute_per_run_metrics(df, 100)
    assert "runs_reaching_threshold" in summarize_methods(per_run).columns
    assert welch_t_tests(per_run, "a", "b")["final_reward"]["p"] < 0.05
