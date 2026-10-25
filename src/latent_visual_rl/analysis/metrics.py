"""Per-run metrics, aggregation and significance tests."""
import glob
from pathlib import Path

import numpy as np
import pandas as pd

_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def load_results(results_dir) -> pd.DataFrame:
    files = sorted(glob.glob(str(Path(results_dir) / "*_seed*.csv")))
    if not files:
        raise FileNotFoundError(f"no result CSVs in {results_dir}; run training first")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def compute_per_run_metrics(df: pd.DataFrame, success_threshold: float) -> pd.DataFrame:
    """One row per (method, seed).

    final_reward         mean of last 3 evaluations
    auc_reward           normalised area under the learning curve (sample efficiency)
    steps_to_threshold   first evaluated step with reward >= threshold (NaN = never)
    time_to_threshold_s  same, in wall-clock (includes autoencoder pretraining)
    """
    rows = []
    for (method, seed), g in df.groupby(["method", "seed"]):
        g = g.sort_values("timesteps")
        hit = g[g.mean_reward >= success_threshold]
        span = max(g.timesteps.max() - g.timesteps.min(), 1)
        rows.append(dict(
            method=method, seed=seed,
            final_reward=g.mean_reward.tail(3).mean(),
            best_reward=g.mean_reward.max(),
            auc_reward=_trapezoid(g.mean_reward, g.timesteps) / span,
            steps_to_threshold=hit.timesteps.iloc[0] if len(hit) else np.nan,
            time_to_threshold_s=hit.total_time_s.iloc[0] if len(hit) else np.nan,
            rl_time_total_s=g.rl_time_s.iloc[-1],
            pipeline_time_total_s=g.total_time_s.iloc[-1]))
    return pd.DataFrame(rows)


def summarize_methods(per_run: pd.DataFrame) -> pd.DataFrame:
    metrics = per_run.drop(columns="seed")
    agg = metrics.groupby("method").agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg["runs_reaching_threshold"] = per_run.groupby("method").steps_to_threshold.apply(
        lambda x: f"{x.notna().sum()}/{len(x)}")
    return agg.round(1)


def welch_t_tests(per_run, method_a, method_b,
                  metrics=("final_reward", "auc_reward", "pipeline_time_total_s")) -> dict:
    from scipy import stats
    out = {}
    for m in metrics:
        a = per_run[per_run.method == method_a][m].dropna()
        b = per_run[per_run.method == method_b][m].dropna()
        if len(a) > 1 and len(b) > 1:
            t, p = stats.ttest_ind(a, b, equal_var=False)
            out[m] = dict(t=round(float(t), 3), p=round(float(p), 4), n_a=len(a), n_b=len(b))
    return out


def latent_dim_ablation_table(per_run: pd.DataFrame) -> pd.DataFrame:
    latent = per_run[per_run.method.str.startswith("latent")].copy()
    latent["latent_dim"] = latent.method.str.extract(r"z(\d+)")[0].astype(int)
    cols = ["final_reward", "auc_reward", "pipeline_time_total_s"]
    return latent.groupby("latent_dim")[cols].agg(["mean", "std"]).round(1)
