"""Read the saved experiment results (tables and learning curves) for the dashboard."""
import json
from pathlib import Path

import pandas as pd


def load_curves(results_dir):
    files = sorted((Path(results_dir) / "raw").glob("*_seed*.csv"))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else None


def read_summary(results_dir):
    p = Path(results_dir) / "tables" / "summary_by_method.csv"
    return pd.read_csv(p).rename(columns={"Unnamed: 0": "method"}) if p.exists() else None


def read_ablation(results_dir):
    """The notebook saves a two-row (mean/std) column header; flatten it to final_reward_mean, ..."""
    p = Path(results_dir) / "tables" / "ablation_latent_dim.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, header=[0, 1], index_col=0)
    df.columns = [f"{a}_{b}" for a, b in df.columns]
    df = df[df.index.notna()].reset_index()
    return df.rename(columns={df.columns[0]: "latent_dim"})


def read_probe(results_dir):
    p = Path(results_dir) / "tables" / "latent_linear_probe_r2.json"
    if not p.exists():
        return None
    return pd.DataFrame(json.load(open(p)).items(), columns=["state variable", "R² (linear probe)"])


def seed_note(curves):
    if curves is None:
        return "No raw learning curves found in `results/raw/`."
    n = curves.groupby("method").seed.nunique().min()
    steps = int(curves.timesteps.max())
    msg = f"Recorded **{steps:,} environment steps** per run, **{n} seed(s)** per method."
    if n < 3:
        msg += " ⚠️ With fewer than 3 seeds no error bars or significance tests are possible — read the curves as *descriptive*."
    return msg
