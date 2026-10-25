# Results (full profile)

Seeds: [0] | steps per run: 50,000 | reward threshold: 400.0

## Summary by method (mean / std over seeds)

| method | final_reward_mean | final_reward_std | auc_reward_mean | steps_to_threshold_mean | time_to_threshold_s_mean | pipeline_time_total_s_mean | runs_reaching_threshold |
|---|---|---|---|---|---|---|---|
| latent_dqn_z16 | 52.1 | nan | 68.6 | nan | nan | 652.2 | 0/1 |
| pixel_dqn | 20.4 | nan | 21.1 | nan | nan | 308.5 | 0/1 |


## Welch t-tests (pixel vs latent)

_not available (needs >= 2 seeds per method)_


## Linear-probe R² (single-frame latent -> physical state)

| method | R2 |
|---|---|
| cart_pos | 0.868 |
| cart_vel | 0.041 |
| pole_angle | 0.011 |
| pole_ang_vel | 0.003 |


## Latent-size ablation

See `tables/ablation_latent_dim.csv` and `figures/ablation_curves.png`.

## Environment

```json
{
  "python": "3.13.16",
  "torch": "2.11.0+cu130",
  "gymnasium": "1.4.0",
  "sb3": "2.9.0",
  "numpy": "2.1.3",
  "sklearn": "1.6.1",
  "gpu": "Tesla T4"
}
```
