# Results

Place the output of the final run here (the notebook's section 11 creates a zip with exactly this layout):

```
results/
├── RESULTS.md          # auto-generated summary of the final numbers
├── run_info.json       # config + library versions + GPU of the run
├── tables/             # per_run_metrics.csv, summary_by_method.csv, welch_t_tests.json, ...
└── figures/            # learning_curves.png, reconstructions.png, cae_loss.png, ...
```

These files are referenced by the main `README.md` (figures and tables). Large artefacts (model checkpoints, cached
frame datasets) are intentionally **not** committed.
