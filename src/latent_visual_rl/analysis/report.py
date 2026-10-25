"""Turn raw result CSVs into tables, statistics and figures."""
import json

from latent_visual_rl.analysis import metrics, plotting
from latent_visual_rl.analysis.representation_diagnostics import linear_probe_r2
from latent_visual_rl.reproducibility import resolve_device


def generate_report(cfg) -> None:
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    cfg.tables_dir.mkdir(parents=True, exist_ok=True)
    thr = cfg.agent.success_threshold
    df = metrics.load_results(cfg.results_dir)
    per_run = metrics.compute_per_run_metrics(df, thr)
    per_run.to_csv(cfg.tables_dir / "per_run_metrics.csv", index=False)
    summary = metrics.summarize_methods(per_run)
    summary.to_csv(cfg.tables_dir / "summary_by_method.csv")
    print(summary.T.to_string())

    main_methods = [cfg.pixel_method, cfg.latent_method()]
    main_df = df[df.method.isin(main_methods)]
    if main_df.method.nunique() == 2:
        plotting.plot_learning_curves(main_df, "timesteps", thr, out_path=cfg.figures_dir / "learning_curve_steps.png")
        plotting.plot_learning_curves(main_df, "total_time_s", thr, out_path=cfg.figures_dir / "learning_curve_time.png")
        tests = metrics.welch_t_tests(per_run, *main_methods)
        json.dump(tests, open(cfg.tables_dir / "welch_t_tests.json", "w"), indent=2)
        print("\nWelch t-tests", main_methods, tests)

    if per_run[per_run.method.str.startswith("latent")].method.nunique() > 1:
        abl = metrics.latent_dim_ablation_table(per_run)
        abl.to_csv(cfg.tables_dir / "ablation_latent_dim.csv")
        plotting.plot_learning_curves(df, "timesteps", thr, "Latent-size ablation",
                                      cfg.figures_dir / "ablation_learning_curves.png")
        print("\nLatent-dim ablation\n", abl.to_string())

    ckpt_dir = cfg.autoencoder_dir()
    if (ckpt_dir / "autoencoder.pt").exists():
        from latent_visual_rl.pipelines.stages import ensure_autoencoder, ensure_frame_dataset
        device = resolve_device(cfg.experiment.device)
        trained, meta = ensure_autoencoder(cfg)
        frames, states, _ = ensure_frame_dataset(cfg)
        plotting.plot_autoencoder_loss(meta["history"], cfg.figures_dir / "autoencoder_loss.png")
        plotting.plot_reconstructions(trained, frames[-64:], device, out_path=cfg.figures_dir / "reconstructions.png")
        probe = linear_probe_r2(trained, frames, states, device)
        json.dump(probe, open(cfg.tables_dir / "latent_linear_probe_r2.json", "w"), indent=2)
        print("\nLinear-probe R^2 (single-frame latent -> physical state):", probe)
    print(f"\nReport written to {cfg.run_dir}")
