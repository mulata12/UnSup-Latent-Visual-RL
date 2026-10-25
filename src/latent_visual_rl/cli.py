"""Command-line interface:  python -m latent_visual_rl <command> --config configs/default.yaml

Commands: collect-data | train-autoencoder | train-agents | ablation | analyze | run-all
Override any config value with  --set section.key=value  (repeatable).
"""
import argparse

from latent_visual_rl.config import Config


def _parser():
    p = argparse.ArgumentParser(prog="latent_visual_rl", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["collect-data", "train-autoencoder", "train-agents", "ablation",
                                       "analyze", "run-all"])
    p.add_argument("--config", default="configs/default.yaml", help="YAML config file")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override config values, e.g. --set autoencoder.latent_dim=32")
    p.add_argument("--methods", nargs="+", default=["pixel", "latent"], choices=["pixel", "latent"],
                   help="train-agents only: which pipelines to train")
    p.add_argument("--dims", nargs="+", type=int, default=[4, 8, 32, 64],
                   help="ablation only: latent sizes to sweep")
    return p


def main(argv=None):
    args = _parser().parse_args(argv)
    cfg = Config.from_yaml(args.config, args.set)
    # imports are deferred so `--help` stays instant
    from latent_visual_rl.pipelines import stages
    from latent_visual_rl.analysis.report import generate_report

    stages.prepare_run_directory(cfg)
    if args.command == "collect-data":
        _, _, sec = stages.ensure_frame_dataset(cfg)
        print(f"frames ready ({sec:.0f}s collection) -> {cfg.frame_dataset_path}")
    elif args.command == "train-autoencoder":
        _, meta = stages.ensure_autoencoder(cfg)
        print(f"autoencoder ready (val loss {meta['final_val_loss']:.5f}) -> {cfg.autoencoder_dir()}")
    elif args.command == "train-agents":
        stages.run_agent_stage(cfg, args.methods)
    elif args.command == "ablation":
        stages.run_latent_dim_ablation(cfg, args.dims)
    elif args.command == "analyze":
        generate_report(cfg)
    elif args.command == "run-all":
        stages.run_agent_stage(cfg, ("pixel", "latent"))
        generate_report(cfg)


if __name__ == "__main__":
    main()
