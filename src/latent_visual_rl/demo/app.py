"""Gradio user interface. Launch with:  python demo.py   (see README)."""
import argparse

import gradio as gr

from latent_visual_rl.config import Config
from latent_visual_rl.demo import episodes, figures, results
from latent_visual_rl.demo.assets import DemoAssets, load_assets
from latent_visual_rl.demo.rendering import LIMIT_CART_POS, LIMIT_POLE_DEG, StateRenderer

HEADER = """
# Unsupervised State Representation Learning for Sample-Efficient Visual RL — interactive demo
**Pipeline:** pixels (64×64) → *frozen convolutional-autoencoder encoder* → latent vector **z** → DQN agent.
This demo uses the encoders trained in the final experiment (CPU only, no GPU needed).
"""
TIP_NO_AGENT = ("_No trained agent found. To add one: `python scripts/train_demo_agent.py` "
                "(≈ 5–6 min), then restart the demo._")


def build_app(assets: DemoAssets) -> gr.Blocks:
    cfg = Config()
    cfg.experiment.device = "cpu"
    renderer = StateRenderer(cfg)
    models = assets.models
    dims = sorted(models)
    default_dim = 16 if 16 in dims else dims[0]
    dim_choices = [(f"z = {d}", d) for d in dims]

    # ----------------------------------------------------------------- handlers
    def lens(pos, angle, dim):
        fig, mse = figures.lens_figure(renderer, models[int(dim)], pos, angle)
        return fig, f"Reconstruction MSE: **{mse:.4f}** (pixel scale 0–1)"

    def sweep(vary, other, dim):
        return figures.sweep_figure(renderer, models[int(dim)], vary, other)

    def bottleneck(pos, angle):
        return figures.bottleneck_figure(renderer, models, pos, angle)

    curves = results.load_curves(assets.results_dir)

    def plot_curves(methods, x_axis):
        if curves is None or not methods:
            return None
        return figures.curves_figure(curves, methods, x_axis)

    def episode(policy_name, seed, dim_for_random):
        if policy_name.startswith("agent:"):
            d = int(policy_name.split(":")[1])
            policy, ae = episodes.agent_policy(assets.agents[d]), models[d]
        else:
            policy, ae = episodes.random_policy, models[int(dim_for_random)]
        frames, Z, total = episodes.run_episode(cfg, ae, policy, int(seed))
        who = "trained agent" if policy_name.startswith("agent:") else "random policy"
        summary = f"**{who}** — episode lasted **{len(frames)} steps**, return **{total:.0f}** (maximum possible 500)."
        return episodes.frames_to_gif(frames), figures.latent_trajectory_figure(Z), summary

    # ----------------------------------------------------------------- layout
    with gr.Blocks(title="Latent Visual RL demo") as app:
        gr.Markdown(HEADER)

        with gr.Tab("1 · Latent lens"):
            gr.Markdown("Move the cart and tilt the pole. The encoder compresses the image into **z**; "
                        "the decoder redraws the scene from **z** alone. Release a slider to update.")
            with gr.Row():
                pos = gr.Slider(-LIMIT_CART_POS, LIMIT_CART_POS, 0.0, step=0.1, label="Cart position")
                ang = gr.Slider(-LIMIT_POLE_DEG, LIMIT_POLE_DEG, 0.0, step=0.5, label="Pole angle (degrees)")
                dim = gr.Dropdown(dim_choices, value=default_dim, label="Latent size")
            lens_plot, lens_info = gr.Plot(label="encoder / decoder"), gr.Markdown()
            gr.on([pos.release, ang.release, dim.change], lens, [pos, ang, dim], [lens_plot, lens_info],
                  api_name="lens")
            app.load(lens, [pos, ang, dim], [lens_plot, lens_info], api_name=False)

        with gr.Tab("2 · Latent sweep"):
            gr.Markdown("Sweep one physical quantity across its range and watch every latent dimension. "
                        "A quantity the encoder represents strongly gives clear structure; a weakly represented "
                        "one gives little change.")
            with gr.Row():
                vary = gr.Radio(["pole angle", "cart position"], value="pole angle", label="Sweep")
                other = gr.Slider(-LIMIT_POLE_DEG, LIMIT_POLE_DEG, 0.0, step=0.5,
                                  label="Value of the other quantity (degrees or position)")
                dim2 = gr.Dropdown(dim_choices, value=default_dim, label="Latent size")
            sweep_plot = gr.Plot()
            gr.on([vary.change, other.release, dim2.change], sweep, [vary, other, dim2], sweep_plot,
                  api_name="sweep")
            app.load(sweep, [vary, other, dim2], sweep_plot, api_name=False)

        with gr.Tab("3 · Bottleneck size"):
            gr.Markdown("The same scene reconstructed through bottlenecks of different sizes.")
            with gr.Row():
                pos3 = gr.Slider(-LIMIT_CART_POS, LIMIT_CART_POS, 0.5, step=0.1, label="Cart position")
                ang3 = gr.Slider(-LIMIT_POLE_DEG, LIMIT_POLE_DEG, 6.0, step=0.5, label="Pole angle (degrees)")
            bott_plot = gr.Plot()
            gr.on([pos3.release, ang3.release], bottleneck, [pos3, ang3], bott_plot, api_name="bottleneck")
            app.load(bottleneck, [pos3, ang3], bott_plot, api_name=False)

        with gr.Tab("4 · Live episode"):
            gr.Markdown("Run an episode through the **same observation pipeline as training** "
                        "(frozen encoder + 4-step latent stack) and watch the latent vector evolve.")
            policy_choices = [("Random policy", "random")] + [
                (f"Trained latent agent (z = {d})", f"agent:{d}") for d in sorted(assets.agents)]
            if not assets.agents:
                gr.Markdown(TIP_NO_AGENT)
            with gr.Row():
                pol = gr.Dropdown(policy_choices, value="random", label="Policy")
                seed = gr.Number(0, precision=0, label="Environment seed")
                dim4 = gr.Dropdown(dim_choices, value=default_dim, label="Latent size (random policy only)")
                run_btn = gr.Button("Run episode", variant="primary")
            with gr.Row():
                gif = gr.Image(label="environment", type="filepath")
                traj = gr.Plot()
            ep_info = gr.Markdown()
            run_btn.click(episode, [pol, seed, dim4], [gif, traj, ep_info], api_name="episode")

        with gr.Tab("5 · Experiment results"):
            gr.Markdown(results.seed_note(curves))
            if curves is None:
                gr.Markdown(f"_No results found in `{assets.results_dir}`. Run `scripts/prepare_github_release.sh`._")
            else:
                names = sorted(curves.method.unique())
                default_methods = [m for m in names if m == "pixel_dqn" or m.endswith(f"z{default_dim}")]
                with gr.Row():
                    methods = gr.CheckboxGroup(names, value=default_methods, label="Methods")
                    xaxis = gr.Radio(["Environment steps", "Wall-clock seconds (incl. pre-training)"],
                                     value="Environment steps", label="x-axis")
                curves_plot = gr.Plot()
                gr.on([methods.change, xaxis.change], plot_curves, [methods, xaxis], curves_plot,
                      api_name="curves")
                app.load(plot_curves, [methods, xaxis], curves_plot, api_name=False)
            for title, df in [("Summary by method", results.read_summary(assets.results_dir)),
                              ("Latent-size ablation", results.read_ablation(assets.results_dir)),
                              ("Linear probe: physical state from a single-frame latent",
                               results.read_probe(assets.results_dir))]:
                if df is not None:
                    gr.Markdown(f"**{title}**")
                    gr.Dataframe(df.round(3), interactive=False)
            for fig_name in ("learning_curves.png", "reconstructions.png"):
                path = assets.results_dir / "figures" / fig_name
                if path.exists():
                    gr.Image(str(path), label=fig_name, interactive=False)
    return app


def main(argv=None):
    p = argparse.ArgumentParser(description="Launch the interactive demo.")
    p.add_argument("--assets", default="demo_assets", help="folder with cae_z*.pt (and optional latent_agent_z*.zip)")
    p.add_argument("--results", default="results", help="folder with tables/, figures/, raw/")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="create a temporary public link (use in Colab)")
    args = p.parse_args(argv)
    assets = load_assets(args.assets, args.results)
    print(f"Loaded encoders: {sorted(assets.models)} | agents: {sorted(assets.agents)}")
    build_app(assets).launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
