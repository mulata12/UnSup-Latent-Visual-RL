from latent_visual_rl.config import Config


def test_defaults_roundtrip(tmp_path):
    cfg = Config()
    cfg.save(tmp_path / "c.yaml")
    assert Config.from_yaml(tmp_path / "c.yaml").to_dict() == cfg.to_dict()


def test_cli_overrides_and_partial_nested_dict():
    cfg = Config.from_yaml(None, ["autoencoder.latent_dim=32", "agent.dqn_pixel.learning_starts=10",
                                  "experiment.seeds=[5,6]"])
    assert cfg.autoencoder.latent_dim == 32
    assert cfg.agent.dqn_pixel["learning_starts"] == 10
    assert cfg.agent.dqn_pixel["batch_size"] == 64          # untouched defaults survive deep merge
    assert cfg.experiment.seeds == [5, 6]


def test_method_names_and_paths():
    cfg = Config()
    assert cfg.latent_method() == "latent_dqn_z16"
    assert cfg.with_latent_dim(8).latent_method() == "latent_dqn_z8"
    assert cfg.autoencoder_dir(8).name == "latent_dim_8"
