import pytest

from latent_visual_rl.cli import main


@pytest.mark.slow
def test_end_to_end_smoke(tmp_path):
    main(["run-all", "--config", "configs/smoke_test.yaml", "--set", f"experiment.output_dir={tmp_path}",
          "--set", "experiment.seeds=[0]"])
    run = tmp_path / "smoke_test"
    assert (run / "config.yaml").exists() and (run / "environment.json").exists()
    assert (run / "tables" / "summary_by_method.csv").exists()
    assert (run / "figures" / "reconstructions.png").exists()
