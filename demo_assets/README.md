# demo_assets

Small pretrained artefacts used by the interactive demo (`python demo.py`):

* `cae_z4.pt`, `cae_z8.pt`, `cae_z16.pt`, `cae_z32.pt`, `cae_z64.pt` – the trained autoencoders (a few MB each).
* `latent_agent_z16.zip` – *(optional)* a trained latent DQN agent for the "Live episode" tab, created with
  `python scripts/train_demo_agent.py`.

Fill this folder with `bash scripts/prepare_github_release.sh` after the final run.
