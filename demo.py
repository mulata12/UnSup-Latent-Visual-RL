#!/usr/bin/env python
"""Interactive demo of the project (Gradio).

    python demo.py                 # local:   http://127.0.0.1:7860
    python demo.py --share         # Colab / remote: prints a temporary public link

Needs the trained encoders in demo_assets/ and the saved results in results/
(create both with  bash scripts/prepare_github_release.sh). No GPU required.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))   # works without `pip install -e .`

from latent_visual_rl.demo.app import main  # noqa: E402

if __name__ == "__main__":
    main()
