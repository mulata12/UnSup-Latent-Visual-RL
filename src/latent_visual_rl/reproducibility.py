"""Seeding, device resolution and environment fingerprinting."""
import platform
import random

import numpy as np


def seed_everything(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(preference: str = "auto") -> str:
    import torch
    if preference == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return preference


def collect_environment_info() -> dict:
    """Library versions + hardware, saved next to every experiment for provenance."""
    import gymnasium
    import stable_baselines3
    import torch
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "gymnasium": gymnasium.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
