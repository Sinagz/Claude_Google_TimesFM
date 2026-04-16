"""
Model Loader
─────────────
Centralised utilities for resolving the compute device, downloading / caching
HuggingFace model weights, and reporting the system configuration.

Both TimesFM and Chronos agents call these helpers so device selection
logic lives in exactly one place.
"""

import os
from typing import Optional

from utils.helpers import setup_logger

logger = setup_logger("model_loader")


# ── Device resolution ─────────────────────────────────────────────────────────

def get_device(preference: str = "auto") -> str:
    """
    Return the best available device string ("cuda", "mps", or "cpu").

    Parameters
    ----------
    preference : "auto" | "cuda" | "cpu" | "mps"
    """
    if preference not in ("auto",):
        return preference

    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info("CUDA device: %s  (%.1f GB VRAM)", name, vram)
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Apple MPS device available")
            return "mps"
        logger.info("No GPU found — using CPU")
        return "cpu"
    except ImportError:
        logger.warning("PyTorch not installed — device defaulting to cpu")
        return "cpu"


# ── HuggingFace cache ─────────────────────────────────────────────────────────

def set_hf_cache(cache_dir: str = "./models/cache") -> None:
    """Point HuggingFace hub to a local cache directory."""
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)
    logger.debug("HuggingFace cache → %s", cache_dir)


# ── System info ────────────────────────────────────────────────────────────────

def print_system_info() -> None:
    """Log Python / PyTorch / CUDA version info for debugging."""
    import sys
    logger.info("Python %s", sys.version.split()[0])

    try:
        import torch
        logger.info("PyTorch %s", torch.__version__)
        if torch.cuda.is_available():
            logger.info("CUDA %s", torch.version.cuda)
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logger.info(
                    "  GPU %d: %s  %.1f GB",
                    i, props.name, props.total_memory / 1e9,
                )
    except ImportError:
        logger.warning("PyTorch not found")

    try:
        import numpy as np
        logger.info("NumPy %s", np.__version__)
    except ImportError:
        pass

    try:
        import pandas as pd
        logger.info("Pandas %s", pd.__version__)
    except ImportError:
        pass


# ── Model availability checks ─────────────────────────────────────────────────

def check_timesfm_available() -> bool:
    try:
        import timesfm  # noqa: F401
        return True
    except ImportError:
        return False


def check_chronos_available() -> bool:
    try:
        from chronos import ChronosBoltPipeline  # noqa: F401
        return True
    except ImportError:
        return False


def report_model_availability() -> dict:
    """Return a dict of {model_name: is_available} for startup logging."""
    status = {
        "timesfm":  check_timesfm_available(),
        "chronos":  check_chronos_available(),
        "finbert":  _check_transformers(),
        "yfinance": _check_yfinance(),
    }
    for name, ok in status.items():
        icon = "✓" if ok else "✗"
        logger.info("  %s %s", icon, name)
    return status


def _check_transformers() -> bool:
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def _check_yfinance() -> bool:
    try:
        import yfinance  # noqa: F401
        return True
    except ImportError:
        return False
