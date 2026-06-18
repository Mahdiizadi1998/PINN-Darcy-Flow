"""
Utilities for reproducibility, device setup, and configuration.
"""

import os
import random
import torch
import numpy as np
import yaml
from typing import Optional


def set_seed(seed: int = 42):
    """
    Set all random seeds for reproducibility.

    Note: full reproducibility on GPU also requires:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    but these slow down training significantly.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Get the best available device.

    Priority: CUDA GPU > MPS (Apple Silicon) > CPU
    """
    if prefer_gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            return device
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using Apple Silicon GPU (MPS)")
            return device

    print("Using CPU")
    return torch.device("cpu")


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_model(model: torch.nn.Module, path: str, metadata: Optional[dict] = None):
    """
    Save model checkpoint with optional metadata.

    Args:
        model: PyTorch model
        path: Save path (.pt)
        metadata: Optional dict with training info (epoch, loss, etc.)
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "class": model.__class__.__name__,
        },
    }
    if metadata:
        checkpoint["metadata"] = metadata

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")


def load_model(model: torch.nn.Module, path: str) -> dict:
    """
    Load model weights from checkpoint.

    Args:
        model: Model instance (must match architecture of saved model)
        path: Checkpoint path

    Returns:
        Metadata dict (or empty dict if no metadata was saved)
    """
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Model loaded from {path}")
    return checkpoint.get("metadata", {})
