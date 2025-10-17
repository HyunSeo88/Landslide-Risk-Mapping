"""
Model checkpoint management
"""

import torch
from pathlib import Path
from typing import Dict, Any, Optional
import shutil


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_dir: str,
    filename: str = "checkpoint.pth",
    is_best: bool = False
):
    """
    Save model checkpoint

    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Dictionary of metrics
        checkpoint_dir: Directory to save checkpoint
        filename: Checkpoint filename
        is_best: Whether this is the best model so far
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }

    checkpoint_path = checkpoint_dir / filename
    torch.save(checkpoint, checkpoint_path)

    # Save best model separately
    if is_best:
        best_path = checkpoint_dir / "best_model.pth"
        shutil.copy(checkpoint_path, best_path)
        print(f"Best model saved to {best_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Load model checkpoint

    Args:
        checkpoint_path: Path to checkpoint file
        model: PyTorch model
        optimizer: Optimizer (optional)
        device: Device to load model

    Returns:
        Dictionary with epoch and metrics
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Metrics: {checkpoint.get('metrics', {})}")

    return {
        'epoch': checkpoint['epoch'],
        'metrics': checkpoint.get('metrics', {})
    }


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Get path to latest checkpoint in directory

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        Path to latest checkpoint or None
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        return None

    checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pth"))

    if not checkpoints:
        # Try generic checkpoint
        generic = checkpoint_dir / "checkpoint.pth"
        if generic.exists():
            return str(generic)
        return None

    # Sort by epoch number
    checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))

    return str(checkpoints[-1])