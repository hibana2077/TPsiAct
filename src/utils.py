"""
Utility module for TPsiAct experiments.

Contains:
- Data augmentation helpers
- Logging utilities
- Configuration management
- Report generation
"""

import torch
import torch.nn as nn
from torchvision import transforms
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import json
import yaml
import random
import numpy as np
from datetime import datetime
import sys


# ============================================================================
# Reproducibility
# ============================================================================

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# Data Augmentation
# ============================================================================

def get_transforms(
    augmentation: str = 'standard',
    image_size: int = 224,
    is_train: bool = True
) -> transforms.Compose:
    """
    Get data transforms based on augmentation strategy.
    
    Args:
        augmentation: Type of augmentation ('standard', 'autoaug', 'randaug', 'none')
        image_size: Target image size
        is_train: Whether this is for training or evaluation
    
    Returns:
        Composed transforms
    """
    # Normalization values (ImageNet)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if not is_train:
        # Evaluation transforms
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize
        ])
    
    # Training transforms based on augmentation type
    if augmentation == 'none':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize
        ])
    
    elif augmentation == 'standard':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            normalize
        ])
    
    
    elif augmentation == 'autoaug':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
            transforms.ToTensor(),
            normalize
        ])
    
    elif augmentation == 'randaug':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            normalize
        ])
    
    # Note: SimCLR-style and custom 'yuki' augmentations removed
    
    else:
        raise ValueError(f"Unknown augmentation: {augmentation}")


# ============================================================================
# Logging
# ============================================================================

class Logger:
    """Simple logger that writes to file and stdout."""
    
    def __init__(self, log_path: Optional[Path] = None, verbose: bool = True):
        self.log_path = Path(log_path) if log_path else None
        self.verbose = verbose
        
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            # Clear or create log file
            with open(self.log_path, 'w') as f:
                f.write(f"Log started at {datetime.now().isoformat()}\n")
                f.write("=" * 80 + "\n")
    
    def log(self, message: str, level: str = 'INFO'):
        """Log a message."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] [{level}] {message}"
        
        if self.verbose:
            print(formatted)
        
        if self.log_path:
            with open(self.log_path, 'a') as f:
                f.write(formatted + "\n")
    
    def info(self, message: str):
        self.log(message, 'INFO')
    
    def warning(self, message: str):
        self.log(message, 'WARNING')
    
    def error(self, message: str):
        self.log(message, 'ERROR')


# ============================================================================
# Configuration
# ============================================================================

def load_config(config_path: Union[str, Path]) -> Dict:
    """Load configuration from YAML or JSON file."""
    config_path = Path(config_path)
    
    if config_path.suffix in ['.yaml', '.yml']:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    elif config_path.suffix == '.json':
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unknown config format: {config_path.suffix}")


def save_config(config: Dict, config_path: Union[str, Path]):
    """Save configuration to YAML or JSON file."""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    if config_path.suffix in ['.yaml', '.yml']:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    elif config_path.suffix == '.json':
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unknown config format: {config_path.suffix}")


# ============================================================================
# GPU/Device Utilities
# ============================================================================

def get_device(gpu_id: Optional[int] = None) -> torch.device:
    """Get the appropriate device for training."""
    if torch.cuda.is_available():
        if gpu_id is not None:
            return torch.device(f'cuda:{gpu_id}')
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information."""
    if not torch.cuda.is_available():
        return {'available': False}
    
    info = {
        'available': True,
        'device_count': torch.cuda.device_count(),
        'current_device': torch.cuda.current_device(),
        'devices': []
    }
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        info['devices'].append({
            'name': props.name,
            'total_memory_gb': props.total_memory / (1024**3),
            'compute_capability': f"{props.major}.{props.minor}"
        })
    
    return info


# ============================================================================
# Model Utilities
# ============================================================================

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable
    }


def get_model_size_mb(model: nn.Module) -> float:
    """Get model size in MB."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


# ============================================================================
# Report Generation
# ============================================================================

def format_time(seconds: float) -> str:
    """Format time in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def generate_experiment_name(
    dataset: str,
    backbone: str,
    use_tpsiact: bool,
    augmentation: str,
    seed: int
) -> str:
    """Generate a descriptive experiment name."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tpsiact_str = "tpsiact" if use_tpsiact else "baseline"
    return f"{dataset}_{backbone}_{tpsiact_str}_{augmentation}_seed{seed}_{timestamp}"


def print_config(config: Dict):
    """Print configuration in a nice format."""
    print("\n" + "=" * 60)
    print(" " * 20 + "CONFIGURATION")
    print("=" * 60)
    
    for key, value in sorted(config.items()):
        if isinstance(value, dict):
            print(f"\n  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    print("=" * 60 + "\n")


def print_dataset_info(dataset_info: Dict):
    """Print dataset information."""
    print("\n" + "-" * 60)
    print(" " * 20 + "DATASET INFO")
    print("-" * 60)
    print(f"  Name: {dataset_info.get('dataset_name', 'Unknown')}")
    print(f"  Description: {dataset_info.get('description', 'N/A')}")
    print(f"  Train samples: {dataset_info.get('train_samples', 'N/A')}")
    print(f"  Test samples: {dataset_info.get('test_samples', 'N/A')}")
    print(f"  Classes: {dataset_info.get('num_classes', 'N/A')}")
    print("-" * 60 + "\n")


def print_model_info(model_info: Dict):
    """Print model information."""
    print("\n" + "-" * 60)
    print(" " * 20 + "MODEL INFO")
    print("-" * 60)
    print(f"  Backbone: {model_info.get('backbone', 'Unknown')}")
    print(f"  Feature dim: {model_info.get('feature_dim', 'N/A')}")
    print(f"  Use TPsiAct: {model_info.get('use_tpsiact', 'N/A')}")
    print(f"  TPsiAct nu: {model_info.get('tpsiact_nu', 'N/A')}")
    print(f"  Total params: {model_info.get('total_params', 0):,}")
    print(f"  Trainable params: {model_info.get('trainable_params', 0):,}")
    print("-" * 60 + "\n")


# ============================================================================
# Data Loading Utilities
# ============================================================================

def create_dataloaders(
    train_dataset,
    test_dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple:
    """Create train and test data loaders."""
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, test_loader


# ============================================================================
# Mixup / CutMix (Optional augmentation)
# ============================================================================

class Mixup:
    """Mixup data augmentation."""
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    
    def __call__(
        self, 
        images: torch.Tensor, 
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply mixup to a batch.
        
        Returns:
            mixed_images: Mixed images
            labels_a: Original labels
            labels_b: Shuffled labels
            lam: Mixing coefficient
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = images.size(0)
        index = torch.randperm(batch_size).to(images.device)
        
        mixed_images = lam * images + (1 - lam) * images[index]
        labels_a, labels_b = labels, labels[index]
        
        return mixed_images, labels_a, labels_b, lam
    
    @staticmethod
    def criterion(
        criterion: nn.Module,
        pred: torch.Tensor,
        labels_a: torch.Tensor,
        labels_b: torch.Tensor,
        lam: float
    ) -> torch.Tensor:
        """Compute mixup loss."""
        return lam * criterion(pred, labels_a) + (1 - lam) * criterion(pred, labels_b)


if __name__ == "__main__":
    # Test utilities
    print("Testing utils module...")
    
    # Test transforms
    train_transform = get_transforms('standard', is_train=True)
    test_transform = get_transforms('standard', is_train=False)
    print(f"Train transform: {train_transform}")
    print(f"Test transform: {test_transform}")
    
    # Test device
    device = get_device()
    print(f"Device: {device}")
    
    # Test GPU info
    gpu_info = get_gpu_info()
    print(f"GPU info: {gpu_info}")
    
    # Test seed setting
    set_seed(42)
    print("Seed set successfully")
    
    print("\nAll tests passed!")
