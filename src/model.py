"""
Model module for TPsiAct experiments.

Contains:
- TPsiActModel: Main model class with timm backbone and optional TPsiAct replacement
- Feature extractor wrapper
- Various backbone configurations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
import timm
from timm.models import create_model

from tpsiact import TPsiAct, replace_activations_with_tpsiact


class FeatureExtractor(nn.Module):
    """
    Wrapper that returns features (before classifier head) from a model.
    """
    
    def __init__(self, model: nn.Module, feature_dim: int):
        super().__init__()
        self.model = model
        self.feature_dim = feature_dim
        self._features = None
        self._hook = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input."""
        # For timm models, use forward_features
        if hasattr(self.model, 'forward_features'):
            features = self.model.forward_features(x)
            # Global average pooling if needed
            if features.dim() == 4:
                features = F.adaptive_avg_pool2d(features, 1).flatten(1)
            elif features.dim() == 3:
                features = features.mean(dim=1)  # For ViT: (B, N, D) -> (B, D)
            return features
        else:
            # Try to get features from a generic model
            return self.model(x)


class TPsiActClassifier(nn.Module):
    """
    Classifier head with optional TPsiAct activation.
    """
    
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dim: Optional[int] = None,
        use_tpsiact: bool = True,
        nu: float = 5.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.use_tpsiact = use_tpsiact
        
        if hidden_dim is not None:
            self.classifier = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                TPsiAct(nu=nu) if use_tpsiact else nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            self.classifier = nn.Linear(in_features, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class TPsiActModel(nn.Module):
    """
    Main model class combining timm backbone with TPsiAct.
    
    Supports:
    - Various timm backbones (ResNet, ViT, ConvNeXt, etc.)
    - Optional activation replacement with TPsiAct
    - Feature extraction mode
    - Uncertainty estimation
    """
    
    def __init__(
        self,
        backbone_name: str = 'resnet50',
        num_classes: int = 200,
        pretrained: bool = True,
        use_tpsiact: bool = True,
        tpsiact_nu: float = 5.0,
        replace_backbone_activations: bool = False,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        freeze_backbone: bool = False
    ):
        """
        Args:
            backbone_name: Name of timm model to use as backbone.
            num_classes: Number of output classes.
            pretrained: Whether to use pretrained weights.
            use_tpsiact: Whether to use TPsiAct in classifier head.
            tpsiact_nu: Degrees of freedom for TPsiAct.
            replace_backbone_activations: Whether to replace backbone activations with TPsiAct.
            hidden_dim: If set, adds a hidden layer before final classifier.
            dropout: Dropout rate.
            freeze_backbone: Whether to freeze backbone parameters.
        """
        super().__init__()
        
        self.backbone_name = backbone_name
        self.use_tpsiact = use_tpsiact
        self.tpsiact_nu = tpsiact_nu
        
        # Create backbone
        self.backbone = create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier head
            global_pool='avg'
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            if features.dim() == 3:
                features = features.mean(dim=1)
            self.feature_dim = features.shape[-1]
        
        # Optionally replace backbone activations
        if replace_backbone_activations:
            self.backbone = replace_activations_with_tpsiact(
                self.backbone,
                nu=tpsiact_nu,
                inplace=True
            )
        
        # Create classifier
        self.classifier = TPsiActClassifier(
            in_features=self.feature_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            use_tpsiact=use_tpsiact,
            nu=tpsiact_nu,
            dropout=dropout
        )
        
        # Freeze backbone if requested
        if freeze_backbone:
            self.freeze_backbone()
        
        # TPsiAct layer for uncertainty (optional, on features)
        self.tpsiact_features = TPsiAct(nu=tpsiact_nu) if use_tpsiact else None
    
    def freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from backbone."""
        features = self.backbone(x)
        if features.dim() == 3:
            features = features.mean(dim=1)
        return features
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        return_uncertainty: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass.
        
        Args:
            x: Input images (B, C, H, W).
            return_features: If True, also return features.
            return_uncertainty: If True, also return uncertainty.
        
        Returns:
            logits: (B, num_classes) classification logits.
            features: (optional) (B, feature_dim) features.
            uncertainty: (optional) (B, feature_dim) per-feature uncertainty.
        """
        # Extract features
        features = self.get_features(x)
        
        # Compute uncertainty if requested
        uncertainty = None
        if return_uncertainty and self.tpsiact_features is not None:
            _, uncertainty = self.tpsiact_features(features, return_uncertainty=True)
        
        # Classify
        logits = self.classifier(features)
        
        # Build return tuple
        outputs = [logits]
        if return_features:
            outputs.append(features)
        if return_uncertainty:
            outputs.append(uncertainty)
        
        return outputs[0] if len(outputs) == 1 else tuple(outputs)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'backbone': self.backbone_name,
            'feature_dim': self.feature_dim,
            'use_tpsiact': self.use_tpsiact,
            'tpsiact_nu': self.tpsiact_nu,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'param_ratio': trainable_params / total_params if total_params > 0 else 0
        }


class EnsembleTPsiActModel(nn.Module):
    """
    Ensemble of TPsiAct models for improved uncertainty estimation.
    """
    
    def __init__(
        self,
        num_models: int = 3,
        backbone_name: str = 'resnet50',
        num_classes: int = 200,
        **kwargs
    ):
        super().__init__()
        self.num_models = num_models
        self.models = nn.ModuleList([
            TPsiActModel(
                backbone_name=backbone_name,
                num_classes=num_classes,
                **kwargs
            )
            for _ in range(num_models)
        ])
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        return_uncertainty: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass with ensemble averaging.
        """
        all_logits = []
        all_features = []
        all_uncertainties = []
        
        for model in self.models:
            outputs = model(
                x,
                return_features=return_features,
                return_uncertainty=return_uncertainty
            )
            
            if isinstance(outputs, tuple):
                all_logits.append(outputs[0])
                if return_features:
                    all_features.append(outputs[1])
                if return_uncertainty:
                    all_uncertainties.append(outputs[-1])
            else:
                all_logits.append(outputs)
        
        # Average predictions
        avg_logits = torch.stack(all_logits, dim=0).mean(dim=0)
        
        outputs = [avg_logits]
        
        if return_features:
            avg_features = torch.stack(all_features, dim=0).mean(dim=0)
            outputs.append(avg_features)
        
        if return_uncertainty:
            # Combine per-model uncertainty with ensemble disagreement
            avg_uncertainty = torch.stack(all_uncertainties, dim=0).mean(dim=0)
            # Add epistemic uncertainty from ensemble variance
            epistemic = torch.stack(all_logits, dim=0).var(dim=0).mean(dim=-1, keepdim=True)
            combined_uncertainty = avg_uncertainty + epistemic.expand_as(avg_uncertainty)
            outputs.append(combined_uncertainty)
        
        return outputs[0] if len(outputs) == 1 else tuple(outputs)


def create_model_from_config(config: Dict) -> TPsiActModel:
    """
    Create a TPsiActModel from a configuration dictionary.
    
    Args:
        config: Dictionary with model configuration.
    
    Returns:
        Configured TPsiActModel instance.
    """
    return TPsiActModel(
        backbone_name=config.get('backbone', 'resnet50'),
        num_classes=config.get('num_classes', 200),
        pretrained=config.get('pretrained', True),
        use_tpsiact=config.get('use_tpsiact', True),
        tpsiact_nu=config.get('tpsiact_nu', 5.0),
        replace_backbone_activations=config.get('replace_backbone_activations', False),
        hidden_dim=config.get('hidden_dim', None),
        dropout=config.get('dropout', 0.0),
        freeze_backbone=config.get('freeze_backbone', False)
    )


def get_available_backbones() -> List[str]:
    """
    Get list of recommended timm backbones for the experiments.
    """
    return [
        # ResNet family
        'resnet18', 'resnet34', 'resnet50', 'resnet101',
        # EfficientNet family
        'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
        # Vision Transformer family
        'vit_tiny_patch16_224', 'vit_small_patch16_224', 'vit_base_patch16_224',
        # Swin Transformer
        'swin_tiny_patch4_window7_224', 'swin_small_patch4_window7_224',
        # ConvNeXt
        'convnext_tiny', 'convnext_small', 'convnext_base',
        # MobileNet
        'mobilenetv3_small_100', 'mobilenetv3_large_100',
        # DenseNet
        'densenet121', 'densenet169',
    ]


if __name__ == "__main__":
    print("Testing model module...")
    
    # Test TPsiActModel
    model = TPsiActModel(
        backbone_name='resnet18',
        num_classes=200,
        pretrained=False,  # Don't download weights for testing
        use_tpsiact=True,
        tpsiact_nu=5.0
    )
    
    print(f"Model info: {model.get_model_info()}")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    
    # Basic forward
    logits = model(x)
    print(f"Logits shape: {logits.shape}")
    
    # With features
    logits, features = model(x, return_features=True)
    print(f"Features shape: {features.shape}")
    
    # With uncertainty
    logits, features, uncertainty = model(x, return_features=True, return_uncertainty=True)
    print(f"Uncertainty shape: {uncertainty.shape}")
    print(f"Uncertainty range: [{uncertainty.min().item():.4f}, {uncertainty.max().item():.4f}]")
    
    # Test feature extraction
    extractor = FeatureExtractor(model.backbone, model.feature_dim)
    features = extractor(x)
    print(f"Extracted features shape: {features.shape}")
    
    print("\nAll tests passed!")
