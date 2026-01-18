"""
TPsiAct - t-Score Activation for Robust Deep Learning

A robust activation function based on Student-t distribution score function,
providing built-in outlier suppression and per-dimension uncertainty estimation.
"""

from .tpsiact import TPsiAct, TPsiActConv, TPsiActBlock, replace_activations_with_tpsiact
from .model import TPsiActModel, FeatureExtractor
from .trainer import Trainer, TrainingConfig, ExperimentRunner
from .metrics import MetricsTracker, KNNEvaluator, ThroughputTracker
from .ufgvc import UFGVCDataset

__version__ = '0.1.0'
__author__ = 'TPsiAct Team'

__all__ = [
    # Core TPsiAct
    'TPsiAct',
    'TPsiActConv', 
    'TPsiActBlock',
    'replace_activations_with_tpsiact',
    # Model
    'TPsiActModel',
    'FeatureExtractor',
    # Training
    'Trainer',
    'TrainingConfig',
    'ExperimentRunner',
    # Metrics
    'MetricsTracker',
    'KNNEvaluator',
    'ThroughputTracker',
    # Data
    'UFGVCDataset',
]
