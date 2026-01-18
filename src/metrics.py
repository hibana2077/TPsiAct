"""
Metrics module for TPsiAct experiments.

Contains:
- k-NN accuracy evaluation (frozen features)
- F1 score computation (macro/micro/weighted)
- Training throughput measurement (img/s, GPU-hours)
- Expected Calibration Error (ECE)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict
import time


class KNNEvaluator:
    """
    k-Nearest Neighbors evaluator for frozen feature evaluation.
    
    Computes k-NN accuracy using features extracted from a model.
    This is commonly used to evaluate representation quality without fine-tuning.
    """
    
    def __init__(
        self,
        k: int = 200,
        temperature: float = 0.07,
        chunk_size: int = 200,
        distance_metric: str = 'cosine'
    ):
        """
        Args:
            k: Number of nearest neighbors to consider.
            temperature: Temperature for softmax weighting.
            chunk_size: Process this many samples at a time to save memory.
            distance_metric: 'cosine' or 'euclidean'.
        """
        self.k = k
        self.temperature = temperature
        self.chunk_size = chunk_size
        self.distance_metric = distance_metric
        
        # Storage for features
        self.train_features: Optional[torch.Tensor] = None
        self.train_labels: Optional[torch.Tensor] = None
        self.num_classes: int = 0
    
    def fit(self, features: torch.Tensor, labels: torch.Tensor):
        """
        Store training features and labels for k-NN lookup.
        
        Args:
            features: (N, D) tensor of training features.
            labels: (N,) tensor of training labels.
        """
        # Normalize features for cosine similarity
        if self.distance_metric == 'cosine':
            features = F.normalize(features, dim=1)
        
        self.train_features = features
        self.train_labels = labels
        self.num_classes = int(labels.max().item()) + 1
    
    @torch.no_grad()
    def predict(
        self, 
        test_features: torch.Tensor,
        return_probs: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Predict labels for test features using k-NN.
        
        Args:
            test_features: (M, D) tensor of test features.
            return_probs: If True, also return soft probabilities.
        
        Returns:
            predictions: (M,) tensor of predicted labels.
            probs: (optional) (M, C) tensor of class probabilities.
        """
        if self.train_features is None:
            raise RuntimeError("Must call fit() before predict()")
        
        device = test_features.device
        train_features = self.train_features.to(device)
        train_labels = self.train_labels.to(device)
        
        # Normalize test features
        if self.distance_metric == 'cosine':
            test_features = F.normalize(test_features, dim=1)
        
        num_test = test_features.shape[0]
        all_predictions = []
        all_probs = [] if return_probs else None
        
        # Process in chunks to save memory
        for start_idx in range(0, num_test, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, num_test)
            chunk_features = test_features[start_idx:end_idx]
            
            # Compute similarities/distances
            if self.distance_metric == 'cosine':
                # Cosine similarity (higher is better)
                sim = torch.mm(chunk_features, train_features.t())
            else:
                # Euclidean distance (lower is better) -> negate for consistency
                diff = chunk_features.unsqueeze(1) - train_features.unsqueeze(0)
                sim = -torch.norm(diff, dim=2)
            
            # Get k nearest neighbors
            k = min(self.k, train_features.shape[0])
            topk_sim, topk_idx = sim.topk(k, dim=1)
            topk_labels = train_labels[topk_idx]
            
            # Weighted voting with temperature
            weights = F.softmax(topk_sim / self.temperature, dim=1)
            
            # Compute class probabilities
            chunk_probs = torch.zeros(chunk_features.shape[0], self.num_classes, device=device)
            chunk_probs.scatter_add_(
                dim=1,
                index=topk_labels,
                src=weights
            )
            
            # Predict class with highest probability
            chunk_predictions = chunk_probs.argmax(dim=1)
            all_predictions.append(chunk_predictions)
            
            if return_probs:
                all_probs.append(chunk_probs)
        
        predictions = torch.cat(all_predictions, dim=0)
        
        if return_probs:
            probs = torch.cat(all_probs, dim=0)
            return predictions, probs
        
        return predictions
    
    @torch.no_grad()
    def evaluate(
        self, 
        test_features: torch.Tensor,
        test_labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Evaluate k-NN accuracy.
        
        Returns:
            dict with 'knn_accuracy', 'knn_top5_accuracy'.
        """
        predictions, probs = self.predict(test_features, return_probs=True)
        
        # Top-1 accuracy
        correct = (predictions == test_labels.to(predictions.device)).float()
        top1_acc = correct.mean().item()
        
        # Top-5 accuracy
        top5_preds = probs.topk(min(5, self.num_classes), dim=1).indices
        top5_correct = (top5_preds == test_labels.to(predictions.device).unsqueeze(1)).any(dim=1).float()
        top5_acc = top5_correct.mean().item()
        
        return {
            'knn_accuracy': top1_acc,
            'knn_top5_accuracy': top5_acc
        }


class MetricsTracker:
    """
    Tracks and computes various training metrics.
    
    Supports:
    - Accuracy (train/test)
    - F1 score (macro/micro/weighted)
    - Loss tracking
    - Training throughput (img/s, GPU-hours)
    """
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.predictions = []
        self.labels = []
        self.losses = []
        self.num_samples = 0
        self.batch_times = []
        self.batch_sizes = []
    
    def update(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        loss: Optional[float] = None,
        batch_time: Optional[float] = None,
        batch_size: Optional[int] = None
    ):
        """
        Update metrics with a batch of predictions.
        
        Args:
            predictions: (N, C) logits or (N,) predicted labels.
            labels: (N,) ground truth labels.
            loss: Optional loss value.
            batch_time: Optional batch processing time in seconds.
            batch_size: Optional batch size for throughput calculation.
        """
        # Convert logits to predictions if needed
        if predictions.dim() > 1:
            predictions = predictions.argmax(dim=1)
        
        self.predictions.append(predictions.detach().cpu())
        self.labels.append(labels.detach().cpu())
        
        if loss is not None:
            self.losses.append(loss)
        
        if batch_time is not None:
            self.batch_times.append(batch_time)
        
        if batch_size is not None:
            self.batch_sizes.append(batch_size)
            self.num_samples += batch_size
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all tracked metrics.
        
        Returns:
            dict with keys: 'accuracy', 'f1_macro', 'f1_micro', 'f1_weighted',
                           'loss', 'throughput_img_s', etc.
        """
        if not self.predictions:
            return {}
        
        all_preds = torch.cat(self.predictions, dim=0).numpy()
        all_labels = torch.cat(self.labels, dim=0).numpy()
        
        results = {}
        
        # Accuracy
        results['accuracy'] = float((all_preds == all_labels).mean())
        
        # F1 scores
        results['f1_macro'] = self._compute_f1(all_preds, all_labels, 'macro')
        results['f1_micro'] = self._compute_f1(all_preds, all_labels, 'micro')
        results['f1_weighted'] = self._compute_f1(all_preds, all_labels, 'weighted')
        
        # Loss
        if self.losses:
            results['loss'] = float(np.mean(self.losses))
        
        # Throughput
        if self.batch_times and self.batch_sizes:
            total_time = sum(self.batch_times)
            total_samples = sum(self.batch_sizes)
            results['throughput_img_s'] = total_samples / total_time if total_time > 0 else 0
            results['total_time_s'] = total_time
        
        return results
    
    def _compute_f1(
        self, 
        predictions: np.ndarray, 
        labels: np.ndarray,
        average: str = 'macro'
    ) -> float:
        """Compute F1 score with specified averaging."""
        # Confusion matrix computation
        cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        for pred, label in zip(predictions, labels):
            cm[label, pred] += 1
        
        # Per-class precision, recall, F1
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        
        # Avoid division by zero
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0)
        f1_per_class = np.where(
            precision + recall > 0,
            2 * precision * recall / (precision + recall),
            0
        )
        
        if average == 'macro':
            return float(f1_per_class.mean())
        elif average == 'micro':
            # Micro F1 = accuracy for multiclass
            tp_total = tp.sum()
            fp_total = fp.sum()
            fn_total = fn.sum()
            micro_precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
            micro_recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
            if micro_precision + micro_recall > 0:
                return float(2 * micro_precision * micro_recall / (micro_precision + micro_recall))
            return 0.0
        elif average == 'weighted':
            support = cm.sum(axis=1)
            weights = support / support.sum() if support.sum() > 0 else np.zeros_like(support)
            return float((f1_per_class * weights).sum())
        else:
            raise ValueError(f"Unknown average: {average}")


class ThroughputTracker:
    """
    Tracks training throughput in img/s and GPU-hours.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset tracking."""
        self.start_time: Optional[float] = None
        self.total_images = 0
        self.epoch_images = 0
        self.epoch_start_time: Optional[float] = None
    
    def start_epoch(self):
        """Mark the start of an epoch."""
        self.epoch_start_time = time.time()
        self.epoch_images = 0
        if self.start_time is None:
            self.start_time = self.epoch_start_time
    
    def update(self, batch_size: int):
        """Update with a processed batch."""
        self.total_images += batch_size
        self.epoch_images += batch_size
    
    def end_epoch(self) -> Dict[str, float]:
        """
        Get epoch throughput statistics.
        
        Returns:
            dict with 'epoch_img_s', 'epoch_time_s', 'total_img_s', 
                      'total_time_s', 'total_gpu_hours'
        """
        current_time = time.time()
        epoch_time = current_time - self.epoch_start_time if self.epoch_start_time else 0
        total_time = current_time - self.start_time if self.start_time else 0
        
        return {
            'epoch_img_s': self.epoch_images / epoch_time if epoch_time > 0 else 0,
            'epoch_time_s': epoch_time,
            'total_img_s': self.total_images / total_time if total_time > 0 else 0,
            'total_time_s': total_time,
            'total_gpu_hours': total_time / 3600
        }


class ECECalculator:
    """
    Expected Calibration Error (ECE) calculator.
    
    Measures how well the predicted probabilities match actual accuracy.
    """
    
    def __init__(self, n_bins: int = 15):
        self.n_bins = n_bins
    
    def compute(
        self, 
        confidences: torch.Tensor,
        predictions: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute ECE.
        
        Args:
            confidences: (N,) tensor of prediction confidences (max prob).
            predictions: (N,) tensor of predicted labels.
            labels: (N,) tensor of ground truth labels.
        
        Returns:
            dict with 'ece', 'mce' (maximum calibration error).
        """
        confidences = confidences.detach().cpu().numpy()
        predictions = predictions.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        ece = 0.0
        mce = 0.0
        
        for i in range(self.n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = (predictions[in_bin] == labels[in_bin]).mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return {
            'ece': float(ece),
            'mce': float(mce)
        }


def extract_features(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    feature_dim: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract features from a model for all samples in a dataloader.
    
    Args:
        model: Feature extractor model (should output features, not logits).
        dataloader: DataLoader with (images, labels) or (images, labels, idx).
        device: Device to run inference on.
        feature_dim: Expected feature dimension (for pre-allocation).
    
    Returns:
        features: (N, D) tensor of features.
        labels: (N,) tensor of labels.
    """
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch
            
            images = images.to(device)
            features = model(images)
            
            # Handle different output formats
            if isinstance(features, tuple):
                features = features[0]  # Take first element if tuple
            
            # Flatten if needed (e.g., from conv output)
            if features.dim() > 2:
                features = features.view(features.size(0), -1)
            
            all_features.append(features.cpu())
            all_labels.append(labels)
    
    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)


def compute_all_metrics(
    train_preds: torch.Tensor,
    train_labels: torch.Tensor,
    test_preds: torch.Tensor,
    test_labels: torch.Tensor,
    train_features: Optional[torch.Tensor] = None,
    test_features: Optional[torch.Tensor] = None,
    knn_k: int = 200,
    num_classes: int = 10,
    throughput_stats: Optional[Dict] = None
) -> Dict[str, float]:
    """
    Compute all metrics for reporting.
    
    Returns:
        dict with all metrics including train/test accuracy, F1, k-NN, throughput.
    """
    results = {}
    
    # Train metrics
    train_tracker = MetricsTracker(num_classes)
    train_tracker.update(train_preds, train_labels)
    train_metrics = train_tracker.compute()
    results['train_acc'] = train_metrics.get('accuracy', 0.0)
    results['train_f1'] = train_metrics.get('f1_macro', 0.0)
    
    # Test metrics
    test_tracker = MetricsTracker(num_classes)
    test_tracker.update(test_preds, test_labels)
    test_metrics = test_tracker.compute()
    results['test_acc'] = test_metrics.get('accuracy', 0.0)
    results['test_f1'] = test_metrics.get('f1_macro', 0.0)
    
    # k-NN metrics (if features provided)
    if train_features is not None and test_features is not None:
        knn = KNNEvaluator(k=knn_k)
        knn.fit(train_features, train_labels)
        knn_results = knn.evaluate(test_features, test_labels)
        results['knn_accuracy'] = knn_results['knn_accuracy']
    else:
        results['knn_accuracy'] = 0.0
    
    # Throughput
    if throughput_stats:
        results['throughput_img_s'] = throughput_stats.get('epoch_img_s', 0.0)
        results['gpu_hours'] = throughput_stats.get('total_gpu_hours', 0.0)
    else:
        results['throughput_img_s'] = 0.0
        results['gpu_hours'] = 0.0
    
    return results


if __name__ == "__main__":
    # Test metrics
    print("Testing metrics module...")
    
    # Test MetricsTracker
    tracker = MetricsTracker(num_classes=10)
    
    # Simulate batches
    for _ in range(5):
        preds = torch.randint(0, 10, (32,))
        labels = torch.randint(0, 10, (32,))
        tracker.update(preds, labels, loss=0.5, batch_time=0.1, batch_size=32)
    
    metrics = tracker.compute()
    print(f"Computed metrics: {metrics}")
    
    # Test KNN
    print("\nTesting KNN evaluator...")
    knn = KNNEvaluator(k=5)
    
    # Simulate features
    train_features = torch.randn(100, 64)
    train_labels = torch.randint(0, 5, (100,))
    test_features = torch.randn(20, 64)
    test_labels = torch.randint(0, 5, (20,))
    
    knn.fit(train_features, train_labels)
    knn_metrics = knn.evaluate(test_features, test_labels)
    print(f"KNN metrics: {knn_metrics}")
    
    print("\nAll tests passed!")
