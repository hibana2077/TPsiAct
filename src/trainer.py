"""
Trainer module for TPsiAct experiments.

Object-oriented training infrastructure with:
- Per-epoch metrics printing (no TQDM)
- Train/test accuracy, F1, k-NN accuracy tracking
- Throughput measurement (img/s, GPU-hours)
- Best model selection based on test accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any, Callable
import time
import json
import csv
from pathlib import Path
from dataclasses import dataclass, field, asdict
from collections import defaultdict

from metrics import (
    MetricsTracker, 
    KNNEvaluator, 
    ThroughputTracker,
    extract_features
)


@dataclass
class EpochResult:
    """Container for epoch results."""
    epoch: int
    train_acc: float
    test_acc: float
    train_f1: float
    test_f1: float
    knn_accuracy: float
    throughput_img_s: float
    gpu_hours: float
    train_loss: float = 0.0
    test_loss: float = 0.0
    learning_rate: float = 0.0


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = 'adamw'
    scheduler: str = 'cosine'
    warmup_epochs: int = 5
    label_smoothing: float = 0.0
    grad_clip: float = 1.0
    knn_k: int = 200
    knn_chunk_size: int = 200
    eval_knn_every: int = 1  # Evaluate k-NN every N epochs


class Trainer:
    """
    Main trainer class for TPsiAct experiments.
    
    Features:
    - Object-oriented design
    - Custom epoch logging format (no TQDM)
    - Tracks all required metrics
    - Best model selection based on test accuracy
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        config: TrainingConfig,
        device: torch.device,
        num_classes: int,
        save_dir: Optional[Path] = None
    ):
        """
        Args:
            model: Model to train.
            train_loader: Training data loader.
            test_loader: Test data loader.
            config: Training configuration.
            device: Device to train on.
            num_classes: Number of classes.
            save_dir: Directory to save results.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.num_classes = num_classes
        self.save_dir = Path(save_dir) if save_dir else Path('.')
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup training components
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Metrics tracking
        self.train_tracker = MetricsTracker(num_classes)
        self.test_tracker = MetricsTracker(num_classes)
        self.throughput_tracker = ThroughputTracker()
        self.knn_evaluator = KNNEvaluator(
            k=config.knn_k,
            chunk_size=config.knn_chunk_size
        )
        
        # Results storage
        self.epoch_results: List[EpochResult] = []
        self.best_result: Optional[EpochResult] = None
        self.best_model_state: Optional[Dict] = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config."""
        params = self.model.parameters()
        
        if self.config.optimizer.lower() == 'adamw':
            return optim.AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == 'adam':
            return optim.Adam(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == 'sgd':
            return optim.SGD(
                params,
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler based on config."""
        if self.config.scheduler.lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs - self.config.warmup_epochs,
                eta_min=1e-6
            )
        elif self.config.scheduler.lower() == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif self.config.scheduler.lower() == 'none':
            return None
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")
    
    def _warmup_lr(self, epoch: int, step: int, total_steps: int):
        """Apply warmup learning rate."""
        if epoch < self.config.warmup_epochs:
            warmup_factor = (epoch * total_steps + step + 1) / (self.config.warmup_epochs * total_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.config.learning_rate * warmup_factor
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.train_tracker.reset()
        self.throughput_tracker.start_epoch()
        
        total_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            batch_start = time.time()
            
            # Unpack batch
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch
            
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Apply warmup
            self._warmup_lr(self.current_epoch, batch_idx, total_batches)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip
                )
            
            self.optimizer.step()
            
            # Track metrics
            batch_time = time.time() - batch_start
            self.train_tracker.update(
                outputs,
                labels,
                loss=loss.item(),
                batch_time=batch_time,
                batch_size=images.size(0)
            )
            self.throughput_tracker.update(images.size(0))
            
            self.global_step += 1
            
            # Print progress every 10% of batches
            if (batch_idx + 1) % max(1, total_batches // 10) == 0:
                progress = (batch_idx + 1) / total_batches * 100
                print(f"  Batch {batch_idx + 1}/{total_batches} ({progress:.0f}%) - Loss: {loss.item():.4f}")
        
        return self.train_tracker.compute()
    
    @torch.no_grad()
    def evaluate(self, loader: DataLoader, tracker: MetricsTracker) -> Dict[str, float]:
        """Evaluate model on a data loader."""
        self.model.eval()
        tracker.reset()
        
        for batch in loader:
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch
            
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            tracker.update(outputs, labels, loss=loss.item())
        
        return tracker.compute()
    
    @torch.no_grad()
    def evaluate_knn(self) -> Dict[str, float]:
        """Evaluate k-NN accuracy using frozen features."""
        self.model.eval()
        
        # Extract features
        def get_features(loader):
            all_features = []
            all_labels = []
            
            for batch in loader:
                if len(batch) == 3:
                    images, labels, _ = batch
                else:
                    images, labels = batch
                
                images = images.to(self.device)
                
                # Get features from model
                if hasattr(self.model, 'get_features'):
                    features = self.model.get_features(images)
                else:
                    # Fallback: use forward with return_features if available
                    outputs = self.model(images, return_features=True)
                    if isinstance(outputs, tuple):
                        features = outputs[1]
                    else:
                        features = outputs
                
                all_features.append(features.cpu())
                all_labels.append(labels)
            
            return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)
        
        train_features, train_labels = get_features(self.train_loader)
        test_features, test_labels = get_features(self.test_loader)
        
        # Fit and evaluate k-NN
        self.knn_evaluator.fit(train_features, train_labels)
        return self.knn_evaluator.evaluate(test_features, test_labels)
    
    def print_epoch_header(self):
        """Print the header for epoch results."""
        print("=" * 100)
        print("Training Progress")
        print("=" * 100)
    
    def print_epoch_result(self, result: EpochResult):
        """Print epoch result in the specified format."""
        line = (
            f"Epoch:{result.epoch}|"
            f"Train Acc:{result.train_acc:.4f}|"
            f"Test Acc:{result.test_acc:.4f}|"
            f"Train F1:{result.train_f1:.4f}|"
            f"Test F1:{result.test_f1:.4f}|"
            f"k-NN Acc:{result.knn_accuracy:.4f}|"
            f"Throughput:{result.throughput_img_s:.1f}img/s|"
            f"GPU-Hours:{result.gpu_hours:.4f}"
        )
        print(line)
    
    def train(self) -> EpochResult:
        """
        Run full training loop.
        
        Returns:
            Best epoch result based on test accuracy.
        """
        print("\n" + "=" * 120)
        print("Starting Training")
        print(f"Total epochs: {self.config.epochs}")
        print(f"Device: {self.device}")
        print(f"k-NN k: {self.config.knn_k}")
        print("=" * 120)
        
        self.print_epoch_header()
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch()
            
            # Evaluate on test set
            test_metrics = self.evaluate(self.test_loader, self.test_tracker)
            
            # Evaluate k-NN (every N epochs or last epoch)
            if (epoch + 1) % self.config.eval_knn_every == 0 or epoch == self.config.epochs - 1:
                knn_metrics = self.evaluate_knn()
            else:
                knn_metrics = {'knn_accuracy': self.epoch_results[-1].knn_accuracy if self.epoch_results else 0.0}
            
            # Update scheduler
            if self.scheduler is not None and epoch >= self.config.warmup_epochs:
                self.scheduler.step()
            
            # Get throughput stats
            throughput_stats = self.throughput_tracker.end_epoch()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Create epoch result
            result = EpochResult(
                epoch=epoch + 1,
                train_acc=train_metrics.get('accuracy', 0.0),
                test_acc=test_metrics.get('accuracy', 0.0),
                train_f1=train_metrics.get('f1_macro', 0.0),
                test_f1=test_metrics.get('f1_macro', 0.0),
                knn_accuracy=knn_metrics.get('knn_accuracy', 0.0),
                throughput_img_s=throughput_stats.get('epoch_img_s', 0.0),
                gpu_hours=throughput_stats.get('total_gpu_hours', 0.0),
                train_loss=train_metrics.get('loss', 0.0),
                test_loss=test_metrics.get('loss', 0.0),
                learning_rate=current_lr
            )
            
            # Print result
            self.print_epoch_result(result)
            
            # Store result
            self.epoch_results.append(result)
            
            # Check if best
            if self.best_result is None or result.test_acc > self.best_result.test_acc:
                self.best_result = result
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
        
        print("=" * 120)
        print("Training Complete!")
        print("=" * 120)
        
        return self.best_result
    
    def load_best_model(self):
        """Load the best model state."""
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        if not self.epoch_results:
            return {}
        
        last_result = self.epoch_results[-1]
        best_result = self.best_result
        
        return {
            'total_epochs': len(self.epoch_results),
            'total_gpu_hours': last_result.gpu_hours,
            'best_epoch': best_result.epoch if best_result else 0,
            'best_test_acc': best_result.test_acc if best_result else 0.0,
            'best_test_f1': best_result.test_f1 if best_result else 0.0,
            'best_knn_accuracy': best_result.knn_accuracy if best_result else 0.0,
            'final_train_acc': last_result.train_acc,
            'final_test_acc': last_result.test_acc,
            'final_train_f1': last_result.train_f1,
            'final_test_f1': last_result.test_f1,
            'final_knn_accuracy': last_result.knn_accuracy,
            'avg_throughput_img_s': sum(r.throughput_img_s for r in self.epoch_results) / len(self.epoch_results),
            'config': asdict(self.config)
        }
    
    def save_summary_json(self, filepath: Optional[Path] = None):
        """Save training summary to JSON."""
        filepath = filepath or self.save_dir / 'summary.json'
        summary = self.get_summary()
        
        # Add epoch history
        summary['epoch_history'] = [asdict(r) for r in self.epoch_results]
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary saved to {filepath}")
    
    def save_summary_csv(self, filepath: Optional[Path] = None):
        """Save epoch results to CSV."""
        filepath = filepath or self.save_dir / 'summary.csv'
        
        if not self.epoch_results:
            return
        
        fieldnames = list(asdict(self.epoch_results[0]).keys())
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in self.epoch_results:
                writer.writerow(asdict(result))
        
        print(f"Results saved to {filepath}")
    
    def save_model(self, filepath: Optional[Path] = None, save_best: bool = True):
        """Save model checkpoint."""
        filepath = filepath or self.save_dir / 'final_model.pt'
        
        state = {
            'model_state_dict': self.best_model_state if save_best and self.best_model_state else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': asdict(self.config),
            'best_result': asdict(self.best_result) if self.best_result else None,
            'epoch': self.current_epoch + 1
        }
        
        if self.scheduler is not None:
            state['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(state, filepath)
        print(f"Model saved to {filepath}")


class ExperimentRunner:
    """
    Runs multiple experiments and generates final report.
    
    Tracks all experiments and selects the best based on test accuracy.
    """
    
    def __init__(
        self,
        save_dir: Path,
        save_summary_json: bool = True,
        save_summary_csv: bool = True,
        save_final_pt: bool = True
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_summary_json = save_summary_json
        self.save_summary_csv = save_summary_csv
        self.save_final_pt = save_final_pt
        
        self.experiment_results: List[Dict] = []
        self.best_experiment: Optional[Dict] = None
    
    def add_experiment(
        self,
        name: str,
        trainer: Trainer,
        config: Dict[str, Any]
    ):
        """Add an experiment result."""
        summary = trainer.get_summary()
        summary['experiment_name'] = name
        summary['experiment_config'] = config
        
        self.experiment_results.append(summary)
        
        # Check if best
        if (self.best_experiment is None or 
            summary.get('best_test_acc', 0) > self.best_experiment.get('best_test_acc', 0)):
            self.best_experiment = summary
    
    def print_final_report(self):
        """Print comprehensive final report."""
        if not self.experiment_results:
            print("No experiments to report.")
            return
        
        print("\n")
        print("=" * 120)
        print(" " * 45 + "FINAL EXPERIMENT REPORT")
        print("=" * 120)
        
        # Summary table header
        print("\n" + "-" * 120)
        print("| {:20s} | {:10s} | {:10s} | {:10s} | {:10s} | {:12s} | {:10s} |".format(
            "Experiment", "Test Acc", "Test F1", "k-NN Acc", "Best Epoch", "GPU-Hours", "Throughput"
        ))
        print("-" * 120)
        
        # Print each experiment
        for result in self.experiment_results:
            print("| {:20s} | {:10.4f} | {:10.4f} | {:10.4f} | {:10d} | {:12.4f} | {:8.1f}/s |".format(
                result.get('experiment_name', 'Unknown')[:20],
                result.get('best_test_acc', 0.0),
                result.get('best_test_f1', 0.0),
                result.get('best_knn_accuracy', 0.0),
                result.get('best_epoch', 0),
                result.get('total_gpu_hours', 0.0),
                result.get('avg_throughput_img_s', 0.0)
            ))
        
        print("-" * 120)
        
        # Best experiment highlight
        if self.best_experiment:
            print("\n" + "=" * 60)
            print(" " * 15 + "BEST EXPERIMENT")
            print("=" * 60)
            print(f"  Name:         {self.best_experiment.get('experiment_name', 'Unknown')}")
            print(f"  Test Acc:     {self.best_experiment.get('best_test_acc', 0.0):.4f}")
            print(f"  Test F1:      {self.best_experiment.get('best_test_f1', 0.0):.4f}")
            print(f"  k-NN Acc:     {self.best_experiment.get('best_knn_accuracy', 0.0):.4f}")
            print(f"  Best Epoch:   {self.best_experiment.get('best_epoch', 0)}")
            print(f"  GPU-Hours:    {self.best_experiment.get('total_gpu_hours', 0.0):.4f}")
            print("=" * 60)
        
        print("\n" + "=" * 120)
        print(" " * 50 + "END OF REPORT")
        print("=" * 120 + "\n")
    
    def save_all_results(self):
        """Save all experiment results."""
        if self.save_summary_json:
            filepath = self.save_dir / 'all_experiments.json'
            with open(filepath, 'w') as f:
                json.dump({
                    'experiments': self.experiment_results,
                    'best_experiment': self.best_experiment
                }, f, indent=2)
            print(f"All experiments saved to {filepath}")
        
        if self.save_summary_csv:
            filepath = self.save_dir / 'all_experiments.csv'
            if self.experiment_results:
                # Flatten nested dicts for CSV
                rows = []
                for result in self.experiment_results:
                    row = {
                        'experiment_name': result.get('experiment_name', ''),
                        'best_test_acc': result.get('best_test_acc', 0.0),
                        'best_test_f1': result.get('best_test_f1', 0.0),
                        'best_knn_accuracy': result.get('best_knn_accuracy', 0.0),
                        'best_epoch': result.get('best_epoch', 0),
                        'total_epochs': result.get('total_epochs', 0),
                        'total_gpu_hours': result.get('total_gpu_hours', 0.0),
                        'avg_throughput_img_s': result.get('avg_throughput_img_s', 0.0)
                    }
                    rows.append(row)
                
                with open(filepath, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
                
                print(f"All experiments CSV saved to {filepath}")


if __name__ == "__main__":
    print("Trainer module loaded successfully.")
