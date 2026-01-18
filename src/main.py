#!/usr/bin/env python3
"""
TPsiAct Experiment Main Entry Point

This script provides the main entry point for running TPsiAct experiments.
It supports:
- Multiple datasets (via ufgvc.py)
- Various backbones (via timm)
- TPsiAct activation with configurable parameters
- Comprehensive metrics tracking (train/test acc, F1, k-NN, throughput)
- Model saving and result export

Usage:
    python main.py --dataset cub_200_2011 --backbone resnet50 --epochs 100 --use-tpsiact
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import json
import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from ufgvc import UFGVCDataset
from model import TPsiActModel, get_available_backbones
from trainer import Trainer, TrainingConfig, ExperimentRunner
from utils import (
    set_seed, 
    get_device, 
    get_transforms, 
    create_dataloaders,
    print_config,
    print_dataset_info,
    print_model_info,
    get_gpu_info,
    generate_experiment_name,
    Logger
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='TPsiAct Experiment Runner',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset arguments
    dataset_group = parser.add_argument_group('Dataset')
    dataset_group.add_argument(
        '--dataset', type=str, default='cub_200_2011',
        choices=list(UFGVCDataset.DATASETS.keys()),
        help='Dataset to use for training'
    )
    dataset_group.add_argument(
        '--data-root', type=str, default='./data',
        help='Root directory for datasets'
    )
    dataset_group.add_argument(
        '--download', action='store_true',
        help='Download dataset if not present'
    )
    
    # Model arguments
    model_group = parser.add_argument_group('Model')
    model_group.add_argument(
        '--backbone', type=str, default='resnet50',
        help='Backbone model from timm (e.g., resnet50, vit_small_patch16_224)'
    )
    model_group.add_argument(
        '--pretrained', action='store_true', default=True,
        help='Use pretrained backbone weights'
    )
    model_group.add_argument(
        '--no-pretrained', action='store_false', dest='pretrained',
        help='Do not use pretrained backbone weights'
    )
    model_group.add_argument(
        '--use-tpsiact', action='store_true',
        help='Use TPsiAct activation in classifier'
    )
    model_group.add_argument(
        '--tpsiact-nu', type=float, default=5.0,
        help='Degrees of freedom (nu) for TPsiAct'
    )
    model_group.add_argument(
        '--replace-backbone-activations', action='store_true',
        help='Replace backbone activations with TPsiAct'
    )
    model_group.add_argument(
        '--hidden-dim', type=int, default=None,
        help='Hidden dimension for classifier (None for direct projection)'
    )
    model_group.add_argument(
        '--dropout', type=float, default=0.0,
        help='Dropout rate'
    )
    model_group.add_argument(
        '--freeze-backbone', action='store_true',
        help='Freeze backbone weights (linear probing)'
    )
    
    # Training arguments
    train_group = parser.add_argument_group('Training')
    train_group.add_argument(
        '--epochs', type=int, default=100,
        help='Number of training epochs'
    )
    train_group.add_argument(
        '--batch-size', type=int, default=32,
        help='Batch size for training'
    )
    train_group.add_argument(
        '--lr', '--learning-rate', type=float, default=1e-3,
        dest='learning_rate',
        help='Learning rate'
    )
    train_group.add_argument(
        '--weight-decay', type=float, default=1e-4,
        help='Weight decay'
    )
    train_group.add_argument(
        '--optimizer', type=str, default='adamw',
        choices=['adamw', 'adam', 'sgd'],
        help='Optimizer'
    )
    train_group.add_argument(
        '--scheduler', type=str, default='cosine',
        choices=['cosine', 'step', 'none'],
        help='Learning rate scheduler'
    )
    train_group.add_argument(
        '--warmup-epochs', type=int, default=5,
        help='Number of warmup epochs'
    )
    train_group.add_argument(
        '--label-smoothing', type=float, default=0.0,
        help='Label smoothing factor'
    )
    train_group.add_argument(
        '--grad-clip', type=float, default=1.0,
        help='Gradient clipping value (0 to disable)'
    )
    
    # Data augmentation
    aug_group = parser.add_argument_group('Augmentation')
    aug_group.add_argument(
        '--augmentation', type=str, default='standard',
        choices=['none', 'standard', 'autoaug', 'randaug'],
        help='Data augmentation strategy'
    )
    aug_group.add_argument(
        '--image-size', type=int, default=224,
        help='Input image size'
    )
    
    # k-NN evaluation
    knn_group = parser.add_argument_group('k-NN Evaluation')
    knn_group.add_argument(
        '--knn-k', type=int, default=200,
        help='Number of neighbors for k-NN evaluation'
    )
    knn_group.add_argument(
        '--knn-chunk-size', type=int, default=200,
        help='Chunk size for k-NN computation'
    )
    knn_group.add_argument(
        '--eval-knn-every', type=int, default=1,
        help='Evaluate k-NN every N epochs'
    )
    
    # Task type
    task_group = parser.add_argument_group('Task')
    task_group.add_argument(
        '--task', type=str, default='classification',
        choices=['classification', 'knn'],
        help='Task type (classification or k-NN only)'
    )
    
    # Saving options
    save_group = parser.add_argument_group('Saving')
    save_group.add_argument(
        '--save-dir', type=str, default='./experiments',
        help='Directory to save results'
    )
    save_group.add_argument(
        '--save-summary-json', action='store_true',
        help='Save summary to JSON file'
    )
    save_group.add_argument(
        '--save-summary-csv', action='store_true',
        help='Save epoch results to CSV file'
    )
    save_group.add_argument(
        '--save-final-pt', action='store_true',
        help='Save final model checkpoint'
    )
    save_group.add_argument(
        '--experiment-name', type=str, default=None,
        help='Custom experiment name (auto-generated if not provided)'
    )
    
    # System arguments
    sys_group = parser.add_argument_group('System')
    sys_group.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )
    sys_group.add_argument(
        '--num-workers', type=int, default=4,
        help='Number of data loading workers'
    )
    sys_group.add_argument(
        '--gpu', type=int, default=None,
        help='GPU ID to use (None for auto-detect)'
    )
    
    # Additional options for compatibility with existing scripts
    parser.add_argument('--tau', type=float, default=0.5, help='Temperature parameter (for compatibility)')
    parser.add_argument('--lambda-lie', type=float, default=1.0, help='Lambda parameter (for compatibility)')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Get device
    device = get_device(args.gpu)
    
    # Generate experiment name if not provided
    if args.experiment_name is None:
        args.experiment_name = generate_experiment_name(
            dataset=args.dataset,
            backbone=args.backbone,
            use_tpsiact=args.use_tpsiact,
            augmentation=args.augmentation,
            seed=args.seed
        )
    
    # Create save directory
    save_dir = Path(args.save_dir) / args.experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize logger
    logger = Logger(save_dir / 'training.log')
    
    # Print header
    print("\n" + "=" * 80)
    print(" " * 25 + "TPsiAct EXPERIMENT")
    print("=" * 80)
    print(f"Experiment: {args.experiment_name}")
    print(f"Device: {device}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 80)
    
    # Print GPU info
    gpu_info = get_gpu_info()
    if gpu_info['available']:
        print(f"\nGPU: {gpu_info['devices'][0]['name']}")
        print(f"Memory: {gpu_info['devices'][0]['total_memory_gb']:.1f} GB")
    
    # Print configuration
    config_dict = vars(args)
    print_config(config_dict)
    
    # Save configuration
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # =========================================================================
    # Load Dataset
    # =========================================================================
    print("\n" + "-" * 60)
    print("Loading Dataset...")
    print("-" * 60)
    
    # Get transforms
    train_transform = get_transforms(
        augmentation=args.augmentation,
        image_size=args.image_size,
        is_train=True
    )
    test_transform = get_transforms(
        augmentation=args.augmentation,
        image_size=args.image_size,
        is_train=False
    )
    
    # Create datasets
    train_dataset = UFGVCDataset(
        dataset_name=args.dataset,
        root=args.data_root,
        split='train',
        transform=train_transform,
        download=args.download
    )
    
    test_dataset = UFGVCDataset(
        dataset_name=args.dataset,
        root=args.data_root,
        split='test',
        transform=test_transform,
        download=args.download
    )
    
    # Get number of classes
    num_classes = len(train_dataset.classes)
    
    # Print dataset info
    print_dataset_info({
        'dataset_name': args.dataset,
        'description': UFGVCDataset.DATASETS[args.dataset]['description'],
        'train_samples': len(train_dataset),
        'test_samples': len(test_dataset),
        'num_classes': num_classes
    })
    
    # Create data loaders
    train_loader, test_loader = create_dataloaders(
        train_dataset,
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # =========================================================================
    # Create Model
    # =========================================================================
    print("\n" + "-" * 60)
    print("Creating Model...")
    print("-" * 60)
    
    model = TPsiActModel(
        backbone_name=args.backbone,
        num_classes=num_classes,
        pretrained=args.pretrained,
        use_tpsiact=args.use_tpsiact,
        tpsiact_nu=args.tpsiact_nu,
        replace_backbone_activations=args.replace_backbone_activations,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone
    )
    
    # Print model info
    model_info = model.get_model_info()
    print_model_info(model_info)
    
    # =========================================================================
    # Create Trainer
    # =========================================================================
    print("\n" + "-" * 60)
    print("Setting up Trainer...")
    print("-" * 60)
    
    training_config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        warmup_epochs=args.warmup_epochs,
        label_smoothing=args.label_smoothing,
        grad_clip=args.grad_clip,
        knn_k=args.knn_k,
        knn_chunk_size=args.knn_chunk_size,
        eval_knn_every=args.eval_knn_every
    )
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=training_config,
        device=device,
        num_classes=num_classes,
        save_dir=save_dir
    )
    
    # =========================================================================
    # Run Training
    # =========================================================================
    print("\n" + "-" * 60)
    print("Starting Training...")
    print("-" * 60)
    
    best_result = trainer.train()
    
    # =========================================================================
    # Final Report
    # =========================================================================
    print("\n")
    print("=" * 120)
    print(" " * 45 + "FINAL REPORT")
    print("=" * 120)
    
    # Load best model for final evaluation
    trainer.load_best_model()
    
    # Run final evaluation
    print("\nRunning final evaluation with best model (Epoch {})...".format(best_result.epoch))
    
    final_train_metrics = trainer.evaluate(train_loader, trainer.train_tracker)
    final_test_metrics = trainer.evaluate(test_loader, trainer.test_tracker)
    final_knn_metrics = trainer.evaluate_knn()
    
    # Print final metrics
    print("\n" + "-" * 80)
    print("BEST MODEL PERFORMANCE")
    print("-" * 80)
    print(f"  Best Epoch:       {best_result.epoch}")
    print(f"  Train Accuracy:   {final_train_metrics['accuracy']:.4f}")
    print(f"  Test Accuracy:    {final_test_metrics['accuracy']:.4f}")
    print(f"  Train F1 (macro): {final_train_metrics['f1_macro']:.4f}")
    print(f"  Test F1 (macro):  {final_test_metrics['f1_macro']:.4f}")
    print(f"  k-NN Accuracy:    {final_knn_metrics['knn_accuracy']:.4f}")
    print(f"  Total GPU-Hours:  {best_result.gpu_hours:.4f}")
    print(f"  Avg Throughput:   {best_result.throughput_img_s:.1f} img/s")
    print("-" * 80)
    
    # =========================================================================
    # Save Results
    # =========================================================================
    print("\n" + "-" * 60)
    print("Saving Results...")
    print("-" * 60)
    
    if args.save_summary_json:
        trainer.save_summary_json(save_dir / 'summary.json')
    
    if args.save_summary_csv:
        trainer.save_summary_csv(save_dir / 'summary.csv')
    
    if args.save_final_pt:
        trainer.save_model(save_dir / 'best_model.pt', save_best=True)
    
    # Save final results
    final_results = {
        'experiment_name': args.experiment_name,
        'dataset': args.dataset,
        'backbone': args.backbone,
        'use_tpsiact': args.use_tpsiact,
        'tpsiact_nu': args.tpsiact_nu,
        'best_epoch': best_result.epoch,
        'final_metrics': {
            'train_accuracy': final_train_metrics['accuracy'],
            'test_accuracy': final_test_metrics['accuracy'],
            'train_f1': final_train_metrics['f1_macro'],
            'test_f1': final_test_metrics['f1_macro'],
            'knn_accuracy': final_knn_metrics['knn_accuracy'],
            'knn_top5_accuracy': final_knn_metrics.get('knn_top5_accuracy', 0.0)
        },
        'training_cost': {
            'gpu_hours': best_result.gpu_hours,
            'throughput_img_s': best_result.throughput_img_s
        }
    }
    
    with open(save_dir / 'final_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nAll results saved to: {save_dir}")
    
    # Final summary line
    print("\n" + "=" * 120)
    print(f"EXPERIMENT COMPLETE: {args.experiment_name}")
    print(f"Best Test Accuracy: {final_test_metrics['accuracy']:.4f} (Epoch {best_result.epoch})")
    print("=" * 120 + "\n")
    
    return final_results


if __name__ == '__main__':
    main()
