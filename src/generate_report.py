#!/usr/bin/env python3
"""
Generate final report from all experiment results.

This script collects results from all experiments in the experiments directory
and generates a comprehensive final report with the best model highlighted.
"""

import json
from pathlib import Path
import argparse
from typing import Dict, List, Optional
import csv


def load_experiment_results(experiments_dir: Path) -> List[Dict]:
    """Load results from all experiments."""
    results = []
    
    for exp_dir in sorted(experiments_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        
        # Look for final_results.json or summary.json
        result_file = exp_dir / 'final_results.json'
        if not result_file.exists():
            result_file = exp_dir / 'summary.json'
        
        if result_file.exists():
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    data['experiment_dir'] = str(exp_dir)
                    results.append(data)
            except Exception as e:
                print(f"Warning: Could not load {result_file}: {e}")
    
    return results


def print_summary_table(results: List[Dict]):
    """Print a summary table of all experiments."""
    print("\n" + "=" * 140)
    print(" " * 50 + "EXPERIMENT SUMMARY TABLE")
    print("=" * 140)
    
    # Header
    header = "| {:<35} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>12} |".format(
        "Experiment", "Test Acc", "Test F1", "k-NN Acc", "Best Epoch", "GPU-Hours", "Throughput"
    )
    print("-" * 140)
    print(header)
    print("-" * 140)
    
    # Sort by test accuracy
    sorted_results = sorted(
        results, 
        key=lambda x: x.get('final_metrics', {}).get('test_accuracy', x.get('best_test_acc', 0)), 
        reverse=True
    )
    
    for result in sorted_results:
        name = result.get('experiment_name', 'Unknown')[:35]
        
        # Handle different result formats
        if 'final_metrics' in result:
            metrics = result['final_metrics']
            test_acc = metrics.get('test_accuracy', 0)
            test_f1 = metrics.get('test_f1', 0)
            knn_acc = metrics.get('knn_accuracy', 0)
        else:
            test_acc = result.get('best_test_acc', 0)
            test_f1 = result.get('best_test_f1', 0)
            knn_acc = result.get('best_knn_accuracy', 0)
        
        if 'training_cost' in result:
            gpu_hours = result['training_cost'].get('gpu_hours', 0)
            throughput = result['training_cost'].get('throughput_img_s', 0)
        else:
            gpu_hours = result.get('total_gpu_hours', 0)
            throughput = result.get('avg_throughput_img_s', 0)
        
        best_epoch = result.get('best_epoch', 0)
        
        row = "| {:<35} | {:>10.4f} | {:>10.4f} | {:>10.4f} | {:>10d} | {:>10.4f}h | {:>8.1f}/s |".format(
            name, test_acc, test_f1, knn_acc, best_epoch, gpu_hours, throughput
        )
        print(row)
    
    print("-" * 140)
    
    return sorted_results[0] if sorted_results else None


def print_best_experiment(best: Dict):
    """Print detailed information about the best experiment."""
    print("\n" + "=" * 80)
    print(" " * 25 + "BEST EXPERIMENT")
    print("=" * 80)
    
    print(f"\n  Experiment Name: {best.get('experiment_name', 'Unknown')}")
    print(f"  Dataset:         {best.get('dataset', 'Unknown')}")
    print(f"  Backbone:        {best.get('backbone', 'Unknown')}")
    print(f"  TPsiAct:         {best.get('use_tpsiact', 'Unknown')}")
    
    if best.get('use_tpsiact'):
        print(f"  TPsiAct nu:      {best.get('tpsiact_nu', 'N/A')}")
    
    print("\n  Performance Metrics:")
    print("  " + "-" * 40)
    
    if 'final_metrics' in best:
        metrics = best['final_metrics']
        print(f"    Test Accuracy:   {metrics.get('test_accuracy', 0):.4f}")
        print(f"    Test F1 (macro): {metrics.get('test_f1', 0):.4f}")
        print(f"    k-NN Accuracy:   {metrics.get('knn_accuracy', 0):.4f}")
        print(f"    Train Accuracy:  {metrics.get('train_accuracy', 0):.4f}")
        print(f"    Train F1 (macro):{metrics.get('train_f1', 0):.4f}")
    else:
        print(f"    Test Accuracy:   {best.get('best_test_acc', 0):.4f}")
        print(f"    Test F1 (macro): {best.get('best_test_f1', 0):.4f}")
        print(f"    k-NN Accuracy:   {best.get('best_knn_accuracy', 0):.4f}")
    
    print("\n  Training Cost:")
    print("  " + "-" * 40)
    
    if 'training_cost' in best:
        cost = best['training_cost']
        print(f"    GPU Hours:       {cost.get('gpu_hours', 0):.4f}")
        print(f"    Throughput:      {cost.get('throughput_img_s', 0):.1f} img/s")
    else:
        print(f"    GPU Hours:       {best.get('total_gpu_hours', 0):.4f}")
        print(f"    Throughput:      {best.get('avg_throughput_img_s', 0):.1f} img/s")
    
    print(f"    Best Epoch:      {best.get('best_epoch', 0)}")
    
    print("\n" + "=" * 80)


def print_ablation_analysis(results: List[Dict]):
    """Print ablation study analysis."""
    print("\n" + "=" * 80)
    print(" " * 25 + "ABLATION ANALYSIS")
    print("=" * 80)
    
    # Group by different factors
    
    # 1. TPsiAct vs Baseline
    tpsiact_results = [r for r in results if r.get('use_tpsiact', False)]
    baseline_results = [r for r in results if not r.get('use_tpsiact', False)]
    
    if tpsiact_results and baseline_results:
        tpsiact_avg = sum(
            r.get('final_metrics', {}).get('test_accuracy', r.get('best_test_acc', 0))
            for r in tpsiact_results
        ) / len(tpsiact_results)
        
        baseline_avg = sum(
            r.get('final_metrics', {}).get('test_accuracy', r.get('best_test_acc', 0))
            for r in baseline_results
        ) / len(baseline_results)
        
        print(f"\n  TPsiAct vs Baseline:")
        print(f"    TPsiAct Average Test Acc:  {tpsiact_avg:.4f} ({len(tpsiact_results)} experiments)")
        print(f"    Baseline Average Test Acc: {baseline_avg:.4f} ({len(baseline_results)} experiments)")
        print(f"    Improvement:               {(tpsiact_avg - baseline_avg) * 100:.2f}%")
    
    # 2. Different nu values
    nu_results = {}
    for r in results:
        nu = r.get('tpsiact_nu')
        if nu is not None:
            if nu not in nu_results:
                nu_results[nu] = []
            nu_results[nu].append(r)
    
    if len(nu_results) > 1:
        print(f"\n  Effect of nu (degrees of freedom):")
        for nu, rs in sorted(nu_results.items()):
            avg_acc = sum(
                r.get('final_metrics', {}).get('test_accuracy', r.get('best_test_acc', 0))
                for r in rs
            ) / len(rs)
            print(f"    nu={nu}: Average Test Acc = {avg_acc:.4f}")
    
    print("\n" + "=" * 80)


def save_report(results: List[Dict], best: Dict, output_dir: Path):
    """Save report to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full report as JSON
    report = {
        'all_experiments': results,
        'best_experiment': best,
        'summary': {
            'total_experiments': len(results),
            'best_test_accuracy': best.get('final_metrics', {}).get('test_accuracy', best.get('best_test_acc', 0)) if best else 0,
            'best_experiment_name': best.get('experiment_name', 'Unknown') if best else None
        }
    }
    
    with open(output_dir / 'full_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Save summary CSV
    csv_file = output_dir / 'experiment_summary.csv'
    if results:
        fieldnames = ['experiment_name', 'dataset', 'backbone', 'use_tpsiact', 'tpsiact_nu',
                     'test_accuracy', 'test_f1', 'knn_accuracy', 'best_epoch', 'gpu_hours', 'throughput']
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for r in results:
                if 'final_metrics' in r:
                    metrics = r['final_metrics']
                    cost = r.get('training_cost', {})
                    row = {
                        'experiment_name': r.get('experiment_name', ''),
                        'dataset': r.get('dataset', ''),
                        'backbone': r.get('backbone', ''),
                        'use_tpsiact': r.get('use_tpsiact', False),
                        'tpsiact_nu': r.get('tpsiact_nu', ''),
                        'test_accuracy': metrics.get('test_accuracy', 0),
                        'test_f1': metrics.get('test_f1', 0),
                        'knn_accuracy': metrics.get('knn_accuracy', 0),
                        'best_epoch': r.get('best_epoch', 0),
                        'gpu_hours': cost.get('gpu_hours', 0),
                        'throughput': cost.get('throughput_img_s', 0)
                    }
                else:
                    row = {
                        'experiment_name': r.get('experiment_name', ''),
                        'dataset': r.get('dataset', ''),
                        'backbone': r.get('backbone', ''),
                        'use_tpsiact': r.get('use_tpsiact', False),
                        'tpsiact_nu': r.get('tpsiact_nu', ''),
                        'test_accuracy': r.get('best_test_acc', 0),
                        'test_f1': r.get('best_test_f1', 0),
                        'knn_accuracy': r.get('best_knn_accuracy', 0),
                        'best_epoch': r.get('best_epoch', 0),
                        'gpu_hours': r.get('total_gpu_hours', 0),
                        'throughput': r.get('avg_throughput_img_s', 0)
                    }
                writer.writerow(row)
    
    print(f"\nReport saved to: {output_dir}")
    print(f"  - {output_dir / 'full_report.json'}")
    print(f"  - {output_dir / 'experiment_summary.csv'}")


def main():
    parser = argparse.ArgumentParser(description='Generate final experiment report')
    parser.add_argument(
        '--experiments-dir', type=str, default='./experiments',
        help='Directory containing experiment results'
    )
    parser.add_argument(
        '--output-dir', type=str, default='./reports',
        help='Directory to save report files'
    )
    args = parser.parse_args()
    
    experiments_dir = Path(args.experiments_dir)
    output_dir = Path(args.output_dir)
    
    if not experiments_dir.exists():
        print(f"Error: Experiments directory not found: {experiments_dir}")
        return
    
    print("\n" + "=" * 140)
    print(" " * 50 + "TPsiAct FINAL EXPERIMENT REPORT")
    print("=" * 140)
    
    # Load results
    results = load_experiment_results(experiments_dir)
    
    if not results:
        print("\nNo experiment results found.")
        return
    
    print(f"\nFound {len(results)} experiment(s)")
    
    # Print summary table and get best
    best = print_summary_table(results)
    
    # Print best experiment details
    if best:
        print_best_experiment(best)
    
    # Print ablation analysis
    if len(results) > 1:
        print_ablation_analysis(results)
    
    # Save report
    save_report(results, best, output_dir)
    
    print("\n" + "=" * 140)
    print(" " * 55 + "END OF REPORT")
    print("=" * 140 + "\n")


if __name__ == '__main__':
    main()
