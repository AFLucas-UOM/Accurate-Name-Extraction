import json
import argparse
import os
from typing import Dict, Set, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate  # For better table formatting
from collections import defaultdict

def load_ground_truth(path: str) -> Dict[str, Set[str]]:
    """Load and parse ground truth data from JSON file."""
    try:
        with open(path) as f:
            data = json.load(f)
        return {v["video"]: {n["name"] for n in v["names"]} for v in data}
    except FileNotFoundError:
        print(f"Error: Ground truth file '{path}' not found.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: '{path}' contains invalid JSON.")
        exit(1)
    except KeyError as e:
        print(f"Error: Ground truth file has unexpected format. Missing key: {e}")
        exit(1)

def load_predictions(path: str, time_key: str = "processing_time") -> Tuple[Dict[str, Set[str]], List[float]]:
    """Load and parse model predictions and processing times from JSON file."""
    if not os.path.exists(path):
        print(f"Warning: Prediction file '{path}' not found. Returning empty results.")
        return {}, []
    
    try:
        with open(path) as f:
            data = json.load(f)
        
        preds = {}
        times = []
        for v in data:
            video = v["video"]
            names = {n["name"] for n in v["names"]}
            preds[video] = names
            if v.get(time_key) is not None:
                times.append(float(v[time_key]))
        return preds, times
    except json.JSONDecodeError:
        print(f"Error: '{path}' contains invalid JSON.")
        return {}, []
    except KeyError as e:
        print(f"Warning: Prediction file has unexpected format. Missing key: {e}")
        return {}, []

def calculate_metrics(true_names: Set[str], pred_names: Set[str]) -> Tuple[float, float, float]:
    """Calculate precision, recall, and F1 for a single video."""
    tp = len(pred_names & true_names)  # Intersection (true positives)
    fp = len(pred_names - true_names)  # Predicted but not in ground truth (false positives)
    fn = len(true_names - pred_names)  # In ground truth but not predicted (false negatives)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

def evaluate(preds: Dict[str, Set[str]], gt: Dict[str, Set[str]]) -> Tuple[Dict[str, Tuple[float, float, float]], List[float], List[float], List[float]]:
    """Evaluate predictions against ground truth, returning per-video metrics and aggregates."""
    video_metrics = {}
    precisions, recalls, f1s = [], [], []
    
    # Track missing videos
    missing_videos = set(gt.keys()) - set(preds.keys())
    if missing_videos:
        print(f"Warning: {len(missing_videos)} videos in ground truth are missing from predictions.")
        if len(missing_videos) <= 5:
            print(f"Missing videos: {', '.join(missing_videos)}")
        else:
            print(f"First 5 missing videos: {', '.join(list(missing_videos)[:5])}")
    
    for video in gt:
        true_names = gt[video]
        pred_names = preds.get(video, set())
        
        # Calculate metrics for this video
        precision, recall, f1 = calculate_metrics(true_names, pred_names)
        
        video_metrics[video] = (precision, recall, f1)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    
    return video_metrics, precisions, recalls, f1s

def generate_summary_statistics(values: List[float]) -> Tuple[float, float, float, float]:
    """Calculate mean, median, std dev, and 95% confidence interval for a list of values."""
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    
    mean = np.mean(values)
    median = np.median(values)
    std_dev = np.std(values)
    ci_95 = 1.96 * std_dev / np.sqrt(len(values)) if len(values) > 1 else 0
    
    return mean, median, std_dev, ci_95

def plot_metrics(models: Dict[str, Tuple[Dict[str, Set[str]], List[float]]], gt: Dict[str, Set[str]], 
                output_dir: str = "./plots") -> None:
    """Generate and save comparison plots."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Collect metrics
    model_metrics = {}
    for name, (preds, times) in models.items():
        _, precisions, recalls, f1s = evaluate(preds, gt)
        model_metrics[name] = {
            'precision': np.mean(precisions),
            'recall': np.mean(recalls),
            'f1': np.mean(f1s),
            'time': np.mean(times) if times else 0
        }
    
    # Create bar charts
    metrics = ['precision', 'recall', 'f1']
    model_names = list(models.keys())
    
    # Metrics comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.25
    index = np.arange(len(model_names))
    
    for i, metric in enumerate(metrics):
        values = [model_metrics[model][metric] for model in model_names]
        ax.bar(index + i*bar_width, values, bar_width, label=metric.capitalize())
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics Comparison')
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/metrics_comparison.png")
    
    # Time comparison
    if any(model_metrics[model]['time'] > 0 for model in model_names):
        fig, ax = plt.subplots(figsize=(8, 5))
        times = [model_metrics[model]['time'] for model in model_names]
        ax.bar(model_names, times, color='skyblue')
        ax.set_xlabel('Model')
        ax.set_ylabel('Average Processing Time (s)')
        ax.set_title('Average Processing Time Comparison')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/time_comparison.png")
    
    print(f"Plots saved to {output_dir}/")

def export_results_to_csv(models: Dict[str, Tuple[Dict[str, Set[str]], List[float]]], 
                         gt: Dict[str, Set[str]], output_path: str) -> None:
    """Export detailed per-video results to CSV."""
    import csv
    
    # Prepare data
    rows = [["video", "model", "precision", "recall", "f1", "num_gt_names", "num_pred_names", "matching_names"]]
    
    for model_name, (preds, _) in models.items():
        for video in gt:
            true_names = gt[video]
            pred_names = preds.get(video, set())
            
            precision, recall, f1 = calculate_metrics(true_names, pred_names)
            matching = true_names & pred_names
            
            rows.append([
                video, 
                model_name, 
                f"{precision:.4f}", 
                f"{recall:.4f}", 
                f"{f1:.4f}",
                len(true_names),
                len(pred_names),
                "|".join(matching)
            ])
    
    # Write to CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    print(f"Detailed results exported to {output_path}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate name recognition models against ground truth.')
    parser.add_argument('--gt', default='GT.json', help='Path to ground truth JSON file')
    parser.add_argument('--anep', default='ANEP.json', help='Path to ANEP predictions')
    parser.add_argument('--google', default='Google.json', help='Path to Google predictions')
    parser.add_argument('--llama', default='Llama4.json', help='Path to Llama4 predictions')
    parser.add_argument('--output-csv', default='results.csv', help='Path to output CSV file')
    parser.add_argument('--plots', action='store_true', help='Generate comparison plots')
    parser.add_argument('--plot-dir', default='./plots', help='Directory to save plots')
    parser.add_argument('--detailed', action='store_true', help='Show detailed per-video statistics')
    parser.add_argument('--export-csv', action='store_true', help='Export detailed results to CSV')
    args = parser.parse_args()
    
    # Load data
    print(f"Loading ground truth from {args.gt}...")
    gt = load_ground_truth(args.gt)
    print(f"Found {len(gt)} videos in ground truth\n")
    
    # Load prediction files with error handling
    print("Loading model predictions...")
    anep_preds, anep_times = load_predictions(args.anep)
    google_preds, google_times = load_predictions(args.google)
    llama_preds, llama_times = load_predictions(args.llama, time_key="processing_time_seconds")
    
    # Define models
    models = {
        "ANEP": (anep_preds, anep_times),
        "Google": (google_preds, google_times),
        "LLaMA4": (llama_preds, llama_times),
    }
    
    # Print model counts for validation
    for name, (preds, times) in models.items():
        print(f"{name}: {len(preds)} videos, {len(times)} time measurements")
    print()
    
    # Evaluate models and collect results
    results = []
    detailed_results = defaultdict(list)
    
    for name, (preds, times) in models.items():
        video_metrics, precisions, recalls, f1s = evaluate(preds, gt)
        
        # Calculate summary statistics
        p_mean, p_median, p_std, p_ci = generate_summary_statistics(precisions)
        r_mean, r_median, r_std, r_ci = generate_summary_statistics(recalls)
        f1_mean, f1_median, f1_std, f1_ci = generate_summary_statistics(f1s)
        t_mean = np.mean(times) if times else 0
        t_std = np.std(times) if times else 0
        
        # Store results for table
        results.append([
            name, 
            f"{p_mean:.4f} ± {p_ci:.4f}", 
            f"{r_mean:.4f} ± {r_ci:.4f}", 
            f"{f1_mean:.4f} ± {f1_ci:.4f}",
            f"{t_mean:.2f} ± {t_std:.2f}" if times else "N/A"
        ])
        
        # Store detailed per-video results
        if args.detailed:
            for video, (p, r, f1) in video_metrics.items():
                detailed_results[video].append([name, f"{p:.4f}", f"{r:.4f}", f"{f1:.4f}"])
    
    # Print result table
    headers = ["Model", "Precision", "Recall", "F1", "Time (s)"]
    print("\nEvaluation Results:")
    print(tabulate(results, headers=headers, tablefmt="grid"))
    
    # Print detailed results if requested
    if args.detailed:
        print("\nDetailed Per-Video Results:")
        for video, model_results in detailed_results.items():
            print(f"\nVideo: {video}")
            print(tabulate(model_results, headers=["Model", "Precision", "Recall", "F1"], tablefmt="simple"))
    
    # Generate plots if requested
    if args.plots:
        plot_metrics(models, gt, args.plot_dir)
    
    # Export to CSV if requested
    if args.export_csv:
        export_results_to_csv(models, gt, args.output_csv)

if __name__ == "__main__":
    main()