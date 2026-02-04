"""
Pipeline for Conformal Prediction with Bernoulli Prediction Sets (BPS)

This pipeline handles:
1. Loading/generating predictions from uncertainty-aware models
2. Creating conformal prediction sets (BPS and APS)
3. Evaluating and printing results

Usage:
    python pipeline.py --dataset cifar10 --model ensemble --alpha 0.1 --calib_size 0.5 --n_seeds 10
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from pathlib import Path
import pickle
import argparse

import numpy as np
import torch
from sklearn.model_selection import train_test_split

ROOT_PATH = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(ROOT_PATH))

from epiuc.conformal_prediction.uacp import (
    solve_b_in_batches, 
    find_optimal_lambda, 
    avg_true_label_inclusion
)
from epiuc.conformal_prediction.eval_metrics import (
    covGap_nonbinary, 
    size_stratified_cov_violation_nonbinary,
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PipelineConfig: 
    dataset: str = "cifar10"
    model: str = "ensemble"
    
    # cp parameters
    alpha: float = 0.1  # significance level (1 - alpha = coverage target)
    calib_size: float = 0.5  # fraction of data for calibration
    n_seeds: int = 10  # number of random seeds for evaluation
    batch_size: int = 100  # batch size for optimization
    
    # Approaches to evaluate
    approaches: List[str] = field(default_factory=lambda: [
        "BPS", "APS", "BPS_cons", "APS_cons", "BPS_nom", "APS_nom"
    ])
    
    # Paths
    results_dir: Path = field(default_factory=lambda: ROOT_PATH / "all_results")
    data_dir: Path = field(default_factory=lambda: ROOT_PATH / "data")
    
    # Optional: soft labels path (for conditional coverage evaluation)
    soft_labels_path: Optional[Path] = None
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    
    @property
    def nominal_coverage(self) -> float:
        return 1 - self.alpha
    
    @property
    def predictions_path(self) -> Path:
        return self.results_dir / self.dataset / self.model / "predictions.pt"
    
    def results_path(self, seed: Optional[int] = None) -> Path:
        base = self.results_dir / self.dataset / self.model / f"calib_size {self.calib_size}" / f"alpha {self.alpha}"
        if seed is not None:
            return base / f"seed {seed}" / "res.pkl"
        return base / "results.pkl"


# =============================================================================
# Data Loading
# =============================================================================

def load_predictions(config: PipelineConfig) -> Dict[str, torch.Tensor]:
    """Load predictions from file."""
    path = config.predictions_path
    if not path.exists():
        raise FileNotFoundError(
            f"Predictions not found at {path}. "
            f"Run predict_from_model.py first."
        )
    
    with open(path, 'rb') as f:
        predictions = torch.load(f, map_location='cpu', weights_only=False)
    
    print(f"Loaded predictions from {path}")
    print(f"- Labels shape: {predictions['labels'].shape}")
    print(f"- Probs shape: {predictions['probs'].shape}")
    print(f"- Mean probs shape: {predictions['mean_probs'].shape}")
    
    return predictions


def load_soft_labels(config: PipelineConfig) -> Optional[np.ndarray]:
    if config.soft_labels_path is not None:
        path = config.soft_labels_path
    elif config.dataset == "cifar10":
        path = config.data_dir / "CIFAR-10-H" / "cifar10h-probs.npy"
    else:
        return None
    
    if path.exists():
        soft_labels = np.load(path)
        print(f"Loaded soft labels from {path}, shape: {soft_labels.shape}")
        return soft_labels
    
    print("No soft labels found.")
    return None


# =============================================================================
# Prediction Set Generation
# =============================================================================

def compute_cp_target(nominal_coverage: float, calibration_size: int) -> float:
    """Compute the finite-sample corrected target coverage."""
    return np.ceil(nominal_coverage * (calibration_size + 1)) / calibration_size


def create_prediction_sets(
    config: PipelineConfig,
    predictions: Dict[str, torch.Tensor],
    seed: int
) -> Dict[str, Any]:
    """
    Create prediction sets for a single seed.
    
    Returns dict with opt_lambda and opt_b for each approach.
    """
    labels = predictions["labels"].numpy()
    probs = predictions["probs"].numpy()
    mean_probs = predictions["mean_probs"].numpy()
    
    calib_probs, test_probs, calib_mean_probs, test_mean_probs, calib_labels, test_labels = \
        train_test_split(probs, mean_probs, labels, train_size=config.calib_size, random_state=seed)
    
    calibration_size = len(calib_labels)
    CP_target = compute_cp_target(config.nominal_coverage, calibration_size)
    
    results = {
        "opt_b": {},
        "opt_lambda": {},
        "test_labels": test_labels,
        "test_probs": test_probs,
        "test_mean_probs": test_mean_probs,
    }
    
    # Find optimal lambda for BPS and APS
    opt_lambda_BPS, _ = find_optimal_lambda(
        calib_probs, calib_labels, target=CP_target, batch_size=config.batch_size
    )
    opt_lambda_APS, _ = find_optimal_lambda(
        calib_mean_probs, calib_labels, target=CP_target, batch_size=config.batch_size
    )
    
    # Store lambdas for all variants
    results["opt_lambda"]["BPS"] = opt_lambda_BPS
    results["opt_lambda"]["BPS_cons"] = max(opt_lambda_BPS, config.nominal_coverage)
    results["opt_lambda"]["BPS_nom"] = config.nominal_coverage
    results["opt_lambda"]["APS"] = opt_lambda_APS
    results["opt_lambda"]["APS_cons"] = max(opt_lambda_APS, config.nominal_coverage)
    results["opt_lambda"]["APS_nom"] = config.nominal_coverage
    
    # Compute optimal b for each approach
    for approach in config.approaches:
        if approach.startswith("BPS"):
            results["opt_b"][approach] = solve_b_in_batches(
                test_probs, results["opt_lambda"][approach], batch_size=config.batch_size
            )
        else:  # APS
            results["opt_b"][approach] = solve_b_in_batches(
                test_mean_probs, results["opt_lambda"][approach], batch_size=config.batch_size
            )
    
    return results


def run_all_seeds(
    config: PipelineConfig,
    predictions: Dict[str, torch.Tensor],
    save_intermediate: bool = True
) -> Dict[int, Dict[str, Any]]:
    """Run prediction set generation for all seeds."""
    all_results = {}
    
    for seed in range(config.n_seeds):
        print(f"  Seed {seed + 1}/{config.n_seeds}...", end=" ", flush=True)
        
        results = create_prediction_sets(config, predictions, seed)
        all_results[seed] = results
        
        if save_intermediate:
            save_path = config.results_path(seed)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump(results, f)
        
        print("done")
    
    return all_results


# =============================================================================
# Evaluation Metrics
# =============================================================================

def evaluate_results(
    config: PipelineConfig,
    all_results: Dict[int, Dict[str, Any]],
    soft_labels: Optional[np.ndarray] = None,
    predictions: Optional[Dict[str, torch.Tensor]] = None
) -> Dict[str, Dict[str, List[float]]]:
    """
    Evaluate prediction sets across all seeds.
    
    Returns dict of {metric_name: {approach: [values_per_seed]}}
    """
    metric_names = ["marginal_cvg", "set_size", "coverage_gap", "SSCV"]
    if soft_labels is not None:
        metric_names.append("conditional_cvg")
    
    # init results dict
    results = {metric: {approach: [] for approach in config.approaches} for metric in metric_names}
    
    num_classes = predictions["mean_probs"].shape[-1] if predictions else 10
    
    for seed in range(config.n_seeds):
        res = all_results[seed]
        test_labels = res["test_labels"]
        
        # get soft labels for this split if available
        test_annotations = None
        if soft_labels is not None:
            labels_np = predictions["labels"].numpy()
            probs_np = predictions["probs"].numpy()
            mean_probs_np = predictions["mean_probs"].numpy()
            _, _, _, _, _, _, _, test_annotations = train_test_split(
                probs_np, mean_probs_np, labels_np, soft_labels,
                train_size=config.calib_size, random_state=seed
            )
        
        for approach in config.approaches:
            opt_b = res["opt_b"][approach]
            
            # Marginal coverage: average b[i, y_i]
            results["marginal_cvg"][approach].append(
                avg_true_label_inclusion(opt_b, test_labels)
            )
            
            # Set size: average sum of b
            results["set_size"][approach].append(
                np.mean(np.sum(opt_b, axis=1))
            )
            
            # Coverage gap
            results["coverage_gap"][approach].append(
                covGap_nonbinary(opt_b, test_labels, config.alpha, num_classes=num_classes)
            )
            
            # Size-stratified coverage violation
            results["SSCV"][approach].append(
                size_stratified_cov_violation_nonbinary(
                    opt_b, test_labels, config.alpha,
                    stratified_size=[[i, i+1] for i in range(num_classes)]
                )
            )
            
            # Conditional coverage (if soft labels available)
            if test_annotations is not None:
                results["conditional_cvg"][approach].append(
                    np.mean(np.sum(np.multiply(opt_b, test_annotations), axis=1))
                )
    
    return results


# =============================================================================
# Results Display
# =============================================================================

def print_results(
    results: Dict[str, Dict[str, List[float]]],
    config: PipelineConfig
): 
    print("\n" + "=" * 80)
    print(f"RESULTS: {config.dataset.upper()} | Model: {config.model} | α={config.alpha}")
    print(f"Target coverage: {config.nominal_coverage:.0%} | Calib size: {config.calib_size:.0%} | Seeds: {config.n_seeds}")
    print("=" * 80)
    
    # BPS vs APS main comparison
    print("\n Main Results (mean +- std): \n")
    
    # Header
    print(f"{'Metric':<18} {'BPS':>14} {'APS':>14} {'BPS_cons':>14} {'APS_cons':>14}")
    print("-" * 78)
    
    # Metrics
    for metric in results:
        row = f"{metric:<18}"
        for approach in ["BPS", "APS", "BPS_cons", "APS_cons"]:
            if approach in results[metric]:
                vals = results[metric][approach]
                mean = np.mean(vals)
                std = np.std(vals)
                row += f"  {mean:>6.4f}±{std:.3f}"
        print(row)
    
    print("-" * 78)
    
    # Key insights
    print("\nInsights:\n")
    
    bps_cvg = np.mean(results["marginal_cvg"]["BPS"])
    aps_cvg = np.mean(results["marginal_cvg"]["APS"])
    bps_size = np.mean(results["set_size"]["BPS"])
    aps_size = np.mean(results["set_size"]["APS"])
    
    target = config.nominal_coverage
    
    print(f"  Target coverage: {target:.0%}")
    print(f"  BPS: {bps_cvg:.2%} coverage, {bps_size:.2f} avg set size")
    print(f"  APS: {aps_cvg:.2%} coverage, {aps_size:.2f} avg set size")
    
    if bps_size > aps_size:
        overhead = (bps_size / aps_size - 1) * 100
        print(f"  → BPS uses {overhead:.1f}% larger sets (accounts for epistemic uncertainty)")
    
    bps_gap = np.mean(results["coverage_gap"]["BPS"])
    aps_gap = np.mean(results["coverage_gap"]["APS"])
    
    if bps_gap < aps_gap:
        print(f"  → BPS has lower coverage gap ({bps_gap:.4f} vs {aps_gap:.4f})")
    
    if "conditional_cvg" in results:
        bps_cond = np.mean(results["conditional_cvg"]["BPS"])
        aps_cond = np.mean(results["conditional_cvg"]["APS"])
        print(f"  Conditional coverage (soft labels): BPS={bps_cond:.2%}, APS={aps_cond:.2%}")
    
    print()


def save_results(
    results: Dict[str, Dict[str, List[float]]],
    config: PipelineConfig
):
    save_path = config.results_path()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {save_path}")


# =============================================================================
# Main Pipeline
# =============================================================================

def run_pipeline(config: PipelineConfig) -> Dict[str, Dict[str, List[float]]]:
    """Run the complete pipeline."""
    
    print("\n" + "=" * 80)
    print("CONFORMAL PREDICTION PIPELINE (BPS)")
    print("=" * 80)
    print(f"  Dataset:      {config.dataset}")
    print(f"  Model:        {config.model}")
    print(f"  Alpha:        {config.alpha} (target coverage: {config.nominal_coverage:.0%})")
    print(f"  Calib size:   {config.calib_size:.0%}")
    print(f"  Seeds:        {config.n_seeds}")
    print(f"  Device:       {config.device}")
    print("=" * 80)
    
    print("\n[1/4] Loading predictions")
    predictions = load_predictions(config)
    
    print("\n[2/4] Loading soft labels")
    soft_labels = load_soft_labels(config)
    
    print("\n[3/4] Creating prediction sets")
    all_results = run_all_seeds(config, predictions, save_intermediate=True)
    
    print("\n[4/4] Evaluating results")
    eval_results = evaluate_results(config, all_results, soft_labels, predictions)
    
    print_results(eval_results, config)
    save_results(eval_results, config)
    
    return eval_results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Conformal Prediction Pipeline with Bernoulli Prediction Sets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset", type=str, default="cifar10", 
                        help="Dataset name (cifar10, cifar100, imagenet)")
    parser.add_argument("--model", type=str, default="ensemble",
                        help="Model type (ensemble, mc, evidential)")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Significance level (0.1 = 90%% coverage)")
    parser.add_argument("--calib_size", type=float, default=0.5,
                        help="Calibration set size as fraction")
    parser.add_argument("--n_seeds", type=int, default=10,
                        help="Number of random seeds")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Batch size for optimization")
    
    args = parser.parse_args()
    
    config = PipelineConfig(
        dataset=args.dataset,
        model=args.model,
        alpha=args.alpha,
        calib_size=args.calib_size,
        n_seeds=args.n_seeds,
        batch_size=args.batch_size,
    )
    
    run_pipeline(config)


if __name__ == "__main__":
    main()
