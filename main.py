"""
Main entry point: train PINN on Darcy flow and generate evaluation figures.

Usage:
    python main.py                                  # Run homogeneous case (default)
    python main.py --case layered                   # Run layered permeability case
    python main.py --case lognormal                 # Run log-normal random field case
    python main.py --case homogeneous --epochs 20000
    python main.py --config configs/default.yaml
    python main.py --all                            # Run all three test cases
"""

import argparse
import os
import yaml

import torch

from src.model import DarcyPINN, HardBCDarcyPINN
from src.physics import get_test_case
from src.data import sample_sparse_observations
from src.train import train, TrainingConfig
from src.evaluate import evaluate_model, generate_all_figures
from src.utils import set_seed, get_device, save_model


def run_case(case_name: str, config: dict):
    """Run training and evaluation for a single test case."""
    print(f"\n{'='*60}")
    print(f"  CASE: {case_name.upper()}")
    print(f"{'='*60}\n")

    # Setup
    seed = config.get("seed", 42)
    set_seed(seed)
    device = get_device()

    # Get test case (permeability, source, optional exact solution)
    test_case = get_test_case(case_name)

    # Build model
    model_cfg = config.get("model", {})
    if model_cfg.get("hard_bc", False):
        model = HardBCDarcyPINN(
            n_frequencies=model_cfg.get("n_frequencies", 64),
            sigma=model_cfg.get("sigma", 2.0),
            hidden_width=model_cfg.get("hidden_width", 128),
            n_residual_blocks=model_cfg.get("n_residual_blocks", 4),
            use_fourier_features=model_cfg.get("use_fourier_features", True),
        )
    else:
        model = DarcyPINN(
            n_frequencies=model_cfg.get("n_frequencies", 64),
            sigma=model_cfg.get("sigma", 2.0),
            hidden_width=model_cfg.get("hidden_width", 128),
            n_residual_blocks=model_cfg.get("n_residual_blocks", 4),
            use_fourier_features=model_cfg.get("use_fourier_features", True),
        )

    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {model.count_parameters():,}")

    # Training config
    train_cfg = config.get("training", {})
    training_config = TrainingConfig(
        learning_rate=train_cfg.get("learning_rate", 1e-3),
        min_learning_rate=train_cfg.get("min_learning_rate", 1e-6),
        epochs=train_cfg.get("epochs", 15000),
        optimizer=train_cfg.get("optimizer", "adam"),
        n_interior=train_cfg.get("n_interior", 4000),
        n_boundary_per_edge=train_cfg.get("n_boundary_per_edge", 200),
        resample_every=train_cfg.get("resample_every", 500),
        lambda_pde=train_cfg.get("lambda_pde", 1.0),
        lambda_bc=train_cfg.get("lambda_bc", 10.0),
        lambda_data=train_cfg.get("lambda_data", 1.0),
        adaptive_weights=train_cfg.get("adaptive_weights", True),
        adaptive_alpha=train_cfg.get("adaptive_alpha", 0.9),
        n_data_points=train_cfg.get("n_data_points", 0),
        data_noise_std=train_cfg.get("data_noise_std", 0.0),
        log_every=train_cfg.get("log_every", 500),
        save_every=train_cfg.get("save_every", 5000),
        device=str(device),
    )

    # Optional sparse observations
    x_data, y_data, p_data = None, None, None
    if training_config.n_data_points > 0 and test_case.has_exact_solution():
        x_data, y_data, p_data = sample_sparse_observations(
            test_case.exact_solution,
            n_obs=training_config.n_data_points,
            noise_std=training_config.data_noise_std,
            device=device,
        )

    # Exact solution for monitoring (if available)
    exact_fn = test_case.exact_solution if test_case.has_exact_solution() else None

    # Train
    history = train(
        model=model,
        permeability_fn=test_case.permeability,
        source_fn=test_case.source,
        config=training_config,
        exact_solution_fn=exact_fn,
        x_data=x_data,
        y_data=y_data,
        p_data=p_data,
    )

    # Evaluate
    eval_cfg = config.get("evaluation", {})
    results = evaluate_model(
        model=model,
        permeability_fn=test_case.permeability,
        source_fn=test_case.source,
        n_grid=eval_cfg.get("n_grid", 100),
        device=device,
        exact_solution_fn=exact_fn,
    )

    # Generate figures
    output_dir = eval_cfg.get("output_dir", "figures")
    generate_all_figures(
        model=model,
        results=results,
        history=history,
        permeability_fn=test_case.permeability,
        source_fn=test_case.source,
        device=device,
        case_name=case_name,
        output_dir=output_dir,
    )

    # Save model checkpoint
    save_model(
        model, f"checkpoints/{case_name}_model.pt",
        metadata={
            "case": case_name,
            "epochs": training_config.epochs,
            "relative_l2": results.get("relative_l2", None),
        },
    )

    return results


def main():
    parser = argparse.ArgumentParser(description="PINN for Darcy Flow")
    parser.add_argument(
        "--case", type=str, default="homogeneous",
        choices=["homogeneous", "layered", "lognormal"],
        help="Test case to run",
    )
    parser.add_argument("--all", action="store_true", help="Run all test cases")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    args = parser.parse_args()

    # Load config
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = {}  # use defaults in TrainingConfig

    # CLI overrides
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.seed is not None:
        config["seed"] = args.seed

    # Run
    if args.all:
        for case in ["homogeneous", "layered", "lognormal"]:
            run_case(case, config)
    else:
        run_case(args.case, config)


if __name__ == "__main__":
    main()
