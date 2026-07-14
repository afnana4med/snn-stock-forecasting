"""Hyperparameter sweep runner for the SNN stock-prediction experiments.

Takes a base config, applies named override sets, runs each variant as a
subprocess (so GeNN code generation stays isolated), and aggregates every
variant's results.json into experiments/sweep/summary.csv.

Usage:
    python scripts/run_experiments.py
"""

import copy
import csv
import json
import logging
import subprocess
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SWEEP_DIR = PROJECT_ROOT / "experiments" / "sweep"
PYTHON = sys.executable


def deep_update(d, overrides):
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d


def run_variant(base_config_path, name, overrides):
    with open(base_config_path) as f:
        config = yaml.safe_load(f)

    config = deep_update(copy.deepcopy(config), overrides)
    config["experiment_name"] = name
    config["logging"]["save_dir"] = str(SWEEP_DIR)

    cfg_dir = SWEEP_DIR / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / f"{name}.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(config, f)

    logging.info(f"▶ Running {name} ...")
    proc = subprocess.run(
        [PYTHON, "-m", "snn_stock.main", "--config", str(cfg_path)],
        cwd=PROJECT_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        logging.error(f"✗ {name} FAILED:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
        return None

    results_file = SWEEP_DIR / name / "results.json"
    with open(results_file) as f:
        results = json.load(f)
    results["name"] = name
    logging.info(f"✓ {name}: " + ", ".join(
        f"{k}={v:.4f}" for k, v in results.items()
        if isinstance(v, float) and not k.endswith("history")))
    return results


# Sweep definitions -----------------------------------------------------------

REGRESSION_BASE = PROJECT_ROOT / "configs" / "rate_eprop.yaml"
CLASSIFICATION_BASE = PROJECT_ROOT / "configs" / "temporal_eventprop.yaml"

LONG = {"training": {"epochs": 40, "lr_decay": 0.7, "lr_decay_every": 10}}

REGRESSION_VARIANTS = {
    # What we ran before (reference point)
    "reg_baseline_e15": {},
    # More epochs with learning-rate decay
    "reg_long_e40": copy.deepcopy(LONG),
    # Capacity
    "reg_long_h64": deep_update(copy.deepcopy(LONG),
                                {"model": {"hidden_layers": [{"units": 64}]}}),
    # Finer rate-code resolution
    "reg_long_T80": deep_update(copy.deepcopy(LONG), {"n_steps": 80}),
    # Predict the move instead of the level (0 == persistence)
    "reg_long_delta": deep_update(copy.deepcopy(LONG),
                                  {"data": {"target_mode": "delta"}}),
    # Everything combined
    "reg_long_delta_h64_T80": deep_update(copy.deepcopy(LONG), {
        "data": {"target_mode": "delta"},
        "model": {"hidden_layers": [{"units": 64}]},
        "n_steps": 80}),
    # Temporal encoding with the combined recipe
    "reg_long_delta_temporal": deep_update(copy.deepcopy(LONG), {
        "data": {"target_mode": "delta", "encoding": "temporal"},
        "model": {"hidden_layers": [{"units": 64}]},
        "n_steps": 50}),
}

CLASSIFICATION_VARIANTS = {
    "cls_baseline_e15": {},
    "cls_long_e30": {"training": {"epochs": 30, "lr_decay": 0.7,
                                  "lr_decay_every": 10}},
    "cls_long_h64_lr002": {
        "training": {"epochs": 30, "learning_rate": 0.002,
                     "lr_decay": 0.7, "lr_decay_every": 10},
        "model": {"hidden_layers": [{"units": 64}]}},
}


if __name__ == "__main__":
    all_results = []
    for name, overrides in REGRESSION_VARIANTS.items():
        r = run_variant(REGRESSION_BASE, name, overrides)
        if r:
            all_results.append(r)
    for name, overrides in CLASSIFICATION_VARIANTS.items():
        r = run_variant(CLASSIFICATION_BASE, name, overrides)
        if r:
            all_results.append(r)

    # Aggregate into a CSV
    fields = ["name", "metric", "val_rmse_dollars",
              "persistence_baseline_rmse_dollars", "directional_accuracy",
              "majority_direction_baseline", "val_accuracy",
              "majority_baseline_accuracy", "val_mse", "val_mae"]
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    with open(SWEEP_DIR / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)

    logging.info(f"Sweep complete: {len(all_results)} variants. "
                 f"Summary at {SWEEP_DIR / 'summary.csv'}")
