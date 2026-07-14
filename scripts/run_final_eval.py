"""Final rigorous evaluation: walk-forward multi-seed comparison of SNN
variants against persistence / ridge / LSTM baselines with Diebold-Mariano
significance tests and computational-cost accounting.

Usage:
    python scripts/run_final_eval.py                # all variants
    python scripts/run_final_eval.py base features  # subset
"""

import copy
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from snn_stock.training.evaluation import run_final_evaluation  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

OUT_DIR = PROJECT_ROOT / "experiments" / "final_eval"

BASE_CONFIG = {
    "experiment_name": "final_eval",
    "task": "regression",
    "algorithm": "eprop",
    "n_steps": 50,
    "dt": 1.0,
    "data": {
        "files": [str(PROJECT_ROOT / "data/processed/daily_AAPL.csv")],
        "features": ["Open", "High", "Low", "Close", "Volume"],
        "target_column": "Close",
        "sequence_length": 20,
        "prediction_horizon": 1,
        "normalize": "window",
        "target_mode": "delta",
        "encoding": "temporal",
    },
    "model": {
        "neuron_params": {"threshold": 0.4, "tau_mem": 20.0, "tau_syn": 5.0,
                          "refractory_period": 1.0},
        "hidden_layers": [{"units": 32}],
        "output_neurons": 1,
        "output_readout": "var",
    },
    "training": {"epochs": 20, "batch_size": 1, "learning_rate": 0.005,
                 "optimiser": "adam", "lr_decay": 0.7, "lr_decay_every": 10},
    "logging": {"save_dir": str(OUT_DIR)},
}

SPY_FILE = str(PROJECT_ROOT / "data/raw/sp500_etf_daily.csv")

VARIANTS = {
    # Winning recipe from the hyperparameter sweep (reference point)
    "base": {},
    # + returns / volatility / volume-ratio / high-low-range features
    "features": {"data": {"engineered_features": True}},
    # + S&P 500 context asset and a recurrent hidden layer (RSNN)
    "context_recurrent": {"data": {"engineered_features": True,
                                   "context_files": [SPY_FILE]},
                          "model": {"recurrent": True}},
    # Temporal-contrast (threshold) coding on the engineered features
    "contrast": {"data": {"engineered_features": True,
                          "encoding": "contrast"}},
}


def deep_update(d, overrides):
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d


def main():
    requested = sys.argv[1:] or list(VARIANTS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = {}
    for name in requested:
        overrides = VARIANTS[name]
        config = deep_update(copy.deepcopy(BASE_CONFIG), overrides)
        logging.info(f"════ Variant: {name} ════")
        results = run_final_evaluation(config, seeds=(42, 123, 777),
                                       n_folds=4, max_samples=2500)
        series = results.pop("_series")
        summary[name] = results

        with open(OUT_DIR / f"{name}_results.json", "w") as f:
            json.dump(results, f, indent=2)
        with open(OUT_DIR / f"{name}_series.json", "w") as f:
            json.dump(series, f)

        m = results["models"]
        logging.info(f"→ {name}: SNN ${m['snn']['rmse_dollars']:.4f} | "
                     f"persistence ${m['persistence']['rmse_dollars']:.4f} | "
                     f"ridge ${m['ridge']['rmse_dollars']:.4f} | "
                     f"LSTM ${m['lstm']['rmse_dollars']:.4f}")
        for test, r in results["dm_tests"].items():
            logging.info(f"   {test}: DM={r['dm_statistic']:+.3f} "
                         f"p={r['p_value']:.3f} ({r['interpretation']})")

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"All variants complete → {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
