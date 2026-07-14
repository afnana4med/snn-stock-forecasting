"""Multi-horizon evaluation: does trend signal at longer forecast horizons
let the SNN (or any model) beat the persistence baseline?

Holds everything constant (AAPL daily, 20-day window, engineered features,
delta target, e-prop) and varies only the prediction horizon over 1, 5, 10,
20 trading days. Each horizon is scored with the full rigorous protocol:
walk-forward CV, 3-seed ensembling, Diebold-Mariano tests, and the
persistence / linear-trend / ridge / LSTM baselines.

Usage:
    python scripts/run_multi_horizon.py            # horizons 1 5 10 20
    python scripts/run_multi_horizon.py 10 20      # specific horizons
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

OUT_DIR = PROJECT_ROOT / "experiments" / "multi_horizon"
HORIZONS = [1, 5, 10, 20]

BASE_CONFIG = {
    "experiment_name": "multi_horizon",
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
        "engineered_features": True,
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


def main():
    horizons = [int(a) for a in sys.argv[1:]] or HORIZONS
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = {}
    for h in horizons:
        config = copy.deepcopy(BASE_CONFIG)
        config["data"]["prediction_horizon"] = h
        logging.info(f"════ Horizon: {h} trading day(s) ════")
        results = run_final_evaluation(config, seeds=(42, 123, 777),
                                       n_folds=4, max_samples=2500)
        results.pop("_series", None)
        summary[f"h{h}"] = results

        with open(OUT_DIR / f"h{h}_results.json", "w") as f:
            json.dump(results, f, indent=2)

        m = results["models"]
        # Which baseline is hardest to beat at this horizon?
        baselines = {k: m[k]["rmse_dollars"]
                     for k in ("persistence", "trend", "ridge", "lstm")}
        best_base = min(baselines, key=baselines.get)
        snn_rmse = m["snn"]["rmse_dollars"]
        verdict = ("BEATS" if snn_rmse < baselines[best_base] else "loses to")
        logging.info(
            f"→ h={h}: SNN ${snn_rmse:.3f} | persistence "
            f"${m['persistence']['rmse_dollars']:.3f} | trend "
            f"${m['trend']['rmse_dollars']:.3f} | ridge "
            f"${m['ridge']['rmse_dollars']:.3f} | LSTM "
            f"${m['lstm']['rmse_dollars']:.3f}  →  SNN {verdict} best "
            f"baseline ({best_base})")
        for test, r in results["dm_tests"].items():
            logging.info(f"   {test}: DM={r['dm_statistic']:+.2f} "
                         f"p={r['p_value']:.3f} ({r['interpretation']})")

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Multi-horizon evaluation complete → "
                 f"{OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
