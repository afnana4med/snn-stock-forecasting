"""Render figures from the multi-horizon evaluation summary."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "experiments" / "multi_horizon"

# Fixed categorical hue order (one color per model, never per rank)
MODEL_COLORS = {"snn": "#2a78d6", "persistence": "#1baf7a",
                "trend": "#4a3aa7", "ridge": "#eda100", "lstm": "#008300"}
MODEL_LABELS = {"snn": "SNN (e-prop)", "persistence": "Persistence",
                "trend": "Linear trend", "ridge": "Ridge", "lstm": "LSTM"}
INK, MUTED = "#0b0b0b", "#52514e"


def main():
    with open(OUT_DIR / "summary.json") as f:
        summary = json.load(f)

    horizons = sorted(summary, key=lambda k: int(k[1:]))
    h_vals = [int(k[1:]) for k in horizons]
    models = ["snn", "persistence", "trend", "ridge", "lstm"]

    # --- Fig 1: RMSE vs horizon, normalized to persistence (ratio) ---
    # Ratio < 1 means the model beats persistence at that horizon.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for m in models:
        if m == "persistence":
            continue
        ratios = [summary[h]["models"][m]["rmse_dollars"] /
                  summary[h]["models"]["persistence"]["rmse_dollars"]
                  for h in horizons]
        ax1.plot(h_vals, ratios, marker="o", color=MODEL_COLORS[m],
                 label=MODEL_LABELS[m], linewidth=2)
    ax1.axhline(1.0, color=MODEL_COLORS["persistence"], linestyle="--",
                linewidth=2, label="Persistence (=1.0)")
    ax1.set_xlabel("Forecast horizon (trading days)")
    ax1.set_ylabel("RMSE relative to persistence — below 1.0 beats it")
    ax1.set_title("Does trend signal emerge at longer horizons?")
    ax1.set_xticks(h_vals)
    ax1.legend()
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(alpha=0.25)

    # --- Fig 2: directional accuracy vs horizon ---
    for m in models:
        accs = [summary[h]["models"][m]["directional_accuracy"]
                for h in horizons]
        style = "--" if m == "persistence" else "-"
        ax2.plot(h_vals, accs, marker="s", color=MODEL_COLORS[m],
                 label=MODEL_LABELS[m], linewidth=2, linestyle=style)
    ax2.axhline(0.5, color=MUTED, linestyle=":", linewidth=1.5,
                label="coin flip (0.50)")
    ax2.set_xlabel("Forecast horizon (trading days)")
    ax2.set_ylabel("Direction accuracy — higher is better")
    ax2.set_title("Direction-of-move accuracy vs horizon")
    ax2.set_xticks(h_vals)
    ax2.legend()
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(alpha=0.25)

    fig.suptitle("Multi-horizon forecasting — AAPL daily, walk-forward "
                 "(4-fold, 3-seed)", y=1.0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "multi_horizon_summary.png", dpi=150,
                facecolor="#fcfcfb")
    plt.close(fig)
    print(f"Saved {OUT_DIR / 'multi_horizon_summary.png'}")


if __name__ == "__main__":
    main()
