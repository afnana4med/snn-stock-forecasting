"""Render figures from the final evaluation summary."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "experiments" / "final_eval"

# Fixed categorical hue order (one color per model, never per rank)
MODEL_COLORS = {"snn": "#2a78d6", "persistence": "#1baf7a",
                "ridge": "#eda100", "lstm": "#008300"}
MODEL_LABELS = {"snn": "SNN (e-prop)", "persistence": "Persistence",
                "ridge": "Ridge", "lstm": "LSTM"}
INK, MUTED = "#0b0b0b", "#52514e"


def main():
    with open(OUT_DIR / "summary.json") as f:
        summary = json.load(f)

    variants = list(summary)
    models = ["snn", "persistence", "ridge", "lstm"]

    # --- Fig 1: RMSE ($) per variant, grouped by model ---
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(variants))
    width = 0.2
    for i, m in enumerate(models):
        vals = [summary[v]["models"][m]["rmse_dollars"] for v in variants]
        bars = ax.bar(x + (i - 1.5) * width, vals, width,
                      color=MODEL_COLORS[m], label=MODEL_LABELS[m])
        for b, val in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, val, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, variants)
    ax.set_ylabel("Walk-forward test RMSE ($) — lower is better")
    ax.set_title("SNN vs baselines: AAPL next-day close "
                 "(4-fold walk-forward, 3-seed ensemble)")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "final_rmse_comparison.png", dpi=150,
                facecolor="#fcfcfb")
    plt.close(fig)

    # --- Fig 2: computational cost per prediction (log scale) ---
    eff = None
    for v in variants:
        if "efficiency" in summary[v]:
            eff = summary[v]["efficiency"]
            eff_variant = v
            break
    if eff:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        names = ["SNN\n(synaptic events + updates)", "LSTM\n(MACs)",
                 "Ridge\n(MACs)"]
        vals = [eff["snn_total_ops_per_prediction"],
                eff["lstm_macs_per_prediction"],
                eff["ridge_macs_per_prediction"]]
        colors = [MODEL_COLORS["snn"], MODEL_COLORS["lstm"],
                  MODEL_COLORS["ridge"]]
        bars = ax.bar(names, vals, color=colors, width=0.55)
        for b, val in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, val * 1.1, f"{val:,.0f}",
                    ha="center", fontsize=10, color=INK)
        ax.set_yscale("log")
        ax.set_ylabel("Operations per prediction (log scale)")
        ax.set_title(f"Computational cost per prediction "
                     f"({eff_variant} variant)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25, which="both")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "final_efficiency_comparison.png", dpi=150,
                    facecolor="#fcfcfb")
        plt.close(fig)

    # --- Fig 3: pooled walk-forward predictions for the best SNN variant ---
    best = min(variants,
               key=lambda v: summary[v]["models"]["snn"]["rmse_dollars"])
    with open(OUT_DIR / f"{best}_series.json") as f:
        series = json.load(f)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(series["true_price"], color=MUTED, linewidth=1.2, label="True")
    ax.plot(series["pred_price_snn"], color=MODEL_COLORS["snn"],
            linewidth=1.2, label=f"SNN prediction ({best})")
    ax.set_xlabel("Walk-forward test sample (chronological)")
    ax.set_ylabel("Close price ($)")
    ax.set_title("Out-of-sample walk-forward predictions — best SNN variant")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "final_predictions.png", dpi=150,
                facecolor="#fcfcfb")
    plt.close(fig)

    print(f"Figures written to {OUT_DIR}")


if __name__ == "__main__":
    main()
