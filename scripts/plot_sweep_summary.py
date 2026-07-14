"""Render a comparison figure from experiments/sweep/summary.csv."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SWEEP_DIR = PROJECT_ROOT / "experiments" / "sweep"

BLUE = "#2a78d6"
INK = "#0b0b0b"
MUTED = "#52514e"


def main():
    with open(SWEEP_DIR / "summary.csv") as f:
        rows = list(csv.DictReader(f))

    reg = [r for r in rows if r["val_rmse_dollars"]]
    cls = [r for r in rows if r["val_accuracy"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 0.55 * max(len(reg), 6) + 2))

    # --- Panel 1: regression RMSE ($) vs persistence baseline ---
    names = [r["name"].replace("reg_", "") for r in reg]
    rmse = [float(r["val_rmse_dollars"]) for r in reg]
    base = float(reg[0]["persistence_baseline_rmse_dollars"])
    y = range(len(reg))
    ax1.barh(y, rmse, height=0.55, color=BLUE)
    ax1.axvline(base, color=MUTED, linestyle="--", linewidth=2,
                label=f"persistence baseline (${base:.2f})")
    for i, v in enumerate(rmse):
        ax1.text(v + 0.004, i, f"${v:.3f}", va="center", fontsize=9, color=INK)
    ax1.set_yticks(list(y), names)
    ax1.invert_yaxis()
    ax1.set_xlabel("Validation RMSE ($) — lower is better")
    ax1.set_title("Next-day close regression (AAPL daily)")
    ax1.legend(loc="lower right")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(axis="x", alpha=0.25)

    # --- Panel 2: direction accuracy (regression sign + classifiers) ---
    dir_rows = ([(r["name"].replace("reg_", ""),
                  float(r["directional_accuracy"]))
                 for r in reg if r["directional_accuracy"]] +
                [(r["name"].replace("cls_", "") + " (clf)",
                  float(r["val_accuracy"])) for r in cls])
    names2 = [n for n, _ in dir_rows]
    accs = [a for _, a in dir_rows]
    y2 = range(len(dir_rows))
    ax2.barh(y2, accs, height=0.55, color=BLUE)
    ax2.axvline(0.5, color=MUTED, linestyle="--", linewidth=2,
                label="coin flip (0.50)")
    for i, v in enumerate(accs):
        ax2.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=9, color=INK)
    ax2.set_yticks(list(y2), names2)
    ax2.invert_yaxis()
    ax2.set_xlim(0.0, max(accs) + 0.12)
    ax2.set_xlabel("Next-day direction accuracy — higher is better")
    ax2.set_title("Direction of next-day move (AAPL daily)")
    ax2.legend(loc="lower right")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="x", alpha=0.25)

    fig.suptitle("SNN hyperparameter sweep — validation results", y=1.0)
    fig.tight_layout()
    out = SWEEP_DIR / "sweep_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight",
                facecolor="#fcfcfb")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
