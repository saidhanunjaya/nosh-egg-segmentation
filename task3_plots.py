"""
Task 3 - Step 2: plots summarising Task 2 vs Task 1 segmentation performance.

Reads:  task3_outputs/failure_modes.csv   (written by task3_failure_modes.py)
Writes: task3_outputs/plots/iou_histogram.png
        task3_outputs/plots/precision_recall_scatter.png
        task3_outputs/plots/category_counts.png
        task3_outputs/plots/area_ratio_vs_iou.png
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save to files only, no pop-up windows
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "task3_outputs"
FAILURE_CSV = OUT_DIR / "failure_modes.csv"
PLOT_DIR = OUT_DIR / "plots"

CATEGORY_ORDER = [
    "pass",
    "total_miss",
    "disjoint_wrong_region",
    "mislocalized",
    "bleed_overseg",
    "partial_underseg",
    "mixed_moderate",
]

CATEGORY_COLORS = {
    "pass": "#2ca02c",                 # green
    "total_miss": "#000000",           # black
    "disjoint_wrong_region": "#d62728",  # red
    "mislocalized": "#ff7f0e",         # orange
    "bleed_overseg": "#9467bd",        # purple
    "partial_underseg": "#1f77b4",     # blue
    "mixed_moderate": "#8c564b",       # brown
}


def load_rows():
    rows = []
    with open(FAILURE_CSV, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "stem": r["stem"],
                "category": r["category"],
                "iou": float(r["iou"]),
                "precision": float(r["precision"]),
                "recall": float(r["recall"]),
                "pred_gt_ratio": float(r["pred_gt_ratio"]),
            })
    return rows


def plot_iou_histogram(rows):
    ious = np.array([r["iou"] for r in rows])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ious, bins=np.linspace(0, 1, 21), color="steelblue", edgecolor="black")
    ax.axvline(0.5, color="red", ls="--", label="IoU = 0.5 (pass threshold)")
    ax.axvline(0.75, color="green", ls="--", label="IoU = 0.75 (good)")
    ax.axvline(np.median(ious), color="black", ls=":", label=f"median = {np.median(ious):.3f}")
    ax.set_xlabel("IoU (Task 2 prediction vs Task 1 ground truth)")
    ax.set_ylabel("Number of images")
    ax.set_title(f"IoU distribution across {len(ious)} images")
    ax.set_xlim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "iou_histogram.png", dpi=150)
    plt.close(fig)


def plot_precision_recall(rows):
    fig, ax = plt.subplots(figsize=(9, 7))
    rng = np.random.default_rng(0)
    for cat in CATEGORY_ORDER:
        members = [r for r in rows if r["category"] == cat]
        if not members:
            continue
        # small jitter so identical (0,0) points don't stack invisibly
        rec = np.array([m["recall"] for m in members]) + rng.uniform(-0.01, 0.01, len(members))
        pre = np.array([m["precision"] for m in members]) + rng.uniform(-0.01, 0.01, len(members))
        ax.scatter(rec, pre, s=40, alpha=0.7, color=CATEGORY_COLORS[cat],
                   edgecolor="black", linewidth=0.4, label=f"{cat} ({len(members)})")
    ax.axvline(0.5, color="grey", ls=":", lw=1)
    ax.axhline(0.5, color="grey", ls=":", lw=1)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Recall (fraction of GT egg pixels recovered)")
    ax.set_ylabel("Precision (fraction of predicted pixels that are egg)")
    ax.set_title("Precision vs recall by failure category (jittered \u00b10.01)")
    # Legend placed OUTSIDE the axes so it can never overlap data points,
    # regardless of where a category's cluster happens to fall (e.g. the
    # disjoint_wrong_region cluster sits exactly at the bottom-left corner,
    # which a default loc="lower left" legend would otherwise hide behind).
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    fig.subplots_adjust(right=0.75)
    fig.savefig(PLOT_DIR / "precision_recall_scatter.png", dpi=150)
    plt.close(fig)


def plot_category_counts(rows):
    counts = [sum(1 for r in rows if r["category"] == c) for c in CATEGORY_ORDER]
    n = len(rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(CATEGORY_ORDER, counts,
                  color=[CATEGORY_COLORS[c] for c in CATEGORY_ORDER], edgecolor="black")
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{cnt}\n({100 * cnt / n:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Number of images")
    ax.set_title(f"Failure-mode categories (n = {n})")
    ax.set_ylim(0, max(counts) * 1.2)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "category_counts.png", dpi=150)
    plt.close(fig)


def plot_area_ratio_vs_iou(rows):
    fig, ax = plt.subplots(figsize=(9, 6))
    for cat in CATEGORY_ORDER:
        members = [r for r in rows if r["category"] == cat]
        if not members:
            continue
        # total_miss has ratio 0 -> clip to 1e-3 so it can be drawn on a log axis
        log_ratio = np.log10([max(m["pred_gt_ratio"], 1e-3) for m in members])
        ious = [m["iou"] for m in members]
        ax.scatter(log_ratio, ious, s=40, alpha=0.7, color=CATEGORY_COLORS[cat],
                   edgecolor="black", linewidth=0.4, label=f"{cat} ({len(members)})")
    ax.axvline(0, color="grey", ls=":", lw=1, label="predicted area = GT area")
    ax.axhline(0.5, color="red", ls="--", lw=1, label="IoU = 0.5")
    ax.set_xlabel("log10(predicted area / GT area)   (>0 = over-segmented)")
    ax.set_ylabel("IoU")
    ax.set_title("Area ratio vs IoU: over-segmentation (bleed) drives IoU collapse")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    fig.subplots_adjust(right=0.75)
    fig.savefig(PLOT_DIR / "area_ratio_vs_iou.png", dpi=150)
    plt.close(fig)


def main():
    if not FAILURE_CSV.exists():
        raise SystemExit(f"Missing {FAILURE_CSV} -- run task3_failure_modes.py first.")
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_rows()
    plot_iou_histogram(rows)
    plot_precision_recall(rows)
    plot_category_counts(rows)
    plot_area_ratio_vs_iou(rows)

    print(f"Wrote 4 plots to: {PLOT_DIR}")
    for name in ["iou_histogram.png", "precision_recall_scatter.png",
                 "category_counts.png", "area_ratio_vs_iou.png"]:
        print(f"  - {name}")


if __name__ == "__main__":
    main()