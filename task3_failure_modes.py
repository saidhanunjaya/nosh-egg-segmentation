"""
Task 3 - Step 1: classify every Task 2 prediction into a failure-mode category,
using the per-image metrics written by task3_evaluate.py.

Reads:  task3_outputs/metrics.csv
Writes: task3_outputs/failure_modes.csv           (one row per image + category)
        task3_outputs/failure_modes_summary.txt   (counts, stats, full member lists)
"""

import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "task3_outputs"
METRICS_CSV = OUT_DIR / "metrics.csv"
FAILURE_CSV = OUT_DIR / "failure_modes.csv"
SUMMARY_TXT = OUT_DIR / "failure_modes_summary.txt"

IOU_PASS = 0.50     # IoU at or above this counts as a usable segmentation
DISJOINT_T = 0.10   # precision AND recall both below this -> disjoint
MID_T = 0.50        # threshold separating "low" from "high" precision/recall

CATEGORY_ORDER = [
    "pass",
    "total_miss",
    "disjoint_wrong_region",
    "mislocalized",
    "bleed_overseg",
    "partial_underseg",
    "mixed_moderate",
]

CATEGORY_DESCRIPTION = {
    "pass": "IoU >= 0.5",
    "total_miss": "Task 2 produced an empty mask",
    "disjoint_wrong_region": "precision < 0.10 and recall < 0.10 (prediction is on something other than the egg)",
    "mislocalized": "precision < 0.5 and recall < 0.5, not disjoint (mostly off-egg, egg mostly missed)",
    "bleed_overseg": "precision < 0.5 and recall >= 0.5 (egg captured, prediction much larger)",
    "partial_underseg": "precision >= 0.5 and recall < 0.5 (prediction on egg but covers only part of it)",
    "mixed_moderate": "precision >= 0.5 and recall >= 0.5 but IoU < 0.5",
}


def classify(iou, precision, recall, pred_area):
    # Order matters: earlier rules take priority.
    if iou >= IOU_PASS:
        return "pass"
    if pred_area == 0:
        return "total_miss"
    if precision < DISJOINT_T and recall < DISJOINT_T:
        return "disjoint_wrong_region"
    if precision < MID_T and recall < MID_T:
        return "mislocalized"
    if precision < MID_T and recall >= MID_T:
        return "bleed_overseg"
    if precision >= MID_T and recall < MID_T:
        return "partial_underseg"
    return "mixed_moderate"


def load_rows():
    with open(METRICS_CSV, newline="") as f:
        for r in csv.DictReader(f):
            gt_area = int(r["gt_area"])
            pred_area = int(r["pred_area"])
            yield {
                "stem": r["stem"],
                "iou": float(r["iou"]),
                "dice": float(r["dice"]),
                "precision": float(r["precision"]),
                "recall": float(r["recall"]),
                "gt_area": gt_area,
                "pred_area": pred_area,
                "pred_gt_ratio": pred_area / max(gt_area, 1),
            }


def main():
    if not METRICS_CSV.exists():
        raise SystemExit(f"Missing {METRICS_CSV} -- run task3_evaluate.py first.")

    rows = list(load_rows())
    for r in rows:
        r["category"] = classify(r["iou"], r["precision"], r["recall"], r["pred_area"])

    # Per-image CSV
    fields = ["stem", "category", "iou", "dice", "precision", "recall",
              "gt_area", "pred_area", "pred_gt_ratio"]
    with open(FAILURE_CSV, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for r in sorted(rows, key=lambda x: (CATEGORY_ORDER.index(x["category"]), x["iou"])):
            wr.writerow({k: (round(r[k], 4) if isinstance(r[k], float) else r[k]) for k in fields})

    # Summary text
    n = len(rows)
    lines = [f"Total images: {n}", ""]
    for cat in CATEGORY_ORDER:
        members = [r for r in rows if r["category"] == cat]
        if not members:
            lines.append(f"{cat:22s}   0  (0.0%)")
            lines.append(f"    definition: {CATEGORY_DESCRIPTION[cat]}")
            continue
        ious = np.array([m["iou"] for m in members])
        ratios = np.array([m["pred_gt_ratio"] for m in members])
        lines.append(
            f"{cat:22s} {len(members):3d} ({100 * len(members) / n:5.1f}%)  "
            f"mean_iou={ious.mean():.3f}  median_pred/gt_area={np.median(ratios):.2f}"
        )
        lines.append(f"    definition: {CATEGORY_DESCRIPTION[cat]}")

    lines.append("")
    lines.append("=" * 70)
    lines.append("Members by category (sorted by IoU, lowest first)")
    lines.append("=" * 70)
    for cat in CATEGORY_ORDER:
        members = sorted([r for r in rows if r["category"] == cat], key=lambda x: x["iou"])
        lines.append(f"\n[{cat}] ({len(members)})")
        for m in members:
            lines.append(
                f"  {m['stem']}  iou={m['iou']:.3f}  p={m['precision']:.3f}  "
                f"r={m['recall']:.3f}  pred/gt={m['pred_gt_ratio']:.2f}"
            )

    text = "\n".join(lines)
    with open(SUMMARY_TXT, "w") as f:
        f.write(text)

    # Console headline only (full member lists are in the summary file)
    headline = [l for l in lines[: lines.index("") ]] + [lines[lines.index("") + 1]]
    print("\n".join(lines[: 2 + len(CATEGORY_ORDER) * 2]))
    print(f"\nWrote: {FAILURE_CSV}")
    print(f"Wrote: {SUMMARY_TXT}")


if __name__ == "__main__":
    main()