"""
Task 3 - Step 3.5: test whether 'partial_underseg' / 'mislocalized' / 'total_miss'
failures are disproportionately multi-egg (multi-instance) scenes, by counting
connected components in each Task 1 ground-truth mask and cross-tabulating
against the failure-mode categories from task3_failure_modes.py.

A GT mask with >=2 separate connected components (above a small noise-area
threshold) is treated as a "multi-instance" scene.

Reads:  task1_outputs_final/masks/*_mask.png
        task3_outputs/failure_modes.csv
Writes: task3_outputs/multiinstance_crosstab.txt
"""

import csv
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
T1_MASKS = HERE / "task1_outputs_final" / "masks"
OUT_DIR = HERE / "task3_outputs"
FAILURE_CSV = OUT_DIR / "failure_modes.csv"
OUT_TXT = OUT_DIR / "multiinstance_crosstab.txt"

CATEGORY_ORDER = [
    "pass",
    "total_miss",
    "disjoint_wrong_region",
    "mislocalized",
    "bleed_overseg",
    "partial_underseg",
    "mixed_moderate",
]

# A connected component smaller than this fraction of the mask's total
# foreground area is treated as noise, not a real second egg.
MIN_COMPONENT_FRAC = 0.03


def build_t1_lookup():
    lookup = {}
    folders = {"train", "valid", "test"}
    for p in T1_MASKS.glob("*_mask.png"):
        stem = p.name[:-len("_mask.png")]
        lookup[stem] = p
        parts = stem.split("_", 1)
        if len(parts) == 2 and parts[0] in folders:
            lookup[parts[1]] = p
    return lookup


def count_instances(mask_path):
    img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    binary = (img > 127).astype(np.uint8)
    total_area = binary.sum()
    if total_area == 0:
        return 0
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    # stats[0] is the background component; skip it
    areas = stats[1:, cv2.CC_STAT_AREA]
    significant = areas[areas >= MIN_COMPONENT_FRAC * total_area]
    return len(significant)


def load_failure_modes():
    rows = []
    with open(FAILURE_CSV, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def main():
    if not FAILURE_CSV.exists():
        raise SystemExit(f"Missing {FAILURE_CSV} -- run task3_failure_modes.py first.")

    t1_lookup = build_t1_lookup()
    rows = load_failure_modes()

    n_instances = {}
    missing = []
    for r in rows:
        stem = r["stem"]
        path = t1_lookup.get(stem)
        if path is None:
            missing.append(stem)
            continue
        n = count_instances(path)
        n_instances[stem] = n

    if missing:
        print(f"WARNING: {len(missing)} stems had no GT mask found: {missing[:5]}")

    for r in rows:
        r["n_instances"] = n_instances.get(r["stem"])
        r["multi_instance"] = (r["n_instances"] is not None and r["n_instances"] >= 2)

    valid_rows = [r for r in rows if r["n_instances"] is not None]
    n_total = len(valid_rows)
    n_multi_total = sum(1 for r in valid_rows if r["multi_instance"])

    lines = []
    lines.append(f"Total images with valid GT mask: {n_total}")
    lines.append(f"Multi-instance GT scenes overall (>=2 components, "
                 f"each >= {int(MIN_COMPONENT_FRAC*100)}% of mask area): "
                 f"{n_multi_total} ({100*n_multi_total/n_total:.1f}%)")
    lines.append("")
    lines.append(f"{'category':22s} {'n':>4s} {'multi-inst n':>13s} {'multi-inst %':>13s}")
    lines.append("-" * 56)

    for cat in CATEGORY_ORDER:
        members = [r for r in valid_rows if r["category"] == cat]
        if not members:
            lines.append(f"{cat:22s} {0:4d} {0:13d} {'n/a':>13s}")
            continue
        n_cat = len(members)
        n_multi = sum(1 for m in members if m["multi_instance"])
        pct = 100 * n_multi / n_cat
        lines.append(f"{cat:22s} {n_cat:4d} {n_multi:13d} {pct:12.1f}%")

    lines.append("")
    lines.append("Baseline (overall dataset) multi-instance rate: "
                  f"{100*n_multi_total/n_total:.1f}%  <- compare each category's % against this")

    lines.append("")
    lines.append("=" * 70)
    lines.append("Multi-instance members by category (for spot-checking)")
    lines.append("=" * 70)
    for cat in CATEGORY_ORDER:
        members = [r for r in valid_rows if r["category"] == cat and r["multi_instance"]]
        if not members:
            continue
        lines.append(f"\n[{cat}] multi-instance members ({len(members)}):")
        for m in members:
            lines.append(f"  {m['stem']}  n_instances={m['n_instances']}  "
                         f"iou={float(m['iou']):.3f}")

    text = "\n".join(lines)
    print(text[:text.index("=" * 70)])  # print summary table to console, skip long member lists
    with open(OUT_TXT, "w") as f:
        f.write(text)
    print(f"\nFull output (incl. member lists) written to: {OUT_TXT}")


if __name__ == "__main__":
    main()