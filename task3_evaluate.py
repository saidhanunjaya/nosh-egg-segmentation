"""
Task 3 - Evaluate Task 2 (classical) against Task 1 (SAM) as ground truth.

For every image in the dev+eval split, loads:
    task1_outputs_final/masks/{stem}_mask.png   (ground truth)
    task2_outputs_final/masks/{stem}_mask.png   (prediction)
and computes IoU, Dice, precision, recall.

Handles the possible filename mismatch: Task 1 prefixes the stem with the
class-folder name only when two images share an identical stem across
train/valid/test ("{folder}_{stem}_mask.png"); otherwise it's a plain
"{stem}_mask.png", same as Task 2 always uses.

Outputs:
    task3_outputs/metrics.csv          per-image IoU/Dice/precision/recall
    task3_outputs/summary.txt          aggregate stats (mean/median/worst)
    task3_outputs/worst_cases/         side-by-side GT vs prediction overlays
                                        for the N lowest-IoU images
"""

import csv
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
T1_MASKS = HERE / "task1_outputs_final" / "masks"
T2_MASKS = HERE / "task2_outputs_final" / "masks"
T1_IMAGES_ROOT = HERE  # train/valid/test subfolders live here
OUT_DIR = HERE / "task3_outputs"
WORST_DIR = OUT_DIR / "worst_cases"
N_WORST = 12


def build_t1_lookup():
    """Map a bare stem -> Task 1 mask path, stripping a folder-name prefix
    if present (e.g. 'train_not_damaged_101_...' -> 'not_damaged_101_...').
    Warns if stripping causes two different masks to collide on the same key."""
    lookup = {}
    collisions = []
    folders = {"train", "valid", "test"}
    for p in T1_MASKS.glob("*_mask.png"):
        stem = p.name[:-len("_mask.png")]
        lookup[stem] = p
        parts = stem.split("_", 1)
        if len(parts) == 2 and parts[0] in folders:
            short = parts[1]
            if short in lookup and lookup[short] != p:
                collisions.append(short)
            lookup[short] = p
    if collisions:
        print(f"WARNING: {len(collisions)} stem collisions in T1 lookup (duplicate "
              f"stems across splits may shadow each other): {collisions[:5]}")
    return lookup


def find_original_image(stem):
    """Recursively search train/valid/test for the original image, since the
    Roboflow 'Folder' export may nest images under a class subfolder
    (e.g. train/not_damaged/xxx.jpg) rather than directly under train/."""
    exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    for sub in ("train", "valid", "test"):
        base = T1_IMAGES_ROOT / sub
        if not base.exists():
            continue
        for ext in exts:
            matches = list(base.rglob(f"{stem}{ext}"))
            if matches:
                return matches[0]
    return None


def load_binary(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return img > 127


def compute_metrics(gt, pred):
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    tp = np.logical_and(gt, pred).sum()
    fp = np.logical_and(~gt, pred).sum()
    fn = np.logical_and(gt, ~pred).sum()
    union = np.logical_or(gt, pred).sum()

    iou = tp / union if union > 0 else (1.0 if gt.sum() == 0 and pred.sum() == 0 else 0.0)
    dice = (2 * tp) / (gt.sum() + pred.sum()) if (gt.sum() + pred.sum()) > 0 else 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if pred.sum() == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if gt.sum() == 0 else 0.0)
    return iou, dice, precision, recall


def make_comparison_image(image_bgr, gt, pred):
    """Green = GT only, Red = prediction only, Yellow = overlap."""
    overlay = image_bgr.copy()
    gt_only = gt & ~pred
    pred_only = pred & ~gt
    both = gt & pred

    overlay[gt_only] = (0.4 * overlay[gt_only] + 0.6 * np.array([0, 255, 0])).astype(np.uint8)
    overlay[pred_only] = (0.4 * overlay[pred_only] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
    overlay[both] = (0.4 * overlay[both] + 0.6 * np.array([0, 255, 255])).astype(np.uint8)
    return overlay


def main():
    OUT_DIR.mkdir(exist_ok=True)
    WORST_DIR.mkdir(exist_ok=True)

    t1_lookup = build_t1_lookup()
    t2_masks = sorted(T2_MASKS.glob("*_mask.png"))
    print(f"Found {len(t2_masks)} Task 2 masks, {len(t1_lookup)} Task 1 masks indexed.")

    rows = []
    missing = []

    for t2_path in t2_masks:
        stem = t2_path.name[:-len("_mask.png")]
        t1_path = t1_lookup.get(stem)
        if t1_path is None:
            missing.append(stem)
            continue

        gt = load_binary(t1_path)
        pred = load_binary(t2_path)
        if gt is None or pred is None:
            missing.append(stem)
            continue
        if gt.shape != pred.shape:
            pred = cv2.resize(pred.astype(np.uint8), (gt.shape[1], gt.shape[0]),
                               interpolation=cv2.INTER_NEAREST) > 0

        iou, dice, precision, recall = compute_metrics(gt, pred)
        rows.append({
            "stem": stem,
            "iou": round(iou, 4),
            "dice": round(dice, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "gt_area": int(gt.sum()),
            "pred_area": int(pred.sum()),
        })

    if missing:
        print(f"WARNING: {len(missing)} Task 2 masks had no matching Task 1 mask "
              f"(or failed to load) -- skipped. First few: {missing[:5]}")

    rows.sort(key=lambda r: r["iou"])

    with open(OUT_DIR / "metrics.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    ious = np.array([r["iou"] for r in rows])
    dices = np.array([r["dice"] for r in rows])
    precisions = np.array([r["precision"] for r in rows])
    recalls = np.array([r["recall"] for r in rows])

    summary_lines = [
        f"Images evaluated: {len(rows)}  (missing/unmatched: {len(missing)})",
        "",
        f"IoU       mean={ious.mean():.4f}  median={np.median(ious):.4f}  "
        f"min={ious.min():.4f}  max={ious.max():.4f}",
        f"Dice      mean={dices.mean():.4f}  median={np.median(dices):.4f}",
        f"Precision mean={precisions.mean():.4f}  median={np.median(precisions):.4f}",
        f"Recall    mean={recalls.mean():.4f}  median={np.median(recalls):.4f}",
        "",
        f"IoU >= 0.75 (good):      {(ious >= 0.75).sum()} / {len(rows)}",
        f"IoU 0.5-0.75 (ok):       {((ious >= 0.5) & (ious < 0.75)).sum()} / {len(rows)}",
        f"IoU < 0.5 (poor/failed): {(ious < 0.5).sum()} / {len(rows)}",
        "",
        "Worst 10 by IoU:",
    ]
    for r in rows[:10]:
        summary_lines.append(f"  {r['stem']}: iou={r['iou']:.3f} dice={r['dice']:.3f} "
                              f"precision={r['precision']:.3f} recall={r['recall']:.3f}")

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)
    with open(OUT_DIR / "summary.txt", "w") as f:
        f.write(summary_text)

    print(f"\nGenerating side-by-side visualizations for {N_WORST} worst cases...")
    n_written = 0
    n_no_image = 0
    for r in rows[:N_WORST]:
        stem = r["stem"]
        img_path = find_original_image(stem)
        if img_path is None:
            n_no_image += 1
            print(f"  [skip] no original image found for stem: {stem}")
            continue
        image_bgr = cv2.imread(str(img_path))
        gt = load_binary(t1_lookup[stem])
        pred = load_binary(T2_MASKS / f"{stem}_mask.png")
        if image_bgr is None or gt is None or pred is None:
            print(f"  [skip] failed to load image/mask for stem: {stem}")
            continue
        if gt.shape != image_bgr.shape[:2]:
            gt = cv2.resize(gt.astype(np.uint8), (image_bgr.shape[1], image_bgr.shape[0]),
                             interpolation=cv2.INTER_NEAREST) > 0
        if pred.shape != image_bgr.shape[:2]:
            pred = cv2.resize(pred.astype(np.uint8), (image_bgr.shape[1], image_bgr.shape[0]),
                               interpolation=cv2.INTER_NEAREST) > 0

        comp = make_comparison_image(image_bgr, gt, pred)
        label = f"iou={r['iou']:.2f}"
        cv2.putText(comp, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        ok = cv2.imwrite(str(WORST_DIR / f"{stem}_compare.png"), comp)
        if not ok:
            print(f"  [warn] cv2.imwrite returned False for stem: {stem}")
        else:
            n_written += 1

    print(f"\nWrote {n_written}/{N_WORST} comparison images "
          f"({n_no_image} had no original image found).")
    print(f"Done. Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()