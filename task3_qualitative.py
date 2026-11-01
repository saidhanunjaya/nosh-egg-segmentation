"""
Task 3 - Step 3: generate labeled comparison images for a hand-picked set of
representative examples, one or more per failure-mode category, for direct
use in the written report.

Categories total_miss and disjoint_wrong_region already have compare images
in task3_outputs/worst_cases/ from the earlier worst-12 run, so this script
only generates the remaining selected examples (pass, mislocalized,
bleed_overseg, partial_underseg).

Reads:  task1_outputs_final/masks/   (ground truth)
        task2_outputs_final/masks/  (prediction)
        train/valid/test/           (original images)
Writes: task3_outputs/qualitative/{category}__{stem}.png
"""

from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
T1_MASKS = HERE / "task1_outputs_final" / "masks"
T2_MASKS = HERE / "task2_outputs_final" / "masks"
IMAGES_ROOT = HERE
OUT_DIR = HERE / "task3_outputs" / "qualitative"

# category -> list of (stem, iou, precision, recall) picked from failure_modes_summary.txt
SELECTED = {
    "pass": [
        ("not_damaged_161_jpg.rf.fd216aa5e11e147fc1527478cceae94e", 0.515, 0.632, 0.736),
        ("not_damaged_34_jpg.rf.4be2a53f051e4372100c90f2d36ea129", 0.994, 0.998, 0.997),
    ],
    "mislocalized": [
        ("not_damaged_114_jpg.rf.7ccf87a5610bafd963468463b2e2907c", 0.170, 0.228, 0.400),
        ("not_damaged_28_jpg.rf.a7e8555e9e2b27a796da5d502582db24", 0.043, 0.046, 0.384),
    ],
    "bleed_overseg": [
        ("not_damaged_126_jpg.rf.2c3342cf9ca7ab55f6682ef198e98bbd", 0.225, 0.225, 1.000),
        ("not_damaged_135_jpg.rf.a6d81ffb82470e031fb3ab469c8f0db2", 0.057, 0.057, 1.000),
        ("not_damaged_43_jpg.rf.66c31342150e751c30214786e72c8683", 0.269, 0.291, 0.782),
    ],
    "partial_underseg": [
        ("not_damaged_133_jpg.rf.a386f448c321cfb45a902bab446a0536", 0.322, 0.995, 0.322),
        ("not_damaged_140_jpg.rf.5a28b6a69e229bf384854ec1e98b3f19", 0.280, 1.000, 0.280),
    ],
}


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


def find_original_image(stem):
    exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    for sub in ("train", "valid", "test"):
        base = IMAGES_ROOT / sub
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


def draw_label(img, lines):
    """Draw a readable multi-line label with a dark background strip at the top."""
    h = 24 * len(lines) + 12
    strip = img[:h].copy()
    img[:h] = (0.25 * strip).astype(np.uint8)  # darken top strip for contrast
    for i, line in enumerate(lines):
        y = 24 * (i + 1)
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t1_lookup = build_t1_lookup()

    n_written = 0
    for category, items in SELECTED.items():
        for stem, iou, precision, recall in items:
            img_path = find_original_image(stem)
            t1_path = t1_lookup.get(stem)
            t2_path = T2_MASKS / f"{stem}_mask.png"

            if img_path is None or t1_path is None or not t2_path.exists():
                print(f"  [skip] missing data for {category} / {stem}")
                continue

            image_bgr = cv2.imread(str(img_path))
            gt = load_binary(t1_path)
            pred = load_binary(t2_path)
            if image_bgr is None or gt is None or pred is None:
                print(f"  [skip] failed to load for {category} / {stem}")
                continue

            if gt.shape != image_bgr.shape[:2]:
                gt = cv2.resize(gt.astype(np.uint8), (image_bgr.shape[1], image_bgr.shape[0]),
                                 interpolation=cv2.INTER_NEAREST) > 0
            if pred.shape != image_bgr.shape[:2]:
                pred = cv2.resize(pred.astype(np.uint8), (image_bgr.shape[1], image_bgr.shape[0]),
                                   interpolation=cv2.INTER_NEAREST) > 0

            comp = make_comparison_image(image_bgr, gt, pred)
            comp = draw_label(comp, [
                f"{category}",
                f"iou={iou:.3f}  p={precision:.3f}  r={recall:.3f}",
            ])

            out_path = OUT_DIR / f"{category}__{stem}.png"
            ok = cv2.imwrite(str(out_path), comp)
            if ok:
                n_written += 1
                print(f"  [ok] {out_path.name}")
            else:
                print(f"  [warn] imwrite failed for {stem}")

    print(f"\nWrote {n_written} qualitative example images to: {OUT_DIR}")


if __name__ == "__main__":
    main()