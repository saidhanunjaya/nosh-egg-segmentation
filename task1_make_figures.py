"""Builds 'Original | SAM mask' side-by-side images for report Figures 1-4.
Run:  python task1_make_figures.py
Output: task1_figures/<stem>_original_vs_sam.png
"""
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "task1_figures"
OUT.mkdir(exist_ok=True)

STEMS = ["not_damaged_2", "not_damaged_11", "not_damaged_18", "not_damaged_125"]
SKIP_DIRS = {"task1_outputs_final", "task2_outputs_final", "task3_outputs",
             "task4_figures", "task1_figures", ".venv"}
TARGET_H = 360


def imread(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite(p, img):
    ok, buf = cv2.imencode(".png", img)
    buf.tofile(str(p))


def find_original(stem):
    for ext in ("jpg", "jpeg", "png"):
        for p in sorted(ROOT.rglob("%s_jpg.rf.*.%s" % (stem, ext))):
            if not (set(p.parts) & SKIP_DIRS):
                return p
    return None


def find_overlay(stem):
    hits = sorted((ROOT / "task1_outputs_final" / "overlays").glob("%s_jpg.rf.*_overlay.png" % stem))
    return hits[0] if hits else None


def resize_h(img, h):
    s = h / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * s), h), interpolation=cv2.INTER_AREA)


def titled(img, text):
    bar = np.full((30, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


for stem in STEMS:
    orig_p, ov_p = find_original(stem), find_overlay(stem)
    print(stem, "| original:", orig_p, "| overlay:", ov_p)
    if orig_p is None or ov_p is None:
        print("  -> SKIPPED (file not found)")
        continue
    orig, ov = imread(orig_p), imread(ov_p)
    print("  sizes: original", orig.shape[:2], "overlay", ov.shape[:2])
    if ov.shape[1] >= 1.8 * orig.shape[1]:
        print("  -> overlay is already side by side, copying as is")
        imwrite(OUT / ("%s_original_vs_sam.png" % stem), ov)
        continue
    left = titled(resize_h(orig, TARGET_H), "Original")
    right = titled(resize_h(ov, TARGET_H), "SAM mask")
    gap = np.full((left.shape[0], 8, 3), 255, np.uint8)
    imwrite(OUT / ("%s_original_vs_sam.png" % stem), np.hstack([left, gap, right]))
    print("  -> saved")