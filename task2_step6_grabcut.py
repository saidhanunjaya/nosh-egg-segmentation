"""
Task 2 - Trial 6: pivot away from contour/shape-scoring entirely.

Diagnosis from Trials 3-5: picking a single contour by hand-tuned shape
heuristics (circularity, solidity, ellipse-IoU, brightness, uniformity) is
fundamentally brittle. Real backgrounds (finger creases, keyboard keys,
fabric weave, ash texture) keep producing small regions that are
geometrically "egg-like" by pure shape, and every patch that fixed one
failing image broke another. Discarding this family of approaches.

New strategy: every photo in this dataset is a close-up shot with one
dominant object roughly centered in frame. Instead of SEARCHING for the
right contour, seed GrabCut with "border = definitely background, center =
probably foreground" and let it build color/texture models (GMMs) to find
the true object boundary. This reasons about GLOBAL color statistics,
not local edge geometry, so it should not be fooled by small locally
egg-shaped regions the way contour scoring was.

Hand vs egg (similar overall shape, different color) is handled by a
skin-color removal pass after GrabCut.
"""

import cv2
import numpy as np
from pathlib import Path

from task2_step0_split import load_or_build_split


def clean_mask(mask, min_area=200):
    """Largest connected component + fill interior holes."""
    m = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = (labels == largest).astype(np.uint8)
    if m.sum() < min_area:
        return None
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled.astype(bool)


def skin_mask_ycrcb(image_bgr):
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], np.uint8)
    upper = np.array([255, 173, 127], np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask > 0


def grabcut_center(image_bgr, margin_frac=0.07, iterations=6):
    """
    Seed GrabCut with a rectangle: everything inside the rectangle is
    'probable foreground', everything outside is 'definite background'.
    GrabCut then builds color GMMs and refines the true boundary.
    """
    h, w = image_bgr.shape[:2]
    mx, my = int(w * margin_frac), int(h * margin_frac)
    rect = (mx, my, w - 2 * mx, h - 2 * my)

    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(image_bgr, mask, rect, bgd_model, fgd_model,
                    iterations, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return np.zeros((h, w), bool)

    fg = np.isin(mask, [cv2.GC_FGD, cv2.GC_PR_FGD])
    return fg


def segment(image_bgr, use_skin_removal=True):
    fg = grabcut_center(image_bgr)
    if use_skin_removal:
        skin = skin_mask_ycrcb(image_bgr)
        fg = fg & (~skin)
    cleaned = clean_mask(fg)
    return cleaned, fg


def draw_overlay(image_bgr, mask):
    overlay = image_bgr.copy()
    if mask is not None:
        overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
    return overlay


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    out_dir = here / "task2_trial6_grabcut"
    out_dir.mkdir(exist_ok=True)

    paths, split = load_or_build_split(
        here, ["train", "valid", "test"],
        here / "task1_outputs_final" / "split.csv")
    dev_paths = [p for p in paths if split[p] == "dev"]
    sample = dev_paths[:8]

    for p in sample:
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue

        cleaned, raw_fg = segment(bgr, use_skin_removal=True)
        overlay = draw_overlay(bgr, cleaned)

        stem = p.stem
        cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), overlay)
        cv2.imwrite(str(out_dir / f"{stem}_mask.png"),
                    (cleaned.astype(np.uint8) * 255) if cleaned is not None
                    else np.zeros(bgr.shape[:2], np.uint8))
        cv2.imwrite(str(out_dir / f"{stem}_raw_fg.png"), (raw_fg.astype(np.uint8)) * 255)

        area = int(cleaned.sum()) if cleaned is not None else 0
        print(f"{p.name}: area={area}")

    print(f"\nWrote outputs to {out_dir}")