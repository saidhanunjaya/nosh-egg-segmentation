"""
Task 2 - Trial 8: fix the two regressions introduced in Trial 7.

Diagnosis from Trial 7 (full 8-image comparison against Trial 6):
  - Every image got flat or worse; 0/8 improved.
  Bug 1: the hard "sure-foreground" center core (GC_FGD) is a constraint
    GrabCut can never undo. When that box includes background pixels
    (true on nearly every image here), the foreground color model gets
    poisoned with background colors, causing MORE background bleed than
    Trial 6's plain soft rectangle. Fix: remove GC_FGD entirely, keep
    only a thin hard GC_BGD border + graded GC_PR_BGD/GC_PR_FGD interior.
  Bug 2: picking the "largest connected component" after skin removal
    grabbed a totally unrelated red object on the hand image, because
    skin removal fragmented the true foreground and an unrelated object
    elsewhere in frame happened to be the largest leftover piece.
    Fix: pick the component whose centroid is closest to the image
    center (subject to a minimum area), matching the same "object is
    roughly centered" assumption the GrabCut seed itself relies on.
"""

import cv2
import numpy as np
from pathlib import Path

from task2_step0_split import load_or_build_split


def build_soft_trimap(h, w, sure_bg_frac=0.05, prob_bg_frac=0.20):
    """
    Only two hard/soft levels, no hard foreground anywhere:
      0..sure_bg_frac from edge   -> GC_BGD      (hard background)
      sure_bg_frac..prob_bg_frac  -> GC_PR_BGD    (probable background)
      beyond prob_bg_frac (center)-> GC_PR_FGD    (probable foreground)
    """
    mask = np.full((h, w), cv2.GC_BGD, np.uint8)

    def rect_from_frac(f):
        mx, my = int(w * f), int(h * f)
        return mx, my, w - mx, h - my

    x, y, rw, rh = rect_from_frac(sure_bg_frac)
    mask[y:y + rh, x:x + rw] = cv2.GC_PR_BGD

    x, y, rw, rh = rect_from_frac(prob_bg_frac)
    mask[y:y + rh, x:x + rw] = cv2.GC_PR_FGD

    return mask


def run_grabcut_soft(image_bgr, sure_bg_frac, prob_bg_frac, iterations=6):
    h, w = image_bgr.shape[:2]
    mask = build_soft_trimap(h, w, sure_bg_frac, prob_bg_frac)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(image_bgr, mask, None, bgd_model, fgd_model,
                    iterations, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return np.zeros((h, w), bool)
    return np.isin(mask, [cv2.GC_FGD, cv2.GC_PR_FGD])


def skin_mask_ycrcb(image_bgr, min_saturation=40):
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], np.uint8)
    upper = np.array([255, 173, 127], np.uint8)
    chroma_mask = cv2.inRange(ycrcb, lower, upper)

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    sat_mask = (hsv[:, :, 1] >= min_saturation).astype(np.uint8) * 255

    mask = cv2.bitwise_and(chroma_mask, sat_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask > 0


def pick_central_component(mask, min_area_frac=0.01):
    """Pick the connected component closest to the image center, not the
    largest one -- avoids grabbing an unrelated object elsewhere in frame
    once skin removal fragments the foreground."""
    h, w = mask.shape
    m = mask.astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return None

    cx, cy = w / 2.0, h / 2.0
    min_area = min_area_frac * h * w

    best_idx, best_dist = None, None
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        gx, gy = centroids[i]
        dist = (gx - cx) ** 2 + (gy - cy) ** 2
        if best_dist is None or dist < best_dist:
            best_dist, best_idx = dist, i

    if best_idx is None:
        return None

    comp = (labels == best_idx).astype(np.uint8)
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(comp)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled.astype(bool)


def segment(image_bgr, use_skin_removal=True, bleed_frac_thresh=0.55):
    h, w = image_bgr.shape[:2]
    total = h * w

    fg = run_grabcut_soft(image_bgr, 0.05, 0.20)
    retried = False
    frac = fg.sum() / total
    if frac > bleed_frac_thresh:
        fg2 = run_grabcut_soft(image_bgr, 0.08, 0.28)
        retried = True
        if fg2.sum() / total < frac:
            fg = fg2

    if use_skin_removal:
        skin = skin_mask_ycrcb(image_bgr)
        fg = fg & (~skin)

    cleaned = pick_central_component(fg)
    final_frac = (cleaned.sum() / total) if cleaned is not None else 0.0
    return cleaned, fg, retried, final_frac


def draw_overlay(image_bgr, mask):
    overlay = image_bgr.copy()
    if mask is not None:
        overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
    return overlay


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    out_dir = here / "task2_trial8_grabcut_centerpick"
    out_dir.mkdir(exist_ok=True)

    paths, split = load_or_build_split(
        here, ["train", "valid", "test"],
        here / "task1_outputs_final" / "split.csv")
    dev_paths = [p for p in paths if split[p] == "dev"]
    sample = dev_paths[:8]

    for idx, p in enumerate(sample, start=1):
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        cleaned, raw_fg, retried, final_frac = segment(bgr, use_skin_removal=True)
        overlay = draw_overlay(bgr, cleaned)

        stem = p.stem
        cv2.imwrite(str(out_dir / f"{idx:02d}_{stem}_overlay.png"), overlay)
        cv2.imwrite(str(out_dir / f"{idx:02d}_{stem}_mask.png"),
                    (cleaned.astype(np.uint8) * 255) if cleaned is not None
                    else np.zeros(bgr.shape[:2], np.uint8))

        area = int(cleaned.sum()) if cleaned is not None else 0
        print(f"[{idx}] {p.name}: area={area}  frame_frac={final_frac:.3f}  retried={retried}")

    print(f"\nWrote outputs to {out_dir}")