"""
Task 2 - Trial 7: GrabCut with proper trimap seeding (not just a rectangle)
+ stricter skin filter + background-bleed sanity retry.

Diagnosis from Trial 6 (8/8 dev images reviewed):
  - 1/8 clean success (rock bg), 1/8 near-success (paper), rest failed in
    two distinct ways:
  (A) Skin removal ate into egg highlights (pale/low-saturation pixels were
      wrongly caught by the YCrCb skin range) -> fix: require a saturation
      floor so only actually-saturated skin tones are removed.
  (B) Plain-rectangle seeding gives GrabCut no guaranteed-foreground
      anchor, so on busy/textured backgrounds (steel pan, ash/rock,
      patterned cloth) the color-GMM optimizer sometimes flips which
      cluster it calls "foreground" -> on the cloth image this produced a
      complete inversion (textured cloth labeled FG, smooth egg labeled BG).
      Fix: seed with a real trimap (sure-bg border, probable-bg band,
      probable-fg band, sure-fg center core) so the optimizer has a strong
      anchor and can't flip the whole frame.

Added: a size-sanity retry (not a shape/circularity filter) -- if the
resulting mask is implausibly large (likely background bleed), retry once
with a tighter trimap. This is just a sanity gate, it does not re-introduce
the discarded contour-scoring approach.
"""

import cv2
import numpy as np
from pathlib import Path

from task2_step0_split import load_or_build_split


def build_trimap_mask(h, w,
                       sure_bg_frac=0.05,
                       prob_bg_frac=0.18,
                       prob_fg_frac=0.35):
    """
    Nested-rectangle trimap:
      0..sure_bg_frac from edge          -> GC_BGD  (sure background)
      sure_bg_frac..prob_bg_frac         -> GC_PR_BGD
      prob_bg_frac..prob_fg_frac         -> GC_PR_FGD
      beyond prob_fg_frac (center core)  -> GC_FGD  (sure foreground)
    """
    mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)

    def rect_from_frac(f):
        mx, my = int(w * f), int(h * f)
        return mx, my, w - mx, h - my

    # start everything as sure background, then carve inward
    mask[:, :] = cv2.GC_BGD
    x, y, rw, rh = rect_from_frac(sure_bg_frac)
    mask[y:y + rh, x:x + rw] = cv2.GC_PR_BGD

    x, y, rw, rh = rect_from_frac(prob_bg_frac)
    mask[y:y + rh, x:x + rw] = cv2.GC_PR_FGD

    x, y, rw, rh = rect_from_frac(prob_fg_frac)
    mask[y:y + rh, x:x + rw] = cv2.GC_FGD

    return mask


def run_grabcut_trimap(image_bgr, sure_bg_frac, prob_bg_frac, prob_fg_frac,
                        iterations=6):
    h, w = image_bgr.shape[:2]
    mask = build_trimap_mask(h, w, sure_bg_frac, prob_bg_frac, prob_fg_frac)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(image_bgr, mask, None, bgd_model, fgd_model,
                    iterations, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return np.zeros((h, w), bool)
    return np.isin(mask, [cv2.GC_FGD, cv2.GC_PR_FGD])


def clean_mask(mask, min_area=200):
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


def skin_mask_ycrcb(image_bgr, min_saturation=40):
    """Skin color in YCrCb, gated by HSV saturation so pale/white egg
    highlights (low saturation) are never classified as skin."""
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


def segment(image_bgr, use_skin_removal=True, bleed_frac_thresh=0.55):
    h, w = image_bgr.shape[:2]
    total = h * w

    # pass 1: normal trimap
    fg = run_grabcut_trimap(image_bgr, 0.05, 0.18, 0.35)
    retried = False

    frac = fg.sum() / total
    if frac > bleed_frac_thresh:
        # pass 2: tighter core, bigger sure-bg border -> stronger anchor
        fg2 = run_grabcut_trimap(image_bgr, 0.10, 0.28, 0.45)
        retried = True
        if fg2.sum() / total < frac:
            fg = fg2

    if use_skin_removal:
        skin = skin_mask_ycrcb(image_bgr)
        fg = fg & (~skin)

    cleaned = clean_mask(fg)
    final_frac = (cleaned.sum() / total) if cleaned is not None else 0.0
    return cleaned, fg, retried, final_frac


def draw_overlay(image_bgr, mask):
    overlay = image_bgr.copy()
    if mask is not None:
        overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
    return overlay


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    out_dir = here / "task2_trial7_grabcut_trimap"
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

        cleaned, raw_fg, retried, final_frac = segment(bgr, use_skin_removal=True)
        overlay = draw_overlay(bgr, cleaned)

        stem = p.stem
        cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), overlay)
        cv2.imwrite(str(out_dir / f"{stem}_mask.png"),
                    (cleaned.astype(np.uint8) * 255) if cleaned is not None
                    else np.zeros(bgr.shape[:2], np.uint8))

        area = int(cleaned.sum()) if cleaned is not None else 0
        print(f"{p.name}: area={area}  frame_frac={final_frac:.3f}  retried={retried}")

    print(f"\nWrote outputs to {out_dir}")