"""
Task 2 - Trial 2: Otsu + morphological cleanup + "most egg-like" component picker.
Targets failure mode: speckle/texture noise (steel pan, patterned cloth, text).
Does NOT yet fix: egg blending into background brightness, hand merged with egg.
"""
import cv2
import numpy as np
from pathlib import Path

from task2_step0_split import load_or_build_split


def otsu_mask(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (mask > 0).mean() > 0.5:
        mask = cv2.bitwise_not(mask)
    return mask


def cleanup_and_pick(mask, image_shape, top_k=1):
    h, w = image_shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    # Open removes small speckle, close fills small gaps inside a blob.
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask)

    img_area = h * w
    center = np.array([w / 2.0, h / 2.0])
    diag = (h ** 2 + w ** 2) ** 0.5

    candidates = []
    for lbl in range(1, n):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < 0.005 * img_area or area > 0.6 * img_area:
            continue
        comp = (labels == lbl).astype(np.uint8)
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea)
        perim = cv2.arcLength(c, True)
        circularity = 4 * np.pi * area / (perim ** 2) if perim > 0 else 0
        hull_area = cv2.contourArea(cv2.convexHull(c))
        solidity = area / hull_area if hull_area > 0 else 0
        dist_center = np.linalg.norm(centroids[lbl] - center) / (diag / 2)
        score = circularity * solidity * (1.0 - 0.4 * dist_center)
        candidates.append((score, lbl, area))

    if not candidates:
        return np.zeros_like(mask)

    candidates.sort(reverse=True)
    out = np.zeros_like(mask)
    for score, lbl, area in candidates[:top_k]:
        out[labels == lbl] = 255
    return out


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    out_dir = here / "task2_trial2_cleanup"
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
        raw_mask = otsu_mask(bgr)
        clean_mask = cleanup_and_pick(raw_mask, bgr.shape, top_k=1)

        overlay = bgr.copy()
        overlay[clean_mask > 0] = (0.5 * overlay[clean_mask > 0]
                                    + 0.5 * np.array([0, 0, 255])).astype(np.uint8)

        stem = p.stem
        cv2.imwrite(str(out_dir / f"{stem}_mask.png"), clean_mask)
        cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), overlay)

    print(f"Wrote {len(sample)} mask/overlay pairs to {out_dir}")