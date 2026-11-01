"""
Task 2 - Trial 4: Otsu (area evidence) + Canny (boundary evidence) fusion,
then shape-based component scoring (same scorer idea as Trial 3).

Why this is a different strategy from Trial 3, not a patch on it:
Trial 3 relied on Canny edges alone to define candidate SHAPE via contour
topology (outer contours + holes). That was fragile -- small local texture
artifacts (shadow folds, fabric weave, key edges) could accidentally form
clean convex "holes" that fooled the shape scorer.

Here we use Otsu to get a rough solid REGION (brightness-based area
evidence), OR it with dilated Canny edges to help seal small gaps in that
region's boundary, then fill external contours to get solid filled blobs.
The shape scorer then runs on solid connected components, not fragile
hole/outer contour topology -- so a shadow fold or key edge can't form an
isolated fake "egg-shaped hole" the way it could in Trial 3.
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


def canny_edges(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edges = cv2.Canny(blur, 35, 110)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    edges = cv2.dilate(edges, kernel, iterations=2)
    return edges


def combined_region(image_bgr):
    """OR the two signals, close, then fill external contours so the
    interior of any closed-ish loop becomes a solid blob."""
    otsu = otsu_mask(image_bgr)
    edges = canny_edges(image_bgr)

    combined = cv2.bitwise_or(otsu, edges)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(combined)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)

    return filled, otsu, edges


def mask_iou(a, b):
    a, b = a > 0, b > 0
    union = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / union if union else 0.0


def score_component(comp_mask, image_shape):
    h, w = image_shape[:2]
    img_area = h * w

    contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 0.004 * img_area or area > 0.55 * img_area:
        return None

    perimeter = cv2.arcLength(c, True)
    if perimeter <= 0:
        return None

    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return None
    solidity = area / hull_area
    circularity = 4 * np.pi * area / (perimeter * perimeter)

    if len(c) < 5:
        return None
    (cx, cy), (axis_a, axis_b), angle = cv2.fitEllipse(c)
    if min(axis_a, axis_b) <= 1:
        return None
    aspect_ratio = max(axis_a, axis_b) / min(axis_a, axis_b)
    if not (1.0 <= aspect_ratio <= 2.3):
        return None

    ellipse_mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(ellipse_mask, ((int(cx), int(cy)), (int(axis_a), int(axis_b)), angle),
                255, thickness=cv2.FILLED)
    ellipse_iou = mask_iou(comp_mask, ellipse_mask)

    if solidity < 0.75:
        return None
    if circularity < 0.35:
        return None
    if ellipse_iou < 0.45:
        return None

    center = np.array([w / 2, h / 2])
    dist = np.linalg.norm(np.array([cx, cy]) - center)
    max_dist = np.sqrt(w * w + h * h) / 2
    center_score = max(0.0, 1.0 - dist / max_dist)

    score = (0.30 * ellipse_iou + 0.30 * solidity
             + 0.20 * circularity + 0.20 * center_score)

    filled = np.zeros((h, w), np.uint8)
    cv2.drawContours(filled, [c], -1, 255, thickness=cv2.FILLED)

    return {
        "mask": filled, "score": score, "area": area,
        "solidity": solidity, "circularity": circularity,
        "aspect_ratio": aspect_ratio, "ellipse_iou": ellipse_iou,
        "ellipse": ((int(cx), int(cy)), (int(axis_a), int(axis_b)), float(angle)),
    }


def detect(image_bgr, top_k=2):
    region, otsu_dbg, edges_dbg = combined_region(image_bgr)
    n, labels = cv2.connectedComponents(region)

    candidates = []
    for lbl in range(1, n):
        comp = (labels == lbl).astype(np.uint8) * 255
        cand = score_component(comp, image_bgr.shape)
        if cand is not None:
            candidates.append(cand)

    candidates.sort(key=lambda x: x["score"], reverse=True)
    kept = []
    for c in candidates:
        if all(mask_iou(c["mask"], k["mask"]) < 0.65 for k in kept):
            kept.append(c)
        if len(kept) >= top_k:
            break

    final_mask = np.zeros(image_bgr.shape[:2], np.uint8)
    for c in kept:
        final_mask[c["mask"] > 0] = 255

    return final_mask, kept, region, otsu_dbg, edges_dbg


def draw_overlay(image_bgr, mask, candidates):
    overlay = image_bgr.copy()
    region = mask > 0
    overlay[region] = (0.5 * overlay[region] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
    for i, c in enumerate(candidates, start=1):
        cv2.ellipse(overlay, c["ellipse"], (0, 255, 0), 2)
        cx, cy = c["ellipse"][0]
        cv2.putText(overlay, f"{i}: {c['score']:.2f}", (cx, max(18, cy - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    return overlay


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    out_dir = here / "task2_trial4_combined"
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
        mask, candidates, region, otsu_dbg, edges_dbg = detect(bgr, top_k=2)
        overlay = draw_overlay(bgr, mask, candidates)

        stem = p.stem
        cv2.imwrite(str(out_dir / f"{stem}_mask.png"), mask)
        cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), overlay)
        cv2.imwrite(str(out_dir / f"{stem}_region.png"), region)
        cv2.imwrite(str(out_dir / f"{stem}_otsu.png"), otsu_dbg)
        cv2.imwrite(str(out_dir / f"{stem}_edges.png"), edges_dbg)

        print(f"{p.name}: {len(candidates)} candidate(s)"
              + (f", best score={candidates[0]['score']:.3f}" if candidates else ""))

    print(f"\nWrote outputs to {out_dir}")