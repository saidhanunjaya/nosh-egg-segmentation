"""
Task 2 - Trial 5: Otsu+Canny combined region, searched with hole-aware
contours, lighter closing, relative-size filter, soft brightness tie-break.

Diagnosis from Trial 4: OR-ing Otsu and Canny then closing aggressively
(9x9, 3 iters) fused the egg into one giant background blob -- worse
over-merging than Otsu alone. detect() only checked RETR_EXTERNAL contours,
so once the egg became part of a bigger blob instead of its own component,
it was invisible (0 candidates on all 8 test images).

Fix: reduce closing aggressiveness, and search BOTH outer and hole contours
(RETR_CCOMP) like Trial 3b did for pure edges -- the egg can appear as an
enclosed hole inside a larger merged blob instead of being lost entirely.
Brightness/uniformity checks from 3d/3e caused regressions when used as
hard filters (shadowed eggs got rejected), so here they only ADD to the
score as a tie-breaker, never hard-reject a candidate.
"""

import cv2
import numpy as np
from pathlib import Path

from task2_step0_split import load_or_build_split


def mask_iou(a, b):
    a, b = a > 0, b > 0
    union = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / union if union else 0.0


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
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=1)
    return edges


def combined_region(image_bgr):
    """Lighter closing than Trial 4 -- just enough to seal tiny pinholes,
    not enough to fuse separate real-world objects together."""
    otsu = otsu_mask(image_bgr)
    edges = canny_edges(image_bgr)
    combined = cv2.bitwise_or(otsu, edges)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)
    return combined, otsu, edges


def contour_to_candidate(contour, image_bgr, is_hole, scene_mean_v):
    h, w = image_bgr.shape[:2]
    image_area = h * w

    area = cv2.contourArea(contour)
    if area < 0.004 * image_area or area > 0.55 * image_area:
        return None

    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return None

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return None
    solidity = area / hull_area
    circularity = 4 * np.pi * area / (perimeter * perimeter)

    if len(contour) < 5:
        return None
    (cx, cy), (axis_a, axis_b), angle = cv2.fitEllipse(contour)
    if min(axis_a, axis_b) <= 1:
        return None
    aspect_ratio = max(axis_a, axis_b) / min(axis_a, axis_b)
    if not (1.0 <= aspect_ratio <= 2.3):
        return None

    mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    ellipse_mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(ellipse_mask, ((int(cx), int(cy)), (int(axis_a), int(axis_b)), angle),
                255, thickness=cv2.FILLED)
    ellipse_iou = mask_iou(mask, ellipse_mask)

    # Hard filters: pure geometry only (these never caused regressions).
    if solidity < 0.78:
        return None
    if circularity < 0.40:
        return None
    if ellipse_iou < 0.48:
        return None

    # Soft cues: brightness/centrality only ADD to the score, never reject.
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    ys, xs = np.where(mask > 0)
    v_values = hsv[:, :, 2][ys, xs].astype(np.float32)
    mean_v, std_v = float(v_values.mean()), float(v_values.std())
    rel_brightness = mean_v - scene_mean_v
    brightness_score = max(0.0, min(1.0, (rel_brightness + 30) / 90))
    uniformity_score = max(0.0, 1.0 - std_v / 60.0)

    center = np.array([w / 2, h / 2])
    dist = np.linalg.norm(np.array([cx, cy]) - center)
    max_dist = np.sqrt(w * w + h * h) / 2
    center_score = max(0.0, 1.0 - dist / max_dist)

    hole_bonus = 0.05 if is_hole else 0.0

    score = (0.30 * ellipse_iou + 0.25 * solidity + 0.15 * circularity
             + 0.10 * center_score + 0.10 * brightness_score
             + 0.10 * uniformity_score + hole_bonus)

    return {
        "mask": mask, "score": score, "area": area,
        "solidity": solidity, "circularity": circularity,
        "aspect_ratio": aspect_ratio, "ellipse_iou": ellipse_iou,
        "mean_v": mean_v, "rel_brightness": rel_brightness, "std_v": std_v,
        "is_hole": is_hole,
        "ellipse": ((int(cx), int(cy)), (int(axis_a), int(axis_b)), float(angle)),
    }


def detect(image_bgr, top_k=2, min_rel_area=0.35):
    region, otsu_dbg, edges_dbg = combined_region(image_bgr)

    # CCOMP -> outer contours AND holes, so the egg can be found even if
    # it's enclosed inside a larger merged blob.
    contours, hierarchy = cv2.findContours(region, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0] if hierarchy is not None else []

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    scene_mean_v = float(hsv[:, :, 2].mean())

    candidates = []
    for i, contour in enumerate(contours):
        is_hole = hierarchy[i][3] != -1 if len(hierarchy) else False
        cand = contour_to_candidate(contour, image_bgr, is_hole, scene_mean_v)
        if cand is not None:
            candidates.append(cand)

    candidates.sort(key=lambda x: x["score"], reverse=True)

    kept, top_area = [], None
    for c in candidates:
        if all(mask_iou(c["mask"], k["mask"]) < 0.65 for k in kept):
            if top_area is None:
                top_area = c["area"]
            elif c["area"] < min_rel_area * top_area:
                continue
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
        tag = "HOLE" if c["is_hole"] else "OUTER"
        cv2.putText(overlay, f"{i}:{tag} {c['score']:.2f}", (cx, max(18, cy - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    return overlay


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    out_dir = here / "task2_trial5_combined_holes"
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

        print(f"{p.name}: {len(candidates)} candidate(s)"
              + (f", best={candidates[0]['score']:.3f} ({'HOLE' if candidates[0]['is_hole'] else 'OUTER'})"
                 if candidates else ""))

    print(f"\nWrote outputs to {out_dir}")