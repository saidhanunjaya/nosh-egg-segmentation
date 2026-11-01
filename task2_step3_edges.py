"""
Task 2 - Trial 3e: replace Trial 3d's absolute brightness threshold with
two relative cues.

Diagnosis from Trial 3d:
  1. Regression: the shadowed ash-background egg fell below the absolute
     V>=95 cutoff and was wrongly rejected (0 candidates).
  2. Still broken: a glossy keyboard key has ONE bright specular highlight
     that pushed its mean V to 104, above threshold, so it still beat the
     egg. Mean brightness alone can't tell "uniformly lit matte egg" from
     "mostly dark key with a glint".

Fix: drop the absolute threshold. Instead score each candidate by
  (a) how much brighter it is than the rest of THAT photo (relative, not
      a fixed number -- handles shadowed scenes), and
  (b) how UNIFORM its brightness is (low std-dev) -- an egg surface is
      smoothly lit, a key-with-glint is not.
"""

import cv2
import numpy as np
from pathlib import Path

from task2_step0_split import load_or_build_split


def mask_iou(a, b):
    a, b = a > 0, b > 0
    union = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / union if union else 0.0


def edge_map(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edges = cv2.Canny(blur, 35, 110)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    return edges, enhanced


def contour_to_candidate(contour, image_bgr, is_hole, scene_mean_v, scene_std_v,
                          min_rel_brightness=-10, max_uniformity_std=45):
    h, w = image_bgr.shape[:2]
    image_area = h * w

    area = cv2.contourArea(contour)
    if area < 0.003 * image_area or area > 0.50 * image_area:
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
    if not (1.0 <= aspect_ratio <= 2.2):
        return None

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    ellipse_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(ellipse_mask, ((int(cx), int(cy)), (int(axis_a), int(axis_b)), float(angle)),
                255, thickness=cv2.FILLED)
    ellipse_iou = mask_iou(mask, ellipse_mask)

    if solidity < 0.80:
        return None
    if circularity < 0.42:
        return None
    if ellipse_iou < 0.50:
        return None

    # --- Relative brightness + uniformity (replaces Trial 3d's absolute cutoff) ---
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    ys, xs = np.where(mask > 0)
    v_values = v_channel[ys, xs].astype(np.float32)
    mean_v = float(v_values.mean())
    std_v = float(v_values.std())

    rel_brightness = mean_v - scene_mean_v   # how much brighter than the photo overall
    if rel_brightness < min_rel_brightness:
        return None
    if std_v > max_uniformity_std:
        return None   # too much internal brightness variation -> glint, not a smooth egg surface

    image_center = np.array([w / 2, h / 2])
    distance = np.linalg.norm(np.array([cx, cy]) - image_center)
    max_distance = np.sqrt(w * w + h * h) / 2
    center_score = max(0.0, 1.0 - distance / max_distance)

    hole_bonus = 0.05 if is_hole else 0.0
    brightness_score = max(0.0, min(1.0, (rel_brightness + 20) / 80))
    uniformity_score = max(0.0, 1.0 - std_v / max_uniformity_std)

    score = (0.25 * ellipse_iou + 0.20 * solidity
             + 0.15 * circularity + 0.10 * center_score
             + 0.15 * brightness_score + 0.15 * uniformity_score + hole_bonus)

    return {
        "mask": mask, "score": score, "area": area,
        "solidity": solidity, "circularity": circularity,
        "aspect_ratio": aspect_ratio, "ellipse_iou": ellipse_iou,
        "mean_v": mean_v, "rel_brightness": rel_brightness, "std_v": std_v,
        "is_hole": is_hole,
        "ellipse": ((int(cx), int(cy)), (int(axis_a), int(axis_b)), float(angle)),
    }


def detect_egg_edges(image_bgr, top_k=2, min_rel_area=0.35):
    edges, enhanced = edge_map(image_bgr)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0] if hierarchy is not None else []

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    scene_mean_v = float(hsv[:, :, 2].mean())
    scene_std_v = float(hsv[:, :, 2].std())

    candidates = []
    for i, contour in enumerate(contours):
        is_hole = hierarchy[i][3] != -1 if len(hierarchy) else False
        candidate = contour_to_candidate(contour, image_bgr, is_hole, scene_mean_v, scene_std_v)
        if candidate is not None:
            candidates.append(candidate)

    candidates.sort(key=lambda x: x["score"], reverse=True)

    kept = []
    top_area = None
    for candidate in candidates:
        if all(mask_iou(candidate["mask"], old["mask"]) < 0.65 for old in kept):
            if top_area is None:
                top_area = candidate["area"]
            elif candidate["area"] < min_rel_area * top_area:
                continue
            kept.append(candidate)
        if len(kept) >= top_k:
            break

    result = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    for candidate in kept:
        result[candidate["mask"] > 0] = 255

    return result, edges, enhanced, kept


def draw_overlay(image_bgr, mask, candidates):
    overlay = image_bgr.copy()
    region = mask > 0
    overlay[region] = (0.50 * overlay[region] + 0.50 * np.array([0, 0, 255])).astype(np.uint8)
    for index, candidate in enumerate(candidates, start=1):
        cv2.ellipse(overlay, candidate["ellipse"], (0, 255, 0), 2)
        center = candidate["ellipse"][0]
        tag = "HOLE" if candidate["is_hole"] else "OUTER"
        text = (f"{index}:{tag} {candidate['score']:.2f} "
                f"relV={candidate['rel_brightness']:.0f} std={candidate['std_v']:.0f}")
        cv2.putText(overlay, text, (center[0], max(18, center[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
    return overlay


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    out_dir = here / "task2_trial3e_relbrightness"
    out_dir.mkdir(exist_ok=True)

    paths, split = load_or_build_split(
        here, ["train", "valid", "test"],
        here / "task1_outputs_final" / "split.csv")
    dev_paths = [p for p in paths if split[p] == "dev"]
    sample = dev_paths[:8]

    processed = 0
    for p in sample:
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue

        mask, edges, enhanced, candidates = detect_egg_edges(bgr, top_k=2)
        overlay = draw_overlay(bgr, mask, candidates)

        stem = p.stem
        cv2.imwrite(str(out_dir / f"{stem}_edges.png"), edges)
        cv2.imwrite(str(out_dir / f"{stem}_mask.png"), mask)
        cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), overlay)

        print(f"{p.name}: {len(candidates)} candidate(s)"
              + (f", best={candidates[0]['score']:.3f} relV={candidates[0]['rel_brightness']:.0f} "
                 f"std={candidates[0]['std_v']:.0f}" if candidates else ""))
        processed += 1

    print(f"\nWrote outputs for {processed} image(s) to:\n{out_dir}")