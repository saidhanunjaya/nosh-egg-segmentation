"""
Task 2 - Classical egg segmentation (no deep learning).

Pipeline: GrabCut (soft trimap seed) -> skin-color removal -> morphological
opening (sever background bleed necks) -> pick connected component closest
to image center.

This mirrors task1_segment_eggs.py's structure (same split.csv reuse, same
output folder layout, same results.csv / review_list.txt pattern) so Task 3
can directly compare per-image outputs between the two pipelines.

Final configuration locked after 9 documented trials (see report experiment
log). Known limitations, accepted rather than further chased:
  - Low-contrast / textured backgrounds (marble, ash/rock, brushed steel)
    cause partial background bleed.
  - Rare cases where an unrelated high-contrast object in frame is captured
    instead of the true target.

Run:
    python task2_segment_eggs.py
    python task2_segment_eggs.py --max-images 5 --out task2_debug
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from task2_step0_split import load_or_build_split


# ==================================================
# 1. ARGUMENTS
# ==================================================

def parse_args():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=here)
    p.add_argument("--subsets", nargs="+", default=["train", "valid", "test"])
    p.add_argument("--split-csv", type=Path,
                   default=here / "task1_outputs_final" / "split.csv")
    p.add_argument("--out", type=Path, default=here / "task2_outputs_final")
    p.add_argument("--max-images", type=int, default=0, help="0 = all images")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--only", nargs="+", default=[])

    # GrabCut trimap
    p.add_argument("--sure-bg-frac", type=float, default=0.05)
    p.add_argument("--prob-bg-frac", type=float, default=0.20)
    p.add_argument("--retry-sure-bg-frac", type=float, default=0.08)
    p.add_argument("--retry-prob-bg-frac", type=float, default=0.28)
    p.add_argument("--bleed-frac-thresh", type=float, default=0.55)
    p.add_argument("--iterations", type=int, default=6)

    # skin removal
    p.add_argument("--min-saturation", type=int, default=40)

    # neck-breaking + component selection
    p.add_argument("--open-frac", type=float, default=0.02)
    p.add_argument("--min-area-frac", type=float, default=0.01)

    # flagging for review
    p.add_argument("--review-frac-thresh", type=float, default=0.45,
                    help="Flag images whose final mask covers more than this "
                         "fraction of the frame (likely background bleed)")
    return p.parse_args()


# ==================================================
# 2. GRABCUT CORE
# ==================================================

def build_soft_trimap(h, w, sure_bg_frac, prob_bg_frac):
    mask = np.full((h, w), cv2.GC_BGD, np.uint8)

    def rect_from_frac(f):
        mx, my = int(w * f), int(h * f)
        return mx, my, w - mx, h - my

    x, y, rw, rh = rect_from_frac(sure_bg_frac)
    mask[y:y + rh, x:x + rw] = cv2.GC_PR_BGD

    x, y, rw, rh = rect_from_frac(prob_bg_frac)
    mask[y:y + rh, x:x + rw] = cv2.GC_PR_FGD

    return mask


def run_grabcut_soft(image_bgr, sure_bg_frac, prob_bg_frac, iterations):
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


def skin_mask_ycrcb(image_bgr, min_saturation):
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


def break_thin_necks(mask, image_diag, open_frac):
    k = max(3, int(image_diag * open_frac))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = mask.astype(np.uint8)
    opened = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
    return opened > 0


def pick_central_component(mask, min_area_frac):
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


def segment(image_bgr, args):
    h, w = image_bgr.shape[:2]
    total = h * w
    diag = (h ** 2 + w ** 2) ** 0.5

    fg = run_grabcut_soft(image_bgr, args.sure_bg_frac, args.prob_bg_frac,
                           args.iterations)
    retried = False
    frac = fg.sum() / total
    if frac > args.bleed_frac_thresh:
        fg2 = run_grabcut_soft(image_bgr, args.retry_sure_bg_frac,
                                args.retry_prob_bg_frac, args.iterations)
        retried = True
        if fg2.sum() / total < frac:
            fg = fg2

    skin = skin_mask_ycrcb(image_bgr, args.min_saturation)
    fg = fg & (~skin)

    fg = break_thin_necks(fg, diag, args.open_frac)

    cleaned = pick_central_component(fg, args.min_area_frac)
    final_frac = (cleaned.sum() / total) if cleaned is not None else 0.0
    return cleaned, retried, final_frac


# ==================================================
# 3. OUTPUT WRITING
# ==================================================

def save_outputs(out_dir, stem, bgr, mask):
    h, w = bgr.shape[:2]
    combined = np.zeros((h, w), np.uint8)
    labels = np.zeros((h, w), np.uint16)
    overlay = bgr.copy()

    if mask is not None:
        combined[mask] = 255
        labels[mask] = 1
        overlay[mask] = (0.45 * overlay[mask]
                         + 0.55 * np.array([0, 0, 255], np.float32)).astype(np.uint8)
        cv2.imwrite(str(out_dir / "instances" / f"{stem}_egg1.png"),
                    mask.astype(np.uint8) * 255)

    cv2.imwrite(str(out_dir / "masks" / f"{stem}_mask.png"), combined)
    cv2.imwrite(str(out_dir / "labels" / f"{stem}_labels.png"), labels)
    cv2.imwrite(str(out_dir / "overlays" / f"{stem}_overlay.png"), overlay)


# ==================================================
# 4. MAIN
# ==================================================

def main():
    args = parse_args()
    out_dir = args.out.resolve()
    for sub in ["masks", "labels", "instances", "overlays"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    paths, split = load_or_build_split(
        args.data_root, args.subsets, args.split_csv)
    print(f"Found {len(paths)} images.")
    n_eval = sum(v == "eval" for v in split.values())
    print(f"dev={len(paths) - n_eval} images, eval={n_eval} images")

    if args.only:
        paths = [p for p in paths if any(s in p.name for s in args.only)]
        print(f"--only matched {len(paths)} image(s)")
    if args.max_images:
        paths = paths[:args.max_images]

    rows, review = [], []
    results_csv = out_dir / "results.csv"
    if results_csv.exists() and not args.overwrite:
        results_csv.unlink()  # start fresh unless appending is desired

    for idx, path in enumerate(paths, start=1):
        stem = path.stem
        print(f"[{idx}/{len(paths)}] {path.name}")

        if not args.overwrite and (out_dir / "masks" / f"{stem}_mask.png").exists():
            print("  already done, skipping (use --overwrite to redo)")
            continue

        bgr = cv2.imread(str(path))
        if bgr is None:
            print("  WARNING: unable to read image")
            review.append((path.name, "unreadable"))
            continue

        mask, retried, final_frac = segment(bgr, args)
        save_outputs(out_dir, stem, bgr, mask)

        area = int(mask.sum()) if mask is not None else 0
        flagged = mask is None or final_frac > args.review_frac_thresh
        reason = ("no mask found" if mask is None
                   else f"large area frac {final_frac:.2f}" if flagged else "")
        if flagged:
            review.append((path.name, reason))

        print(f"  area={area}  frame_frac={final_frac:.3f}  retried={retried}"
              + (f"  [FLAGGED: {reason}]" if flagged else ""))

        rows.append({
            "image": path.name,
            "stem": stem,
            "split": split.get(path, ""),
            "area": area,
            "frame_frac": round(final_frac, 4),
            "retried": retried,
            "flagged": flagged,
            "flag_reason": reason,
        })

        new_file = not results_csv.exists()
        with open(results_csv, "a", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[-1].keys()))
            if new_file:
                wr.writeheader()
            wr.writerow(rows[-1])

    if review:
        with open(out_dir / "review_list.txt", "w") as f:
            for name, reason in review:
                f.write(f"{name}\t{reason}\n")
        print(f"\n{len(review)} image(s) need a visual check -> review_list.txt")

    print(f"\nFinished. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()