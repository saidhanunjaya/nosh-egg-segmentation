"""
Task 2 - Trial 1: plain Otsu threshold, no cleanup.
Just to see what a naive classical baseline captures (egg + hand + floor?).
"""
import cv2
import numpy as np
from pathlib import Path

from task2_step0_split import load_or_build_split

def otsu_mask(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Otsu doesn't know which side is foreground; assume foreground is the
    # smaller-area class (egg+hand usually smaller than background floor/wall).
    if (mask > 0).mean() > 0.5:
        mask = cv2.bitwise_not(mask)
    return mask

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    out_dir = here / "task2_trial1_otsu"
    out_dir.mkdir(exist_ok=True)

    paths, split = load_or_build_split(
        here, ["train", "valid", "test"],
        here / "task1_outputs_final" / "split.csv")

    dev_paths = [p for p in paths if split[p] == "dev"]

    # Just run on a handful first to eyeball results quickly.
    sample = dev_paths[:8]
    for p in sample:
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        mask = otsu_mask(bgr)

        overlay = bgr.copy()
        overlay[mask > 0] = (0.5 * overlay[mask > 0] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)

        stem = p.stem
        cv2.imwrite(str(out_dir / f"{stem}_mask.png"), mask)
        cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), overlay)

    print(f"Wrote {len(sample)} mask/overlay pairs to {out_dir}")