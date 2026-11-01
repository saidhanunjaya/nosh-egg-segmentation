"""
Task 1 - Egg segmentation with a pretrained model.

Pipeline:  OWL-ViT (text-prompted detector: "egg")  ->  SAM (box-prompted mask)

Why: SAM's automatic mask generator segments everything and has no idea what
an egg is. Giving SAM a box from a text-prompted detector removes the need for
hand-written shape heuristics and handles any number of eggs per image.

Also produces the Task 2 dev/eval split (4:1), grouped by ORIGINAL photo so the
Roboflow augmented copies (name_jpg.rf.<hash>.jpg) never straddle both splits.

Install:
    pip install torch torchvision opencv-python numpy pillow transformers
    pip install git+https://github.com/facebookresearch/segment-anything.git

Run (from the dataset folder that contains train/ test/ valid/ and the .pth):
    python task1_segment_eggs.py
    python task1_segment_eggs.py --max-images 5 --out task1_debug
"""

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms
from transformers import OwlViTForObjectDetection, OwlViTProcessor
from segment_anything import SamPredictor, sam_model_registry

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COLORS = [
    (255, 40, 40), (40, 220, 40), (40, 100, 255),
    (255, 200, 0), (200, 40, 255), (0, 220, 220),
]


# ==================================================
# 1. ARGUMENTS
# ==================================================

def parse_args():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=here,
                   help="Folder containing train/ valid/ test/ subfolders")
    p.add_argument("--subsets", nargs="+", default=["train", "valid", "test"],
                   help="Subfolders of data-root to scan (missing ones are skipped)")
    p.add_argument("--checkpoint", type=Path, default=here / "sam_vit_b_01ec64.pth")
    p.add_argument("--out", type=Path, default=here / "task1_outputs_final")
    p.add_argument("--max-images", type=int, default=0, help="0 = all images")
    p.add_argument("--det-thresh", type=float, default=0.15,
                   help="OWL-ViT score threshold (first attempt)")
    p.add_argument("--fallback-thresh", type=float, default=0.07,
                   help="Used if nothing is found at det-thresh")
    p.add_argument("--rel-thresh", type=float, default=0.5,
                   help="Keep detections scoring >= rel-thresh * best score")
    p.add_argument("--nms-iou", type=float, default=0.3)
    p.add_argument("--max-area-ratio", type=float, default=4.0,
                   help="With 2+ detections, drop masks larger than this many "
                        "times the area of the highest-scoring detection")
    p.add_argument("--min-sam-score", type=float, default=0.90,
                   help="Drop masks whose SAM predicted-IoU is below this")
    p.add_argument("--eval-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true",
                   help="Re-process images that already have outputs")
    p.add_argument("--only", nargs="+", default=[],
                   help="Only process images whose filename contains one of these "
                        "strings, e.g. --only not_damaged_116_ not_damaged_117_")
    p.add_argument("--manual-boxes", type=Path, default=None,
                   help="JSON file {\"not_damaged_116_\": [[x1,y1,x2,y2], ...]} "
                        "with boxes to prompt SAM with (overrides detection)")
    return p.parse_args()


# ==================================================
# 2. DATASET DISCOVERY AND GROUPED SPLIT
# ==================================================

def base_name(stem: str) -> str:
    """
    Roboflow exports augmented copies as  <orig>_jpg.rf.<hash>.jpg
    Everything before '_jpg.rf.' identifies the original photo.
    """
    m = re.split(r"[_.](?:jpg|jpeg|png)\.rf\.", stem, maxsplit=1)
    if len(m) > 1:
        return m[0]
    return stem.split(".rf.")[0]


def find_images(data_root: Path, subsets, out_dir: Path):
    paths = []
    for sub in subsets:
        d = data_root / sub
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if p.suffix.lower() in IMAGE_EXTS and out_dir not in p.parents:
                paths.append(p)
    return sorted(paths)


def grouped_split(paths, eval_frac, seed):
    """
    Split by group (original photo), not by file, so augmented copies of the
    same photo stay together. Groups are keyed by (class folder, base name).
    Returns {path: 'dev' | 'eval'}.
    """
    groups = defaultdict(list)
    for p in paths:
        groups[(p.parent.name, base_name(p.stem))].append(p)

    keys = sorted(groups)
    random.Random(seed).shuffle(keys)

    target_eval = round(len(paths) * eval_frac)
    assignment, n_eval = {}, 0
    for k in keys:
        if n_eval < target_eval:
            for p in groups[k]:
                assignment[p] = "eval"
            n_eval += len(groups[k])
        else:
            for p in groups[k]:
                assignment[p] = "dev"
    return assignment, len(groups)


# ==================================================
# 3. MODELS
# ==================================================

class EggSegmenter:
    def __init__(self, checkpoint: Path, device: str):
        if not checkpoint.exists():
            raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")
        self.device = device

        print(f"Loading OWL-ViT on {device}...")
        self.owl_proc = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.owl = (OwlViTForObjectDetection
                    .from_pretrained("google/owlvit-base-patch32")
                    .to(device).eval())

        print(f"Loading SAM ViT-B on {device}...")
        sam = sam_model_registry["vit_b"](checkpoint=str(checkpoint))
        sam.to(device=device).eval()
        self.predictor = SamPredictor(sam)

        self.queries = [["a photo of an egg", "an egg"]]

    def _post_process(self, outputs, thresh, sizes):
        """Works across transformers versions (method moved between classes)."""
        for owner in (self.owl_proc, getattr(self.owl_proc, "image_processor", None)):
            if owner is not None and hasattr(owner, "post_process_object_detection"):
                return owner.post_process_object_detection(
                    outputs, threshold=thresh, target_sizes=sizes)
        return self.owl_proc.post_process_grounded_object_detection(
            outputs, threshold=thresh, target_sizes=sizes)

    @torch.no_grad()
    def detect(self, image_rgb, thresh, nms_iou, rel_thresh, queries=None):
        pil = Image.fromarray(image_rgb)
        inputs = self.owl_proc(text=queries or self.queries, images=pil,
                               return_tensors="pt").to(self.device)
        # NOTE: inputs is a dict-like BatchEncoding (input_ids, attention_mask,
        # pixel_values, ...). It MUST be unpacked with ** so each field is
        # passed as its own keyword argument. Calling self.owl(inputs) without
        # ** passes the whole dict positionally and raises:
        #   TypeError: forward() missing 1 required positional argument: 'pixel_values'
        out = self.owl(**inputs)
        sizes = torch.tensor([pil.size[::-1]]).to(self.device)
        res = self._post_process(out, thresh, sizes)[0]

        boxes, scores = res["boxes"], res["scores"]
        if len(boxes) == 0:
            return np.zeros((0, 4)), np.zeros(0)

        # Drop boxes covering almost the whole frame (background, not an egg).
        h, w = image_rgb.shape[:2]
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        ok = areas < 0.9 * h * w
        boxes, scores = boxes[ok], scores[ok]
        if len(boxes) == 0:
            return np.zeros((0, 4)), np.zeros(0)

        # Class-agnostic NMS (two text queries give overlapping boxes).
        keep = nms(boxes, scores, nms_iou)
        boxes, scores = boxes[keep], scores[keep]

        # Drop weak detections relative to the best one.
        good = scores >= rel_thresh * scores.max()
        return boxes[good].cpu().numpy(), scores[good].cpu().numpy()

    def segment(self, image_rgb, args, image_name=""):
        """
        Returns (instances, source) where source says how the boxes were found:
        'auto', 'fallback-1', 'fallback-2', 'fallback-3', 'manual', or 'none'.
        """
        boxes, det_scores, source = np.zeros((0, 4)), np.zeros(0), "none"

        # 1) Manual boxes (from --manual-boxes) override everything.
        manual = next((v for k, v in getattr(self, "manual", {}).items()
                       if k in image_name), None)
        if manual:
            boxes = np.array(manual, dtype=np.float32).reshape(-1, 4)
            det_scores = np.ones(len(boxes))
            source = "manual"
        else:
            # 2) Automatic chain: each stage runs only if the previous found nothing.
            extra = [["a white egg", "a boiled egg in water", "an oval white object"]]
            stages = [
                ("auto",       args.det_thresh,      None),
                ("fallback-1", args.fallback_thresh, None),
                ("fallback-2", 0.03,                 None),
                ("fallback-3", 0.03,                 extra),
            ]
            for name, thresh, queries in stages:
                boxes, det_scores = self.detect(
                    image_rgb, thresh, args.nms_iou, args.rel_thresh, queries)
                if len(boxes):
                    source = name
                    break

        self.predictor.set_image(image_rgb)
        instances = []
        for box, ds in zip(boxes, det_scores):
            masks, sam_scores, _ = self.predictor.predict(
                box=box, multimask_output=False)
            mask = clean_mask(masks[0])
            if mask is None:
                continue
            instances.append({
                "mask": mask,
                "box": [float(v) for v in box],
                "det_score": float(ds),
                "sam_score": float(sam_scores[0]),
                "area": int(mask.sum()),
            })

        instances = dedupe_instances(instances)
        # Highest detection score first -> stable instance numbering.
        instances.sort(key=lambda d: d["det_score"], reverse=True)
        if source != "manual":
            instances = filter_instances(
                instances, args.max_area_ratio, args.min_sam_score)
        return instances, (source if instances else "none")


# ==================================================
# 4. MASK POST-PROCESSING
# ==================================================

def clean_mask(mask, min_area=80):
    """Keep the largest connected component and fill interior holes."""
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


def mask_iou(a, b):
    union = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / union if union else 0.0


def filter_instances(instances, max_area_ratio, min_sam_score):
    """
    Post-filter for automatic detections (instances sorted by det_score desc).
    Keep the top-scoring detection, then drop any other mask that is
      - much larger than the top mask (> max_area_ratio x its area), e.g. a lid
        or a pan full of contents mistaken for an egg, or
      - low quality according to SAM's own score (< min_sam_score).
    Eggs in one photo have similar size, so a big outlier is not an egg.
    """
    if not instances:
        return instances
    top = instances[0]
    kept = [top]
    for inst in instances[1:]:
        if inst["sam_score"] < min_sam_score:
            continue
        if inst["area"] > max_area_ratio * top["area"]:
            continue
        kept.append(inst)
    return kept


def dedupe_instances(instances, iou_thresh=0.7):
    instances = sorted(instances, key=lambda d: d["det_score"], reverse=True)
    kept = []
    for inst in instances:
        if all(mask_iou(inst["mask"], k["mask"]) <= iou_thresh for k in kept):
            kept.append(inst)
    return kept


# ==================================================
# 5. OUTPUT WRITING
# ==================================================

def save_outputs(out_dir: Path, stem: str, image_rgb, instances):
    # A re-run may find fewer eggs than before; clear old per-egg files.
    for old in (out_dir / "instances").glob(f"{stem}_egg*.png"):
        old.unlink()

    h, w = image_rgb.shape[:2]
    combined = np.zeros((h, w), np.uint8)
    labels = np.zeros((h, w), np.uint16)
    overlay = image_rgb.copy()
    diagnostic = image_rgb.copy()

    for i, inst in enumerate(instances, start=1):
        mask = inst["mask"]
        color = COLORS[(i - 1) % len(COLORS)]
        combined[mask] = 255
        labels[mask] = i

        overlay[mask] = (0.45 * overlay[mask]
                         + 0.55 * np.array(color, np.float32)).astype(np.uint8)

        x1, y1, x2, y2 = [int(round(v)) for v in inst["box"]]
        cv2.rectangle(diagnostic, (x1, y1), (x2, y2), color, 2)
        cv2.putText(diagnostic,
                    f"{i}: det={inst['det_score']:.2f} sam={inst['sam_score']:.2f}",
                    (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        cv2.imwrite(str(out_dir / "instances" / f"{stem}_egg{i}.png"),
                    mask.astype(np.uint8) * 255)

    cv2.imwrite(str(out_dir / "masks" / f"{stem}_mask.png"), combined)
    cv2.imwrite(str(out_dir / "labels" / f"{stem}_labels.png"), labels)
    cv2.imwrite(str(out_dir / "overlays" / f"{stem}_overlay.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / "diagnostics" / f"{stem}_diagnostic.png"),
                cv2.cvtColor(diagnostic, cv2.COLOR_RGB2BGR))


# ==================================================
# 6. MAIN
# ==================================================

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = args.out.resolve()
    for sub in ["masks", "labels", "instances", "overlays", "diagnostics"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    # ---- find images + split -------------------------------------------
    paths = find_images(args.data_root, args.subsets, out_dir)
    if not paths:
        raise FileNotFoundError(
            f"No images found under {args.data_root} in {args.subsets}")
    print(f"Found {len(paths)} images.")

    split, n_groups = grouped_split(paths, args.eval_frac, args.seed)
    n_eval = sum(v == "eval" for v in split.values())
    print(f"{n_groups} original-photo groups -> "
          f"dev={len(paths) - n_eval} images, eval={n_eval} images")

    with open(out_dir / "split.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["path", "class_folder", "group", "split"])
        for p in paths:
            wr.writerow([str(p), p.parent.name, base_name(p.stem), split[p]])

    # Sanity check: no group appears in both splits.
    seen = defaultdict(set)
    for p, s in split.items():
        seen[(p.parent.name, base_name(p.stem))].add(s)
    assert all(len(v) == 1 for v in seen.values()), "Leakage between splits!"

    if args.only:
        paths = [p for p in paths if any(s in p.name for s in args.only)]
        print(f"--only matched {len(paths)} image(s)")
    if args.max_images:
        paths = paths[:args.max_images]

    # ---- segment -------------------------------------------------------
    seg = EggSegmenter(args.checkpoint, device)
    seg.manual = {}
    if args.manual_boxes:
        seg.manual = json.loads(args.manual_boxes.read_text())
        print(f"Loaded manual boxes for {len(seg.manual)} image(s)")
    rows, review = [], []

    # Duplicate stems across subfolders would overwrite each other; prefix them.
    stem_counts = defaultdict(int)
    for p in paths:
        stem_counts[p.stem] += 1

    for idx, path in enumerate(paths, start=1):
        stem = path.stem if stem_counts[path.stem] == 1 else f"{path.parent.name}_{path.stem}"
        print(f"[{idx}/{len(paths)}] {path.name}")

        if not args.overwrite and (out_dir / "masks" / f"{stem}_mask.png").exists():
            print("  already done, skipping (use --overwrite to redo)")
            continue

        bgr = cv2.imread(str(path))
        if bgr is None:
            print("  WARNING: unable to read image")
            review.append((path.name, "unreadable"))
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        instances, source = seg.segment(rgb, args, path.name)
        save_outputs(out_dir, stem, rgb, instances)

        n = len(instances)
        print(f"  eggs found: {n}" + ("" if source == "auto" else f" (source: {source})"))
        if n == 0:
            review.append((path.name, "no detections"))
        elif source != "auto":
            review.append((path.name, f"source: {source}"))

        rows.append({
            "image": path.name,
            "stem": stem,
            "split": split.get(path, ""),
            "num_eggs": n,
            "used_fallback": source,
            "areas": json.dumps([d["area"] for d in instances]),
            "det_scores": json.dumps([round(d["det_score"], 3) for d in instances]),
            "sam_scores": json.dumps([round(d["sam_score"], 3) for d in instances]),
        })

        # Append incrementally so a crash doesn't lose progress.
        results_csv = out_dir / "results.csv"
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