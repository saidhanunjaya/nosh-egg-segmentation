# task2_step0_split.py
import csv, re, random
from pathlib import Path
from collections import defaultdict

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def base_name(stem):
    m = re.split(r"[_.](?:jpg|jpeg|png)\.rf\.", stem, maxsplit=1)
    return m[0] if len(m) > 1 else stem.split(".rf.")[0]

def find_images(data_root, subsets):
    paths = []
    for sub in subsets:
        d = data_root / sub
        if d.exists():
            paths += [p for p in d.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    return sorted(paths)

def load_or_build_split(data_root, subsets, split_csv, eval_frac=0.2, seed=42):
    paths = find_images(data_root, subsets)
    if split_csv.exists():
        lut = {}
        with open(split_csv, newline="") as f:
            for row in csv.DictReader(f):
                lut[(row["class_folder"], row["group"])] = row["split"]
        split = {p: lut.get((p.parent.name, base_name(p.stem))) for p in paths}
        if all(split.values()):
            print(f"Loaded split from {split_csv}")
            return paths, split
    # fallback: rebuild (shouldn't happen if Task 1 was run)
    groups = defaultdict(list)
    for p in paths:
        groups[(p.parent.name, base_name(p.stem))].append(p)
    keys = sorted(groups); random.Random(seed).shuffle(keys)
    target_eval = round(len(paths) * eval_frac)
    split, n_eval = {}, 0
    for k in keys:
        tag = "eval" if n_eval < target_eval else "dev"
        for p in groups[k]:
            split[p] = tag
        if tag == "eval":
            n_eval += len(groups[k])
    return paths, split

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    paths, split = load_or_build_split(
        here, ["train", "valid", "test"],
        here / "task1_outputs_final" / "split.csv")
    n_eval = sum(v == "eval" for v in split.values())
    print(f"total={len(paths)}  dev={len(paths)-n_eval}  eval={n_eval}")