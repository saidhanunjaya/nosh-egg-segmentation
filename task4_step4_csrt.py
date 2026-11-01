import cv2
import numpy as np

from task4_step2_detector import detect_eggs
from sort_tracker import Sort

INPUT_VIDEO = "task4_trimmed.mp4"
OUTPUT_VIDEO = "task4_output_csrt.mp4"

MAX_COAST_FRAMES = 120          # give up on a visual track after 5 s
AREA_RATIO_RANGE = (0.4, 2.5)   # reject if box size changes too much (drift)
OVERLAP_KILL_IOU = 0.30         # kill if it lands on a detected egg


def make_csrt():
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    raise RuntimeError("CSRT not found. Run: pip install opencv-contrib-python")


def iou(a, b):
    xx1, yy1 = max(a[0], b[0]), max(a[1], b[1])
    xx2, yy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-6)


cap = cv2.VideoCapture(INPUT_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

sort = Sort(max_age=15, min_hits=2, iou_threshold=0.25, lost_buffer_age=120)

np.random.seed(42)
colors = {}
last_seen = {}     # id -> last detected bbox
csrt = {}          # id -> {"trk", "area", "frames"}
gave_up = set()    # ids whose visual track failed; cleared on next real detection
prev_clean = None  # previous UNANNOTATED frame (used to init CSRT)
n_detected = n_coasted = 0

frame_idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    clean = frame.copy()                       # never draw on this one

    tracked = sort.update(detect_eggs(clean))
    detected_ids = {o["id"] for o in tracked}
    detected_boxes = [[float(v) for v in o["bbox"]] for o in tracked]

    # 1) real detections: solid box + mask
    for o in tracked:
        tid = o["id"]
        colors.setdefault(tid, tuple(int(c) for c in np.random.randint(60, 255, 3)))
        last_seen[tid] = [float(v) for v in o["bbox"]]
        csrt.pop(tid, None)
        gave_up.discard(tid)
        n_detected += 1

        if o["mask"] is not None:
            m = o["mask"].astype(bool)
            ov = frame.copy()
            ov[m] = colors[tid]
            frame = cv2.addWeighted(ov, 0.45, frame, 0.55, 0)
        x1, y1, x2, y2 = [int(v) for v in o["bbox"]]
        cv2.rectangle(frame, (max(x1, 0), max(y1, 0)), (min(x2, W), min(y2, H)), colors[tid], 2)
        cv2.putText(frame, f"Egg {tid}", (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[tid], 2, cv2.LINE_AA)

    # 2) ids with no detection now -> follow with CSRT
    for tid, box in list(last_seen.items()):
        if tid in detected_ids or tid in gave_up:
            continue

        if tid not in csrt:
            if prev_clean is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in box]
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, W - 1), min(y2, H - 1)
            if x2 - x1 < 20 or y2 - y1 < 20:
                gave_up.add(tid)
                continue
            trk = make_csrt()
            trk.init(prev_clean, (x1, y1, x2 - x1, y2 - y1))   # last frame it was detected
            csrt[tid] = {"trk": trk, "area": (x2 - x1) * (y2 - y1), "frames": 0}

        info = csrt[tid]
        ok_t, bb = info["trk"].update(clean)
        info["frames"] += 1
        bx, by, bw, bh = bb
        est = [bx, by, bx + bw, by + bh]
        ratio = (bw * bh) / max(info["area"], 1)
        cx, cy = bx + bw / 2, by + bh / 2

        bad = (not ok_t
               or info["frames"] > MAX_COAST_FRAMES
               or not (AREA_RATIO_RANGE[0] <= ratio <= AREA_RATIO_RANGE[1])
               or not (0 <= cx < W and 0 <= cy < H)
               or any(iou(est, db) > OVERLAP_KILL_IOU for db in detected_boxes))
        if bad:
            csrt.pop(tid, None)
            gave_up.add(tid)
            continue

        n_coasted += 1
        c = colors.get(tid, (0, 165, 255))
        cv2.rectangle(frame, (int(bx), int(by)), (int(bx + bw), int(by + bh)), c, 2)
        cv2.putText(frame, f"Egg {tid} (est.)", (int(bx), max(int(by) - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)

    out.write(frame)
    prev_clean = clean
    frame_idx += 1
    if frame_idx % 60 == 0:
        print(f"Processed {frame_idx}/{total}")

cap.release()
out.release()
print(f"Done -> {OUTPUT_VIDEO}")
print(f"Egg-frames with detected boxes: {n_detected} | filled by CSRT: {n_coasted}")