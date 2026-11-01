import cv2
import numpy as np

from task4_step2_detector import detect_eggs
from sort_tracker import Sort, KalmanBoxTracker

INPUT_VIDEO = "task4_trimmed.mp4"
OUTPUT_VIDEO = "task4_output.mp4"

cap = cv2.VideoCapture(INPUT_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Input: {INPUT_VIDEO} | {w}x{h} @ {fps}fps | {total_frames} frames")

out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

tracker = Sort(max_age=15, min_hits=2, iou_threshold=0.25, lost_buffer_age=120)

np.random.seed(42)
colors = {}  # track id -> color

frame_idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break

    detections = detect_eggs(frame)
    tracked = tracker.update(detections)

    for obj in tracked:
        tid = obj['id']
        if tid not in colors:
            colors[tid] = tuple(int(c) for c in np.random.randint(60, 255, 3))
        color = colors[tid]

        x1, y1, x2, y2 = map(int, obj['bbox'])
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w), min(y2, h)

        if obj['mask'] is not None:
            mask = obj['mask'].astype(bool)
            overlay = frame.copy()
            overlay[mask] = color
            frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame, f"Egg {tid}", (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
        )

    out.write(frame)
    frame_idx += 1
    if frame_idx % 30 == 0:
        print(f"Processed {frame_idx}/{total_frames} frames")

cap.release()
out.release()
print(f"\nDone. Output written to: {OUTPUT_VIDEO}")
print("Total IDs created:", KalmanBoxTracker.count, "(ideal = 2)")