import cv2
from collections import defaultdict
from task4_step2_detector import detect_eggs
from sort_tracker import Sort, KalmanBoxTracker

cap = cv2.VideoCapture("task4_trimmed.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
tracker = Sort(max_age=15, min_hits=2, iou_threshold=0.25, lost_buffer_age=120)

frames_of = defaultdict(list)   # id -> frame indices where it was drawn
pos_of = defaultdict(list)      # id -> (cx, cy)
frame_idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    for obj in tracker.update(detect_eggs(frame)):
        x1, y1, x2, y2 = obj['bbox']
        frames_of[obj['id']].append(frame_idx)
        pos_of[obj['id']].append((int((x1 + x2) / 2), int((y1 + y2) / 2)))
    frame_idx += 1
cap.release()

print(f"Tracks born: {KalmanBoxTracker.count} | "
      f"Distinct IDs actually drawn: {len(frames_of)}\n")

for tid in sorted(frames_of):
    fr = frames_of[tid]
    # split into continuous segments
    segs, start, prev = [], fr[0], fr[0]
    for f in fr[1:]:
        if f != prev + 1:
            segs.append((start, prev))
            start = f
        prev = f
    segs.append((start, prev))
    print(f"ID {tid}: {len(fr)} frames, "
          f"first pos {pos_of[tid][0]}, last pos {pos_of[tid][-1]}")
    for s, e in segs:
        print(f"    frames {s}-{e}  ({s/fps:.1f}s - {e/fps:.1f}s)")