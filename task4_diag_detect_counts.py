import cv2
from task4_step2_detector import detect_eggs

cap = cv2.VideoCapture("task4_trimmed.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)

frame_idx = 0
counts = []
zero_runs = []
current_zero_run = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    dets = detect_eggs(frame)
    n = len(dets)
    counts.append(n)

    if n == 0:
        current_zero_run += 1
    else:
        if current_zero_run > 0:
            zero_runs.append((frame_idx - current_zero_run, frame_idx - 1, current_zero_run))
        current_zero_run = 0

    frame_idx += 1

cap.release()

from collections import Counter
print("Detection count histogram (num detections -> num frames):")
print(Counter(counts))

print(f"\nTotal frames: {frame_idx}")
print(f"Frames with 0 detections: {counts.count(0)}")
print(f"Frames with 1 detection:  {counts.count(1)}")
print(f"Frames with 2 detections: {counts.count(2)}")
print(f"Frames with 3+ detections: {sum(1 for c in counts if c >= 3)}")

print(f"\nLongest zero-detection runs (start_frame, end_frame, length):")
zero_runs.sort(key=lambda x: -x[2])
for run in zero_runs[:15]:
    print(run, f"-> {run[2]/fps:.2f}s")