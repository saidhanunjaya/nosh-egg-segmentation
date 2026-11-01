import cv2
import numpy as np

from task4_step2_detector import detect_eggs
from sort_tracker import Sort, iou_batch

cap = cv2.VideoCapture("task4_trimmed.mp4")

tracker = Sort(max_age=15, min_hits=2, iou_threshold=0.25, lost_buffer_age=120)

frame_idx = 0
prev_ids = set()

while True:
    ok, frame = cap.read()
    if not ok:
        break

    detections = detect_eggs(frame)
    det_boxes = np.array([d['bbox'] for d in detections]) if detections else np.empty((0, 4))

    # Predicted boxes BEFORE update, with their current ids/ages
    pre_info = [(t.id, t.time_since_update, t.predict()) for t in tracker.trackers]

    tracked = tracker.update(detections)

    current_ids = set(obj['id'] for obj in tracked)
    new_ids = current_ids - prev_ids

    if new_ids:
        print(f"\n--- Frame {frame_idx}: NEW ID(S) {new_ids} spawned ---")
        print(f"Detections this frame: {[list(map(int, b)) for b in det_boxes]}")
        if len(pre_info) > 0:
            for tid, tsu, pbox in pre_info:
                if len(det_boxes) > 0:
                    ious = iou_batch(det_boxes, np.array([pbox]))
                    print(f"  existing track id={tid} time_since_update={tsu} "
                          f"predicted_box={list(map(int, pbox))} iou_vs_dets={ious.flatten()}")
                else:
                    print(f"  existing track id={tid} time_since_update={tsu} predicted_box={list(map(int, pbox))}")
        else:
            print("  (no existing tracks at this point)")

    prev_ids = current_ids
    frame_idx += 1

cap.release()
print("\nDone.")