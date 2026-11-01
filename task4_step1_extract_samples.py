import cv2
import os

VIDEO_PATH = "task4_input.mp4"
OUT_DIR = "task4_samples"
os.makedirs(OUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames: {total}, FPS: {fps}")

# Dense sampling: one frame per second from 10s to 45s
start_sec, end_sec = 10, 45
for sec in range(start_sec, end_sec + 1):
    idx = int(sec * fps)
    if idx >= total:
        continue
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if ok:
        out_path = os.path.join(OUT_DIR, f"t{sec:02d}s_frame{idx}.png")
        cv2.imwrite(out_path, frame)
        print(f"Saved {out_path}")

cap.release()
print("Done.")