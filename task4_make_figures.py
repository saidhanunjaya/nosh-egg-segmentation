import cv2
from pathlib import Path

PLAIN = "task4_final_plain.mp4"
FINAL = "task4_final.mp4"
TIMES = [11.5, 12.5, 21.5]          # seconds in the trimmed clip
OUT = Path("task4_figures")
OUT.mkdir(exist_ok=True)


def grab(path, t):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read {path} at {t}s")
    return cv2.resize(frame, (640, 360))


for t in TIMES:
    a, b = grab(PLAIN, t), grab(FINAL, t)
    cv2.putText(a, "Detector + tracker only", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(b, "Final (with CSRT)", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 160, 0), 2)
    out = OUT / f"compare_{t}s.png"
    cv2.imwrite(str(out), cv2.hconcat([a, b]))
    print("wrote", out)