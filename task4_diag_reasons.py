import sys
import cv2
import numpy as np
from task4_step2_detector import TABLE_TOP_Y, MIN_SIDE_PX

VIDEO = "task4_trimmed.mp4"
t_sec = float(sys.argv[1]) if len(sys.argv) > 1 else 11.0

cap = cv2.VideoCapture(VIDEO)
cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000)
ok, frame = cap.read()
cap.release()
if not ok:
    raise SystemExit("Could not read frame at that time")

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
cv2.imwrite("diag_thresh.png", thresh)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
vis = frame.copy()
cv2.line(vis, (0, TABLE_TOP_Y), (frame.shape[1], TABLE_TOP_Y), (255, 0, 0), 2)

print(f"--- t = {t_sec}s ---")
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 500:
        continue
    x, y, bw, bh = cv2.boundingRect(cnt)
    per = cv2.arcLength(cnt, True)
    circ = 4 * np.pi * area / (per ** 2) if per > 0 else 0
    ar = max(bw, bh) / max(min(bw, bh), 1)

    m = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(m, [cnt], -1, 255, -1)
    sat = hsv[..., 1][m.astype(bool)].mean()

    reasons = []
    if y + bh / 2 < TABLE_TOP_Y: reasons.append("above TABLE_TOP_Y")
    if area < 3000 or area > 50000: reasons.append(f"area={area:.0f}")
    if bw < MIN_SIDE_PX or bh < MIN_SIDE_PX: reasons.append(f"size={bw}x{bh}")
    if circ < 0.60: reasons.append(f"circ={circ:.2f}")
    if ar > 1.8: reasons.append(f"aspect={ar:.2f}")
    if sat > 55: reasons.append(f"sat={sat:.0f}")

    ok_ = not reasons
    color = (0, 255, 0) if ok_ else (0, 0, 255)
    label = "PASS" if ok_ else ", ".join(reasons)
    cv2.rectangle(vis, (x, y), (x + bw, y + bh), color, 2)
    cv2.putText(vis, label, (x, max(y - 6, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    print(f"box=({x},{y},{bw},{bh}) area={area:.0f} circ={circ:.2f} "
          f"aspect={ar:.2f} sat={sat:.0f} -> {label}")

cv2.imwrite("diag_reasons.png", vis)
print("Saved diag_thresh.png and diag_reasons.png")