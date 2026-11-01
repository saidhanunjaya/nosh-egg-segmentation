import cv2
import numpy as np
import os

# The tabletop begins at approximately this y-coordinate.
# Ignore shirt/background above this static-camera table boundary.
TABLE_TOP_Y = 255

# Eggs in this video are consistently large (~150-180px wide/tall).
# Nail/skin highlights are much smaller -> use this to reject them.
MIN_SIDE_PX = 60


def detect_eggs(frame_bgr, min_area=3000, max_area=50000, sat_thresh=55):
    """
    Detect white eggs on the dark tabletop using classical CV.

    Steps:
      1. grayscale brightness threshold
      2. crop to tabletop ROI
      3. morphology to remove small noise
      4. contour filtering by area, min size, circularity, aspect ratio, saturation

    Returns:
      list of dictionaries:
      {
          'bbox': [x1, y1, x2, y2],
          'mask': boolean NumPy array,
          'score': circularity
      }
    """

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Eggs are white/bright; table is dark.
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Keep only the tabletop. This removes shirt/background detections.
    roi_mask = np.zeros_like(thresh)
    roi_mask[TABLE_TOP_Y:, :] = 255
    thresh = cv2.bitwise_and(thresh, roi_mask)

    # Remove small bright noise and smooth egg boundaries.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    detections = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Reject tiny bright spots/fingernail highlights and huge regions.
        if area < min_area or area > max_area:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        # Reject anything smaller than a real egg's known size on screen.
        # This is the key fix for nail/fingertip specular highlights,
        # which pass brightness/circularity/saturation checks but are
        # much smaller in absolute pixel size than an actual egg.
        if bw < MIN_SIDE_PX or bh < MIN_SIDE_PX:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue

        # Circle = 1.0. Eggs are oval, but still reasonably compact.
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < 0.60:
            continue

        # Reject thin or long shapes such as sleeve edges.
        aspect_ratio = max(bw, bh) / max(min(bw, bh), 1)
        if aspect_ratio > 1.8:
            continue

        # Build full-frame binary mask for this instance.
        mask_uint8 = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask_uint8, [cnt], -1, 255, thickness=-1)
        mask_bool = mask_uint8.astype(bool)

        # Eggs are near-white / low saturation.
        # Fingertips/skin tend to have higher saturation
        # (except specular highlights, which MIN_SIDE_PX catches instead).
        mean_saturation = hsv[..., 1][mask_bool].mean()
        if mean_saturation > sat_thresh:
            continue

        detections.append({
            "bbox": [x, y, x + bw, y + bh],
            "mask": mask_bool,
            "score": float(circularity),
        })

    return detections


if __name__ == "__main__":
    # Test detector on the sample frames generated in Step 1.
    TEST_DIR = "task4_samples"
    OUT_DIR = "task4_detect_debug"
    os.makedirs(OUT_DIR, exist_ok=True)

    for filename in sorted(os.listdir(TEST_DIR)):
        if not filename.lower().endswith(".png"):
            continue

        image_path = os.path.join(TEST_DIR, filename)
        frame = cv2.imread(image_path)

        if frame is None:
            print(f"Could not read: {image_path}")
            continue

        detections = detect_eggs(frame)

        # Draw detections for visual inspection.
        visualization = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            score = det["score"]

            cv2.rectangle(
                visualization,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                visualization,
                f"egg {score:.2f}",
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        output_path = os.path.join(OUT_DIR, filename)
        cv2.imwrite(output_path, visualization)

        print(f"{filename}: detected {len(detections)} egg(s)")

    print(f"\nDone. Check annotated outputs in: {OUT_DIR}")