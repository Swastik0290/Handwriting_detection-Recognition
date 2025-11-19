# detection.py
"""
Detection module:
- Uses EAST text detector if 'frozen_east_text_detection.pb' exists in the project root.
- Otherwise falls back to a morphological contour-based detector (robust for handwriting).
Saves crops to output/crops/ and visualization to output/vis/detections.png
Also writes output/crops_meta.txt listing crops and bounding boxes.
"""
import os
import cv2
import numpy as np
from utils import ensure_dir, sort_boxes_top_to_bottom_left_to_right

OUTPUT = "output"
CROPS_DIR = os.path.join(OUTPUT, "crops")
VIS_DIR = os.path.join(OUTPUT, "vis")
META_FILE = os.path.join(OUTPUT, "crops_meta.txt")

# EAST model filename (optional). If missing, fallback used.
EAST_MODEL = "frozen_east_text_detection.pb"

ensure_dir(CROPS_DIR)
ensure_dir(VIS_DIR)

def _decode_east(scores, geometry, scoreThresh):
    # Reference helper to decode east outputs
    detections = []
    confidences = []

    rows, cols = scores.shape[2:4]
    for y in range(rows):
        scoresData = scores[0,0,y]
        x0_data = geometry[0,0,y]
        x1_data = geometry[0,1,y]
        x2_data = geometry[0,2,y]
        x3_data = geometry[0,3,y]
        anglesData = geometry[0,4,y]
        for x in range(cols):
            if scoresData[x] < scoreThresh:
                continue
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]
            endX = int(x * 4.0 + (cos * x1_data[x]) + (sin * x2_data[x]))
            endY = int(y * 4.0 - (sin * x1_data[x]) + (cos * x2_data[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            detections.append((startX, startY, int(w), int(h)))
            confidences.append(float(scoresData[x]))
    return detections, confidences

def detect_with_east(image, min_confidence=0.5, width=640, height=640):
    net = cv2.dnn.readNet(EAST_MODEL)
    orig_h, orig_w = image.shape[:2]
    rW = orig_w / float(width)
    rH = orig_h / float(height)
    blob = cv2.dnn.blobFromImage(image, 1.0, (width, height), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(["feature_fusion/Conv_7/Sigmoid","feature_fusion/concat_3"])
    rects, confidences = _decode_east(scores, geometry, min_confidence)
    # scale rects back
    boxes = []
    for (x, y, w, h) in rects:
        x0 = int(max(0, x * rW))
        y0 = int(max(0, y * rH))
        x1 = int(min(orig_w, (x + w) * rW))
        y1 = int(min(orig_h, (y + h) * rH))
        boxes.append((x0, y0, x1, y1))
    # NMS
    if len(boxes):
        boxes_np = np.array(boxes)
        scores_np = np.array(confidences)
        picked = cv2.dnn.NMSBoxes(
            [ [b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in boxes ],
            confidences, min_confidence, 0.4
        )
        picked = [p[0] if isinstance(p, (list,tuple,np.ndarray)) else p for p in picked]
        boxes = [boxes[i] for i in picked] if len(picked) else []
    return boxes

def detect_with_morphology(image, min_area=300):
    """Fallback detector: threshold -> close -> contours"""
    orig = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # adaptive threshold
    th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,15,11)
    # close to connect letters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,3))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if w*h < min_area:
            continue
        pad_x = int(0.03 * w)
        pad_y = int(0.12 * h)
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(image.shape[1], x + w + pad_x)
        y1 = min(image.shape[0], y + h + pad_y)
        boxes.append((x0,y0,x1,y1))
    return boxes

def detect_and_crop(image_path, save_crops=True, visualize=True):
    assert os.path.exists(image_path), "Image path not found: " + image_path
    img = cv2.imread(image_path)
    orig_vis = img.copy()
    # Choose detector
    boxes = []
    if os.path.exists(EAST_MODEL):
        try:
            boxes = detect_with_east(img)
            if not boxes:
                # fallback
                boxes = detect_with_morphology(img)
        except Exception as e:
            print("EAST detection failed, falling back to morphology. Error:", e)
            boxes = detect_with_morphology(img)
    else:
        print("EAST model not found (frozen_east_text_detection.pb). Using morphology fallback.")
        boxes = detect_with_morphology(img)

    # sort boxes in reading order
    boxes = sort_boxes_top_to_bottom_left_to_right(boxes, row_tol=25)

    # save crops and visualization
    meta = []
    for i, (x0,y0,x1,y1) in enumerate(boxes):
        crop = img[y0:y1, x0:x1]
        fname = f"crop_{i:03d}.png"
        if save_crops:
            cv2.imwrite(os.path.join(CROPS_DIR, fname), crop)
        meta.append((i, fname, x0, y0, x1, y1))
        if visualize:
            color = (0, 255, 0)
            cv2.rectangle(orig_vis, (x0,y0), (x1,y1), color, 2)
            cv2.putText(orig_vis, str(i), (x0, max(10,y0-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

    vis_path = os.path.join(VIS_DIR, "detections.png")
    cv2.imwrite(vis_path, orig_vis)
    # write meta
    with open(META_FILE, "w", encoding="utf-8") as f:
        for m in meta:
            f.write("{},{},{},{},{},{}\n".format(*m))
    print(f"Detection finished. {len(meta)} regions saved to {CROPS_DIR}. Visualization -> {vis_path}")
    return meta

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Detect handwritten text regions (EAST or morphology fallback).")
    parser.add_argument("--image", required=True, help="Path to input image")
    args = parser.parse_args()
    detect_and_crop(args.image)
