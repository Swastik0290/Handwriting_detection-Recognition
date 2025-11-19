# utils.py
import os
import cv2

def ensure_dir(path):
    if not path:
        return
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def sort_boxes_top_to_bottom_left_to_right(boxes, row_tol=20):
    """
    boxes: list of (x, y, w, h) or (x0,y0,x1,y1)
    Return boxes sorted reading order.
    """
    # Normalize to (x0,y0,x1,y1)
    norm = []
    for b in boxes:
        if len(b) == 4:
            x0,y0,x1,y1 = b
        else:
            x,y,w,h = b
            x0,y0,x1,y1 = x,y,x+w,y+h
        norm.append((x0,y0,x1,y1))
    # sort by y then x with tolerance
    norm = sorted(norm, key=lambda r: r[1])
    rows = []
    cur = []
    for r in norm:
        if not cur:
            cur = [r]
            continue
        avg_y = sum([c[1] for c in cur]) / len(cur)
        if abs(r[1] - avg_y) <= row_tol:
            cur.append(r)
        else:
            rows.append(sorted(cur, key=lambda c: c[0]))
            cur = [r]
    if cur:
        rows.append(sorted(cur, key=lambda c: c[0]))
    out = [c for row in rows for c in row]
    return out

def resize_keep_aspect(img, target_h=32, max_w=1024):
    """Resize grayscale image to target_h keeping aspect ratio. Return HxW array normalized [0,1]."""
    h, w = img.shape[:2]
    new_h = target_h
    new_w = max(1, int(w * (new_h / float(h))))
    if new_w > max_w:
        new_w = max_w
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    resized = resized.astype('float32') / 255.0
    return resized
