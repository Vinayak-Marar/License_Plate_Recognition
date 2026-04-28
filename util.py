import string
import re
import numpy as np
import cv2
import torch
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHARS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N','O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
]

BLANK_IDX = len(CHARS)

dict_char_to_int = {'O': '0','C':'0','D':'0','Q':'0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5', 'Z': '2'}
dict_int_to_char = {'0': 'O', '1': 'I','2':'Z', '3': 'J', '4': 'A', '6': 'G', '5': 'S', '7': 'T','8':'B'}

def license_complies_format(text):
    length = len(text)
    if length not in [9, 10, 11]:
        return False

    if length == 9:
        return bool(re.match(r'^[A-Z]{2}[0-9]{2}[A-Z]{1}[0-9]{4}$', text))
    if length == 10:
        return bool(re.match(r'^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$', text))
    if length == 11:
        return bool(re.match(r'^[A-Z]{2}[0-9]{2}[A-Z]{3}[0-9]{4}$', text))
    # print(text)
    return False

def format_license(text):
    length = len(text)
    res = list(text)

    if length == 10:
        alpha_indices, num_indices = [0, 1, 4, 5], [2, 3, 6, 7, 8, 9]
    elif length == 9:
        alpha_indices, num_indices = [0, 1, 4], [2, 3, 5, 6, 7, 8]
    else:
        return False

    for i in range(length):
        if i in alpha_indices and res[i] in dict_int_to_char:
            res[i] = dict_int_to_char[res[i]]
        elif i in num_indices and res[i] in dict_char_to_int:
            res[i] = dict_char_to_int[res[i]]

    return "".join(res)

def _preprocess_crop(crop):
    img = cv2.resize(crop, (94, 24)).astype("float32")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    return img

def _decode_indices(pred_indices):
    decoded_label = []
    prev = None
    for idx in pred_indices:
        idx = int(idx)
        if idx != BLANK_IDX and idx != prev:
            decoded_label.append(CHARS[idx])
        prev = idx
    return "".join(decoded_label)

def compute_confidence_single(probs, indices, blank_idx):
    """
    probs: [C, T]
    indices: [T]
    """
    conf_list = []
    prev = -1

    for t in range(indices.shape[0]):
        idx = int(indices[t])
        if idx != blank_idx and idx != prev:
            conf_list.append(float(probs[idx, t].item()))
        prev = idx

    if not conf_list:
        return 0.0
    return float(sum(conf_list) / len(conf_list))

def batch_read_license_plates(crops, lpr_model):
    """
    crops: list[np.ndarray]
    returns: list[tuple[text, confidence]]
    """
    if not crops:
        return []

    processed = []
    valid_positions = []

    for i, crop in enumerate(crops):
        if crop is None or crop.size == 0:
            continue
        try:
            processed.append(_preprocess_crop(crop))
            valid_positions.append(i)
        except Exception:
            continue

    if not processed:
        return [(None, 0.0) for _ in crops]

    batch = torch.from_numpy(np.stack(processed, axis=0)).to(DEVICE)

    with torch.no_grad():
        preds = lpr_model(batch)

    probs = F.softmax(preds, dim=1)                      # [B, C, T]
    _, max_indices = torch.max(probs, dim=1)             # [B, T]
    argmax_seq = preds.permute(2, 0, 1).argmax(2).permute(1, 0)  # [B, T]

    outputs = [(None, 0.0) for _ in crops]

    for batch_idx, orig_idx in enumerate(valid_positions):
        pred_indices = argmax_seq[batch_idx].cpu().numpy()
        text = _decode_indices(pred_indices)
        conf = compute_confidence_single(
            probs[batch_idx].cpu(),
            max_indices[batch_idx].cpu(),
            BLANK_IDX
        )
  
        outputs[orig_idx] = (format_license(text), conf)

    return outputs

def read_license_plate(crop, lpr_model):
    results = batch_read_license_plates([crop], lpr_model)
    return results[0] if results else (None, 0.0)

def draw_box(frame, box, conf, plate_text, color='r'):
    x1, y1, x2, y2 = map(int, box)

    box_h = max(1, y2 - y1)
    box_w = max(1, x2 - x1)

    # Colors
    if color == 'b':
        c = (255, 255, 0)   # blue
    elif color == 'r':
        c = (0, 165, 255)   # red
    else:
        c = (0, 255, 0)   # green

    # Draw rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)

    if not plate_text:
        return

    label = f"{plate_text}"

    # Different defaults for plate vs vehicle labels
    if color == 'b':
        font_scale = min(1.0, max(0.45, box_h / 70.0))
    else:
        font_scale = min(0.85, max(0.40, box_h / 110.0))

    thickness = max(1, int(round(font_scale * 2)))

    # Shrink text until it fits the box width reasonably well
    max_text_width = max(40, box_w - 8)
    while True:
        (tw, th), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness
        )
        if tw <= max_text_width or font_scale <= 0.35:
            break
        font_scale *= 0.90
        thickness = max(1, int(round(font_scale * 2)))

    # Put label above the box if there is space, otherwise inside the top
    pad = 4
    if y1 - th - baseline - 2 * pad > 0:
        text_y = y1 - 6
        bg_top = text_y - th - baseline - pad
        bg_bottom = text_y + baseline + pad
    else:
        text_y = y1 + th + pad
        bg_top = y1
        bg_bottom = y1 + th + baseline + 2 * pad

    bg_left = x1
    bg_right = min(frame.shape[1] - 1, x1 + tw + 2 * pad)

    # Background
    cv2.rectangle(frame, (bg_left, max(0, bg_top)), (bg_right, min(frame.shape[0] - 1, bg_bottom)), c, -1)

    # Text
    cv2.putText(
        frame,
        label,
        (bg_left + pad, min(frame.shape[0] - 1, text_y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA
    )

def draw_quit_hint(frame, text="Quit (Q)"):
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)

    x = w - tw - 15
    y = th + 15

    cv2.rectangle(frame, (x - 10, y - th - 10), (x + tw + 10, y + 5), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, scale, (0, 255, 0), thickness, cv2.LINE_AA)


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def box_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) // 2, (y1 + y2) // 2


def point_inside_box(px, py, box):
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    area_a = box_area(a)
    area_b = box_area(b)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def safe_crop(img, box):
    x1, y1, x2, y2 = box
    h, w = img.shape[:2]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def ocr_plate(plate_crop, lprnet):
    try:
        out = read_license_plate(plate_crop, lprnet)
    except Exception:
        return None, 0.0
    
    if isinstance(out, tuple):
        if len(out) >= 2:
            text = out[0]
            conf = out[1]
            return text, float(conf) if conf is not None else 0.0
        if len(out) == 1:
            return out[0], 0.0

    if out is None:
        return None, 0.0

    return out, 0.0

def match_plate_to_vehicle(plate_box, vehicles):
    # DSA: Point-in-box check is O(1) compared to IoU O(N)
    pcx, pcy = (plate_box[0] + plate_box[2]) // 2, (plate_box[1] + plate_box[3]) // 2
    
    best_match = None
    min_dist = float('inf')

    for v in vehicles:
        vx1, vy1, vx2, vy2 = v["bbox"]
        # Fast check: Is the plate center even inside the vehicle box?
        if vx1 <= pcx <= vx2 and vy1 <= pcy <= vy2:
            # If multiple vehicles overlap (rare), pick the one with the smallest area
            area = (vx2 - vx1) * (vy2 - vy1)
            if area < min_dist:
                min_dist = area
                best_match = v
                
    return best_match
