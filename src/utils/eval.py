from difflib import SequenceMatcher
import numpy as np

def calculate_iou(boxA, boxB):
    # Konwersja do xyxy
    def to_xyxy(box):
        x, y, w, h = box
        return [x-w/2, y-h/2, x+w/2, y+h/2]
    boxA = np.array(boxA, dtype=float)
    boxB = np.array(boxB, dtype=float)
    boxA = to_xyxy(boxA)
    boxB = to_xyxy(boxB)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def fuzzy_match(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()
