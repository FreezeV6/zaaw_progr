import numpy as np

def crop_bbox(image, bbox):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(round(b)) for b in bbox]
    # Odetnij 12-20 pikseli z lewej (usuwa pasek PL/UE):
    x1 += 15
    x1 = min(x1, x2-1)  # zabezpieczenie
    x1 = max(0, x1)
    x2 = min(x2, w)
    y1 = max(0, y1)
    y2 = min(y2, h)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return image[y1:y2, x1:x2]

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)
