import numpy as np


def crop_bbox(image, bbox, offsets: dict | None = None):
    """Crop bounding box with optional offsets for each side.

    Offsets dictionary can contain keys ``x1`` (left), ``x2`` (right),
    ``y1`` (top) and ``y2`` (bottom). Positive values shrink the crop
    from a given side, negative values extend it.
    """
    if offsets is None:
        offsets = {"x1": 15, "x2": 0, "y1": 0, "y2": 0}

    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(round(b)) for b in bbox]

    x1 += offsets.get("x1", 0)
    x2 -= offsets.get("x2", 0)
    y1 += offsets.get("y1", 0)
    y2 -= offsets.get("y2", 0)

    x1 = min(max(0, x1), x2 - 1)
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