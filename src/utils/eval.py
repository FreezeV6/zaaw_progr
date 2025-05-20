def calculate_accuracy(true_plates, pred_plates):
    correct = 0
    for t, p in zip(true_plates, pred_plates):
        if t and p and t == p:
            correct += 1
    return (correct / len(true_plates)) * 100 if true_plates else 0

def calculate_iou(boxA, boxB):
    if boxA is None or boxB is None or len(boxA) != 4 or len(boxB) != 4:
        return 0.0
    # dalej bez zmian
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def calculate_final_grade(accuracy, time_sec):
    # Twoja metryka
    if time_sec < 20 and accuracy > 90:
        return 5.0
    elif accuracy > 70:
        return 4.0
    else:
        return 3.0

from .image_utils import yolo2abs  # konieczne do IoU
