# src/eval.py

def calculate_accuracy(true_list, pred_list):
    correct = sum([1 for t, p in zip(true_list, pred_list) if t == p])
    return 100 * correct / len(true_list) if len(true_list) else 0.0

def calculate_final_grade(accuracy_percent, processing_time_sec):
    if accuracy_percent < 60 or processing_time_sec > 60:
        return 2.0
    accuracy_norm = (accuracy_percent - 60) / 40
    time_norm = (60 - processing_time_sec) / 50
    score = 0.7 * accuracy_norm + 0.3 * time_norm
    grade = 2.0 + 3.0 * score
    return round(grade * 2) / 2

def calculate_iou(boxA, boxB):
    # (xc, yc, w, h), wszystko w proporcjach [0, 1]
    xA1 = boxA[0] - boxA[2]/2
    yA1 = boxA[1] - boxA[3]/2
    xA2 = boxA[0] + boxA[2]/2
    yA2 = boxA[1] + boxA[3]/2
    xB1 = boxB[0] - boxB[2]/2
    yB1 = boxB[1] - boxB[3]/2
    xB2 = boxB[0] + boxB[2]/2
    yB2 = boxB[1] + boxB[3]/2
    inter_x1 = max(xA1, xB1)
    inter_y1 = max(yA1, yB1)
    inter_x2 = min(xA2, xB2)
    inter_y2 = min(yA2, yB2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    areaA = (xA2 - xA1) * (yA2 - yA1)
    areaB = (xB2 - xB1) * (yB2 - yB1)
    union_area = areaA + areaB - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area
