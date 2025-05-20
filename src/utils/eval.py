import numpy as np

def calculate_iou(boxA, boxB):
    # [xmin, ymin, xmax, ymax]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    if boxAArea + boxBArea - interArea == 0:
        return 0.0
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def calculate_accuracy(true_plates, pred_plates):
    # Liczba identycznych predykcji (po usunięciu białych znaków i "-")
    acc = np.mean([tp == pp for tp, pp in zip(true_plates, pred_plates)]) * 100
    return acc

def calculate_final_grade(accuracy, total_time):
    # Przykładowy wzór, dostosuj wg wymagań prowadzącego
    if accuracy > 85 and total_time < 20:
        return 5.0
    elif accuracy > 60:
        return 4.0
    elif accuracy > 30:
        return 3.0
    else:
        return 2.0
