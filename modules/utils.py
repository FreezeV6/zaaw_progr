def calculate_iou(box1, box2):
    """
    Oblicza Intersection over Union (IoU) dla dwóch prostokątów.
    Każdy box definiowany przez krotkę (x_min, y_min, x_max, y_max).
    Zwraca wartość IoU w zakresie [0, 1].
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    # Oblicz współrzędne części wspólnej prostokątów
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height
    # Obszary poszczególnych prostokątów
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    # Union area (suma pól minus część wspólna)
    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0.0
    iou = inter_area / union_area
    return iou

def calculate_final_grade(accuracy_percent, processing_time_sec):
    """
    Oblicza ocenę końcową systemu ALPR na podstawie dokładności OCR i czasu przetwarzania.
    Zgodnie z podaną formułą:
      - Jeśli accuracy < 60% lub czas > 60s (dla 100 zdjęć) => ocena 2.0.
      - W przeciwnym razie liczymy ocenę wagowo: 0.7 * (acc_norm) + 0.3 * (time_norm) przeskalowaną na [2.0, 5.0].
    :param accuracy_percent: dokładność OCR w procentach (0-100).
    :param processing_time_sec: czas przetworzenia 100 zdjęć (sekundy).
    :return: końcowa ocena (float z zaokrągleniem do najbliższej 0.5).
    """
    # Sprawdź warunki minimalne
    if accuracy_percent < 60 or processing_time_sec > 60:
        return 2.0
    # Normalizacja accuracy: 60% -> 0.0, 100% -> 1.0
    acc_norm = (accuracy_percent - 60.0) / 40.0
    if acc_norm < 0: acc_norm = 0.0
    if acc_norm > 1: acc_norm = 1.0
    # Normalizacja czasu: 60s -> 0.0, 10s -> 1.0
    time_norm = (60.0 - processing_time_sec) / 50.0
    if time_norm < 0: time_norm = 0.0
    if time_norm > 1: time_norm = 1.0
    # Metryka ważona z wagami 0.7 (accuracy) i 0.3 (czas)
    score = 0.7 * acc_norm + 0.3 * time_norm
    # Przeskalowanie na ocenę 2.0 - 5.0
    grade = 2.0 + 3.0 * score
    # Zaokrąglenie do najbliższej połowy (0.5)
    grade_rounded = round(grade * 2) / 2.0
    return grade_rounded
