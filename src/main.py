import pandas as pd
from detector import PlateDetector
from evaluation import evaluate, calculate_final_grade
import random
from config import (
    CSV_PATH,
    IMAGES_DIR,
    PREPROCESS_PARAMS,
    CROP_OFFSETS,
    SHRINK_RATIO,
    YOLO_CONFIDENCE,
    YOLO_NMS_IOU,
    TESSERACT_CONFIG,
)

def load_dataset(csv_path):
    data = []
    df = pd.read_csv(csv_path)
    for row in df.itertuples(index=False):
        data.append((row.fname, row.xtl, row.ytl, row.xbr, row.ybr, row.plate))
    return data

if __name__ == "__main__":
    detector = PlateDetector()
    data = load_dataset(CSV_PATH)
    if len(data) > 100:
        data = random.sample(data, 100)
    accuracy, elapsed, avg_iou = evaluate(
        detector,
        data,
        IMAGES_DIR,
        preprocess_params=PREPROCESS_PARAMS,
        crop_offsets=CROP_OFFSETS,
        conf=YOLO_CONFIDENCE,
        nms_iou=YOLO_NMS_IOU,
        shrink_ratio=SHRINK_RATIO,
        tesseract_config=TESSERACT_CONFIG,
    )
    print(f"Dokładność: {accuracy*100:.2f}%")
    print(f"Czas przetwarzania: {elapsed:.2f} s")
    print(f"Średnie IoU: {avg_iou:.3f}")
    grade = calculate_final_grade(accuracy*100, elapsed)
    print(f"Ocena końcowa: {grade}")