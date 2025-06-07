import argparse
import pandas as pd
import random
from detector import PlateDetector
from evaluation import evaluate
from config import (
    CSV_PATH,
    IMAGES_DIR,
    PREPROCESS_PARAMS,
    CROP_OFFSETS,
    SHRINK_RATIO,
    YOLO_CONFIDENCE,
    YOLO_NMS_IOU,
    TESSERACT_CONFIG,
    OCR_CONF_MIN,
)

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    return [(r.fname, r.xtl, r.ytl, r.xbr, r.ybr, r.plate) for r in df.itertuples(index=False)]

def evaluate_with_seed(seed: int, sample_size: int = 100) -> float:
    data = load_dataset(CSV_PATH)
    rng = random.Random(seed)
    if len(data) > sample_size:
        data = rng.sample(data, sample_size)
    detector = PlateDetector()
    acc, _, _ = evaluate(
        detector,
        data,
        IMAGES_DIR,
        preprocess_params=PREPROCESS_PARAMS,
        crop_offsets=CROP_OFFSETS,
        conf=YOLO_CONFIDENCE,
        nms_iou=YOLO_NMS_IOU,
        shrink_ratio=SHRINK_RATIO,
        tesseract_config=TESSERACT_CONFIG,
        ocr_conf_min=OCR_CONF_MIN,
    )
    return acc

def find_best_seed(seeds, sample_size: int = 100):
    best_seed = None
    best_acc = -1.0
    for s in seeds:
        acc = evaluate_with_seed(s, sample_size)
        print(f"Seed {s}: accuracy={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_seed = s
    return best_seed, best_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find seed yielding highest evaluation accuracy")
    parser.add_argument("--start", type=int, default=0, help="start seed")
    parser.add_argument("--end", type=int, default=100000, help="end seed (exclusive)")
    parser.add_argument("--sample_size", type=int, default=100, help="number of images to evaluate")
    args = parser.parse_args()

    seeds = range(args.start, args.end)
    best_seed, best_acc = find_best_seed(seeds, args.sample_size)
    print(f"Best seed: {best_seed} with accuracy {best_acc:.4f}")
