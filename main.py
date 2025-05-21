import os
from src.data_utils import prepare_dataset
from src.train_detector import train
from src.plate_detector import PlateDetector
from src.plate_ocr import PlateOCR
from src.evaluate import evaluate

def run_training():
    train(
        data_yaml='data/data.yaml',
        epochs=50,
        imgsz=640,
        batch=8,
        model_size='n'
    )

def run_evaluation():
    detector = PlateDetector('models/best_plate_model.tflite', threshold=0.25)
    ocr = PlateOCR(gpu=False)
    # Parsowanie testowych rekordów (po podziale)
    _, test_recs = prepare_dataset(
        xml_path='data/annotations.xml',
        images_src_dir='data/images',    # tutaj Twoje oryginały
        images_dst_dir='data/images',    # nadpisujemy w subfolderach
        labels_dst_dir='data/labels',
        test_size=0.3
    )
    acc, miou, ttime, grade = evaluate(detector, ocr, test_recs, 'data/images')
    print(f"Accuracy OCR: {acc:.2f}%")
    print(f"Mean IoU:     {miou:.2f}%")
    print(f"Time (100 imgs ext.): {ttime:.2f} s")
    print(f"Final grade:  {grade:.1f}")

if __name__ == '__main__':
    # Przygotowanie danych + (opcjonalnie) trening + ewaluacja
    prepare_train, _ = prepare_dataset(
        xml_path='data/annotations.xml',
        images_src_dir='data/images',    # folder z oryginałami
        images_dst_dir='data/images',    # zostanie utworzony train/val
        labels_dst_dir='data/labels',
        test_size=0.3
    )
    # Odkomentuj, jeśli chcesz trenować
    run_training()
    run_evaluation()
