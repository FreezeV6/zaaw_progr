import os
from src.data_utils import prepare_dataset
from src.train_detector import train
from src.plate_detector import PlateDetector
import cv2

def run_training():
    train(
        data_yaml='data/data.yaml',
        epochs=50,
        imgsz=640,
        batch=8,
        model_size='n'
    )

def run_evaluation():
    # wskazujesz swoją ścieżkę do wygenerowanego .onnx
    model_path = "runs/detect/plate-detector10/weights/best.onnx"
    detector = PlateDetector(model_path,
                             conf_threshold=0.25,
                             iou_threshold=0.45,
                             input_size=640)

    img_dir   = "data/images/val"
    lbl_dir   = "data/labels/val"
    # proste demo detekcji na zbiorze walidacyjnym:
    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith((".jpg",".png")): continue
        img_path = os.path.join(img_dir, fname)
        img      = cv2.imread(img_path)
        dets     = detector.detect(img)
        print(f"{fname}: {dets}")
        # narysuj i wyświetl:
        for (x1,y1,x2,y2), score, cls in dets:
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(img, f"{score:.2f}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.imshow("det", img); cv2.waitKey(1000)
    cv2.destroyAllWindows()

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
    # run_training()
    run_evaluation()
