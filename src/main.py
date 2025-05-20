import time
from src.utils.config import XML_PATH, IMAGES_DIR, CNN_MODEL_PATH, CROP_SIZE, TEST_RATIO, RANDOM_SEED
from src.utils.loader import load_cvat_xml
from src.detector.plate_detector import train_plate_detector, load_plate_detector, predict_box
from src.detector.ocr_reader import ocr_plate
from src.utils.image_utils import crop_plate_from_yolo
from src.utils.eval import calculate_accuracy, calculate_final_grade, calculate_iou
import cv2
from utils.image_utils import draw_box_on_img

def main(train=True):
    # Ładowanie danych
    train_set, test_set = load_cvat_xml(XML_PATH, IMAGES_DIR, TEST_RATIO, RANDOM_SEED)

    # Trening detektora
    if train:
        print("== Trening modelu detektora ==")
        train_plate_detector(
            train_set, test_set,
            model_path=CNN_MODEL_PATH,
            crop_size=(512, 256),
            epochs=20, batch_size=16, lr=1e-3
        )

    # Ewaluacja
    print("== Ewaluacja na zbiorze testowym ==")
    model = load_plate_detector(CNN_MODEL_PATH, crop_size=(512, 256))
    print("== Podgląd detekcji na pierwszych 10 zdjęciach ==")
    for i, (img_path, true_box, true_plate, img_w, img_h) in enumerate(test_set[:10]):
        pred_box = predict_box(model, img_path, crop_size=(512, 256))
        img = cv2.imread(img_path)
        img_pred = draw_box_on_img(img, pred_box, img_w, img_h, color=(0, 255, 0))
        img_all = draw_box_on_img(img_pred, true_box, img_w, img_h, color=(0, 0, 255))
        out_path = f"debug_result_{i}.jpg"
        cv2.imwrite(out_path, img_all)
        print(f"Saved debug visualization: {out_path}")

    # Ewaluacja
    true_plates = []
    pred_plates = []
    ious = []
    start_time = time.time()

    for (img_path, true_box, true_plate, img_w, img_h) in test_set[:100]:
        pred_box = predict_box(model, img_path, crop_size=(512, 256))
        iou = calculate_iou(true_box, pred_box)
        ious.append(iou)
        plate_img = crop_plate_from_yolo(img_path, pred_box, img_w, img_h, out_size=CROP_SIZE)
        pred_plate = ocr_plate(plate_img)
        true_plates.append(true_plate)
        pred_plates.append(pred_plate)
    total_time = time.time() - start_time
    accuracy = calculate_accuracy(true_plates, pred_plates)
    mean_iou = sum(ious) / len(ious) if ious else 0.0

    print(f'Dokładność OCR: {accuracy:.2f}%')
    print(f'IoU (średnie): {mean_iou:.3f}')
    print(f'Czas przetwarzania 100 zdjęć: {total_time:.2f}s')
    print(f'Ocena końcowa: {calculate_final_grade(accuracy, total_time)}')

if __name__ == '__main__':
    main(train=False)  # lub train=False jeśli nie chcesz ponownie trenować
