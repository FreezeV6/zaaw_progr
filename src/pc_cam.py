import cv2
from detector import PlateDetector
from ocr import recognize_plate
from utils import crop_bbox
from config import (
    CAMERA_INDEX,
    LIVE_PREPROCESS_PARAMS,
    LIVE_CROP_OFFSETS,
    LIVE_SHRINK_RATIO,
    YOLO_CONFIDENCE,
    YOLO_NMS_IOU,
    TESSERACT_CONFIG,
    OCR_CONF_MIN,
)


def camera_loop():
    """Run detection on frames from a PC webcam and display results."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    detector = PlateDetector()

    if not cap.isOpened():
        print(f"Cannot open camera index {CAMERA_INDEX}")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            dets = detector.detect(frame, conf=YOLO_CONFIDENCE, iou=YOLO_NMS_IOU)
            for det in dets:
                bbox = det[:4]
                x1, y1, x2, y2 = [int(x) for x in bbox]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                plate_img = crop_bbox(frame, bbox, offsets=LIVE_CROP_OFFSETS, shrink_ratio=LIVE_SHRINK_RATIO)
                plate_txt, prep_img = recognize_plate(
                    plate_img,
                    "cam",
                    preprocess_params=LIVE_PREPROCESS_PARAMS,
                    tesseract_config=TESSERACT_CONFIG,
                    ocr_conf_min=OCR_CONF_MIN,
                    show_prep_img=True
                )
                cv2.putText(frame, plate_txt, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.imshow('OCR', prep_img)

            cv2.imshow("ALPR", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    camera_loop()