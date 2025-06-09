import cv2
from detector import PlateDetector
from ensemble import process_image
from utils import crop_bbox
from config import CAMERA_INDEX, YOLO_CONFIDENCE, YOLO_NMS_IOU


def camera_loop():
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
                plate_img = crop_bbox(frame, bbox)
                plate_text = process_image(plate_img)
                x1, y1, x2, y2 = [int(x) for x in bbox]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, plate_text, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.imshow("ALPR Live", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    camera_loop()