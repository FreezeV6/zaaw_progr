import os
import cv2
import numpy as np
import pytesseract
import re
from ultralytics import YOLO

# ---------- Helper functions ----------
def order_points(pts):
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def crop_and_deskew_plate(image, box, shrink_ratio=0.05):
    x1,y1,x2,y2 = box
    h_img, w_img = image.shape[:2]
    # shrink box
    pad_w = int(shrink_ratio * (x2 - x1))
    pad_h = int(shrink_ratio * (y2 - y1))
    x1p = np.clip(x1 + pad_w, 0, w_img)
    y1p = np.clip(y1 + pad_h, 0, h_img)
    x2p = np.clip(x2 - pad_w, 0, w_img)
    y2p = np.clip(y2 - pad_h, 0, h_img)
    plate = image[y1p:y2p, x1p:x2p].copy()
    # detect contours
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _,thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts,_ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return plate
    cnt = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    pts = cv2.boxPoints(rect).astype("float32")
    src = order_points(pts)
    tl,tr,br,bl = src
    widthA = np.linalg.norm(br-bl)
    widthB = np.linalg.norm(tr-tl)
    maxW = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr-br)
    heightB = np.linalg.norm(tl-bl)
    maxH = max(int(heightA), int(heightB))
    dst = np.array([[0,0], [maxW-1, 0], [maxW-1, maxH-1], [0, maxH-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(plate, M, (maxW, maxH))
    return warped


def preprocess_plate(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=10, tileGridSize=(8,8))
    cl = clahe.apply(gray)
    blur = cv2.GaussianBlur(cl, (5,5), 0)
    _,binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.bitwise_not(binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cv2.resize(closed, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

# ---------- Live ANPR + Barrier Control ----------
if __name__ == '__main__':
    # Tesseract path
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    # Load YOLO model (license plate detector)
    model = YOLO('runs/detect/train/weights/best.pt')
    # Start camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Nie można otworzyć kamery")
        exit(1)
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # detection
        results = model.predict(source=frame, conf=0.15, verbose=False)[0]
        for box in results.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            # deskew and preprocess
            deskewed = crop_and_deskew_plate(frame, [x1, y1, x2, y2], shrink_ratio=0.05)
            # show deskewed raw plate
            cv2.imshow('Deskewed Plate', deskewed)
            proc = preprocess_plate(deskewed)
            # show preprocessed plate for OCR
            cv2.imshow('OCR Input', proc)
            # OCR
            config = '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            text = pytesseract.image_to_string(proc, config=config)
            clean = re.sub(r'[^A-Z0-9]', '', text.upper())
            # draw results
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = clean if clean else 'Nie odczytano'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow('Live ANPR', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
