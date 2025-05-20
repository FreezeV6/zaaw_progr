from ultralytics import YOLO
import numpy as np
import logging

class PlateDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    def detect(self, img):
        results = self.model(img)
        boxes = results[0].boxes.xywhn.cpu().numpy() # normalized xywh
        scores = results[0].boxes.conf.cpu().numpy()
        if len(boxes) == 0:
            logging.warning("No plate detected!")
            return None
        # wybierz box z najwyższym score
        idx = np.argmax(scores)
        return boxes[idx]
    def crop_plate(self, img, box):
        h, w = img.shape[:2]
        x, y, bw, bh = box
        x1 = int((x - bw/2) * w)
        y1 = int((y - bh/2) * h)
        x2 = int((x + bw/2) * w)
        y2 = int((y + bh/2) * h)
        crop = img[max(y1,0):min(y2,h), max(x1,0):min(x2,w)]
        return crop
