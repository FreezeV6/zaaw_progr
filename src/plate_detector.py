import cv2
import numpy as np
import torch
from ultralytics import YOLO

class PlateDetector:
    def __init__(self, model_path: str, conf_thres: float = 0.25):
        # will auto‐use GPU if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading YOLO model `{model_path}` on {device}…")
        self.model = YOLO(model_path)
        self.conf_thres = conf_thres

    def detect(self, img: np.ndarray):
        """
        Runs detection, returns list of (x1,y1,x2,y2,score).
        """
        # YOLO returns a list of Results; we take the first
        results = self.model(img, imgsz=640, conf=self.conf_thres)[0]
        boxes = results.boxes.xyxy.cpu().numpy()   # shape (N,4)
        scores = results.boxes.conf.cpu().numpy()  # shape (N,)
        out = []
        for (x1,y1,x2,y2), s in zip(boxes, scores):
            out.append((int(x1), int(y1), int(x2), int(y2), float(s)))
        return out

    def detect_and_crop(self, img: np.ndarray):
        """
        Returns list of cropped plate images + their scores.
        """
        dets = self.detect(img)
        crops = []
        for x1,y1,x2,y2,score in dets:
            plate = img[y1:y2, x1:x2]
            if plate.size == 0:
                continue
            crops.append((plate, score))
        return crops
