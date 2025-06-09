from ultralytics import YOLO
from config import MODEL_PATH
import numpy as np


class PlateDetector:
    def __init__(self, model_path=MODEL_PATH, default_conf: float = 0.25, default_iou: float = 0.45):
        self.model = YOLO(model_path)
        self.default_conf = default_conf
        self.default_iou = default_iou

    def detect(self, image, conf: float | None = None, iou: float | None = None):
        conf_val = conf if conf is not None else self.default_conf
        iou_val = iou if iou is not None else self.default_iou
        results = self.model.predict(image, conf=conf_val, iou=iou_val, verbose=False)
        if not results:
            return []
        # xyxy: (N,4), confidence, class
        boxes = results[0].boxes
        preds = np.hstack(
            (
                boxes.xyxy.cpu().numpy(),
                boxes.conf.cpu().numpy().reshape(-1, 1),
                boxes.cls.cpu().numpy().reshape(-1, 1),
            )
        )
        preds = preds[preds[:, 4].argsort()[::-1]]
        return preds