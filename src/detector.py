from ultralytics import YOLO
from config import MODEL_PATH


class PlateDetector:
    def __init__(self, model_path=MODEL_PATH):
        self.model = YOLO(model_path)

    def detect(self, image):
        results = self.model(image)
        if not results:
            return []
        # xyxy: (N,4), confidence, class
        preds = results[0].boxes.xyxy.cpu().numpy()
        return preds
