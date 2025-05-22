import torch
import cv2
import numpy as np


class PlateDetector:
    def __init__(self, weight_path='model/best.pt', device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Load the YOLOv5 model with custom weights for license plate detection
        self.model = torch.hub.load(
            'ultralytics/yolov5', 'custom', path=weight_path, source='github', device=device
        )  # Load YOLO model:contentReference[oaicite:6]{index=6}
        # Model loaded; will run on GPU if available, otherwise on CPU.

    def detect_plate(self, img_path):
        # Read the image from file
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        # Perform plate detection using YOLO
        results = self.model(img)  # inference on the image
        # Extract detection results (format: x1, y1, x2, y2, confidence, class)
        detections = results.xyxy[0].cpu().numpy()
        if detections.shape[0] == 0:
            return None, None  # No plate detected in the image
        # Take the highest-confidence detection (assuming one plate per image)
        x1, y1, x2, y2, conf, cls = detections[0]
        # Convert coordinates to int and crop the plate region
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        plate_img = img[y1:y2, x1:x2].copy()
        return plate_img, (x1, y1, x2, y2)
