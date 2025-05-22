import os
import torch
from ultralytics import YOLO

def train_detector(epochs=20, img_size=640, batch_size=8):
    """
    Trenuje model detekcji tablic rejestracyjnych YOLOv8.
    Jeśli jest GPU, używa go automatycznie; w przeciwnym razie CPU.
    """
    # Wybór urządzenia
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = YOLO("yolov8n.pt")
    # Trening
    model.train(
        data=os.path.join("data", "data.yaml"),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        name="plate-detector",
        exist_ok=True
    )
    # Wczytanie najlepszych wag
    best_weights = os.path.join("runs", "plate-detector", "weights", "best.pt")
    return YOLO(best_weights)

def detect_plates(model, image_paths, imgsz=640, iou=0.5, conf=0.5, batch=16):
    """
    Wykrywa tablice na liście obrazów.
    Automatycznie wybiera device='cuda:0' lub 'cpu'.
    Zwraca dict: filename -> lista słowników {'bbox':(x1,y1,x2,y2), 'conf': float}.
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    results = model.predict(
        source=image_paths,
        imgsz=imgsz,
        iou=iou,
        conf=conf,
        device=device,
        batch=batch,
        verbose=False
    )
    detections = {}
    for img_path, res in zip(image_paths, results):
        fname = os.path.basename(img_path)
        preds = []
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            conf_score = float(box.conf.cpu().numpy()[0])
            preds.append({'bbox': (x1, y1, x2, y2), 'conf': conf_score})
        # sort by pewność malejąco
        preds.sort(key=lambda x: x['conf'], reverse=True)
        detections[fname] = preds
    return detections
