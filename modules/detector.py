import os
import torch
from ultralytics import YOLO

def train_detector(epochs=20, img_size=640, batch_size=8, device=None, project="runs", name="plate-detector"):
    """
    Trenuje model detekcji tablic rejestracyjnych YOLOv8.
    Zwraca ścieżkę do wytrenowanych wag (best.pt).
    """
    # Wybór urządzenia
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = YOLO("yolov8n.pt")
    # Trening
    model.train(
        data=os.path.join("data", "data.yaml"),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=project,
        name=name,
        exist_ok=True
    )
    # Ścieżki do wag
    weights_dir = os.path.join(project, name, "weights")
    best = os.path.join(weights_dir, "best.pt")
    last = os.path.join(weights_dir, "last.pt")
    # Wybieramy best.pt lub, jeśli go nie ma, last.pt
    weights = best if os.path.isfile(best) else last
    if not os.path.isfile(weights):
        raise FileNotFoundError(f"Po treningu nie znaleziono wag: {best} ani {last}")
    return weights

def load_detector(weights_path=None, device=None):
    """
    Wczytuje model YOLOv8 z podanych wag.
    Jeśli weights_path nie podano, szuka runs/plate-detector/weights/best.pt.
    """
    if weights_path is None:
        default = os.path.join("runs", "plate-detector", "weights", "best.pt")
        weights_path = default if os.path.isfile(default) else None
    if not weights_path or not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Nie znaleziono pliku wag do ładowania: {weights_path}")
    # Wybór urządzenia
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return YOLO(weights_path)

def detect_plates(model, image_paths, imgsz=640, iou=0.5, conf=0.5, batch=16):
    """
    Wykrywa tablice na liście obrazów za pomocą wczytanego modelu.
    Zwraca dict: filename -> lista {'bbox':(x1,y1,x2,y2), 'conf': float}.
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
        preds.sort(key=lambda x: x['conf'], reverse=True)
        detections[fname] = preds
    return detections
