from ultralytics import YOLO

def train():
    model = YOLO("yolo11x.pt")

    results = model.train(
        data="../dataset.yaml",
        epochs=120,
        imgsz=1280,
        batch=4,
        patience=30,
        optimizer='auto',
        val=True,
        workers=4
    )

    print("Trening zakończony! Najlepszy model: runs/detect/train/weights/best.pt")

if __name__ == "__main__":
    train()