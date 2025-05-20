import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import cv2
import numpy as np
from src.utils.image_utils import convert_to_yolo_box

class PlateDataset(Dataset):
    def __init__(self, data, crop_size=(224, 224)):
        self.data = data
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(crop_size),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img_path, box, _, img_w, img_h = self.data[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_w, img_h))
        img = self.transform(img)
        yolo_box = convert_to_yolo_box(*box, img_w, img_h)
        return img, torch.tensor(yolo_box, dtype=torch.float32)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.resnet18(weights=None)
        self.base.fc = nn.Linear(self.base.fc.in_features, 4)  # [xc, yc, w, h]
    def forward(self, x):
        return self.base(x)

def train_plate_detector(train_set, val_set, model_path, crop_size=(224, 224), epochs=30, batch_size=16, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ds = PlateDataset(train_set, crop_size)
    val_ds = PlateDataset(val_set, crop_size)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_dl.dataset)
        # Walidacja
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_dl.dataset)
        print(f"Epoch {epoch+1}/{epochs} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print("Best model saved.")
    print("Training finished.")

def load_plate_detector(model_path, crop_size=(224, 224)):
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def predict_box(model, img_path, crop_size=(224, 224)):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w = img.shape[:2]
    img_resized = cv2.resize(img, crop_size)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    x = transform(img_resized).unsqueeze(0)
    with torch.no_grad():
        pred = model(x).cpu().numpy().flatten()
    pred = np.clip(pred, 0, 1)  # YOLO box, safety
    return pred  # [xc, yc, w, h] w [0,1]
