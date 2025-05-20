import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import random
import os

# ... tutaj wczytanie własnych utili, zależnie od projektu

class PlateDataset(Dataset):
    # Zaimplementuj z augmentacją – np. flipping, brightness, blur, noise
    def __init__(self, samples, crop_size=(224,224), augment=False):
        self.samples = samples
        self.crop_size = crop_size
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, box, _, img_w, img_h = self.samples[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.crop_size)
        # Konwersja do [0,1], torch.Tensor itd.
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2,0,1))
        # Augmentacje
        if self.augment:
            if random.random() > 0.5:
                img = img[:, :, ::-1]  # flip
                img = img.copy()  # flip
            # inne augmentacje (np. gaussian blur, brightness)
        target = np.array(box, dtype=np.float32)
        return torch.tensor(img.copy()), torch.tensor(target)

# Prosty model CNN jako regresor YOLO-style (można tu użyć np. MobileNet lub inny feature extractor)
class SimplePlateDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,32,3,1,1), nn.ReLU(),
            nn.Conv2d(32,64,3,1,1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(64, 4)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        return self.fc(x)

def train_plate_detector(train_set, test_set, model_path, crop_size, epochs, batch_size, lr):
    model = SimplePlateDetector()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(PlateDataset(train_set, crop_size, augment=True), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(PlateDataset(test_set, crop_size, augment=False), batch_size=batch_size)
    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for imgs, targets in train_loader:
            preds = model(imgs)
            loss = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                preds = model(imgs)
                loss = criterion(preds, targets)
                val_loss += loss.item()
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}/{epochs} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print("Best model saved.")
    print("Training finished.")

def load_plate_detector(model_path, crop_size=(224,224)):
    model = SimplePlateDetector()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_box(model, img_path, crop_size=(224,224)):
    img = cv2.imread(img_path)
    img = cv2.resize(img, crop_size)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2,0,1))
    img_tensor = torch.tensor(img).unsqueeze(0)
    with torch.no_grad():
        pred = model(img_tensor).squeeze().numpy()
    # fallback - center box if prediction is weird
    if np.any(np.isnan(pred)) or np.max(pred) > 1.0 or np.min(pred) < 0:
        pred = np.array([0.5, 0.5, 0.5, 0.2])
    return pred
