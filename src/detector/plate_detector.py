import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os

class PlateBoxDataset(Dataset):
    def __init__(self, data, crop_size=(512, 256)):
        self.data = data
        self.crop_size = crop_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, box, _, img_w, img_h = self.data[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.crop_size)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # CxHxW
        box = np.array(box, dtype=np.float32)
        return torch.tensor(img, dtype=torch.float32), torch.tensor(box, dtype=torch.float32)

class PlateBoxCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.regressor = nn.Sequential(
            nn.Linear(64 * 64 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid()  # wyjście w zakresie 0-1
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x

def train_plate_detector(train_set, valid_set, model_path, crop_size=(512, 256),
                        epochs=20, batch_size=16, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PlateBoxCNN().to(device)
    train_ds = PlateBoxDataset(train_set, crop_size)
    val_ds = PlateBoxDataset(valid_set, crop_size)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for imgs, targets in train_dl:
            imgs, targets = imgs.to(device), targets.to(device)
            preds = model(imgs)
            loss = loss_fn(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_dl.dataset)

        # Walidacja
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, targets in val_dl:
                imgs, targets = imgs.to(device), targets.to(device)
                preds = model(imgs)
                loss = loss_fn(preds, targets)
                val_loss += loss.item() * imgs.size(0)
            val_loss /= len(val_dl.dataset)
        print(f"Epoch {epoch+1}/{epochs} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print("Best model saved.")

    print("Training finished.")
    return model

def load_plate_detector(model_path, crop_size=(512, 256)):
    model = PlateBoxCNN()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def predict_box(model, img_path, crop_size=(512, 256)):
    img = cv2.imread(img_path)
    img = cv2.resize(img, crop_size)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = torch.tensor(img).unsqueeze(0)
    with torch.no_grad():
        pred = model(img).squeeze(0).numpy()
    return tuple(pred)  # (x_center, y_center, width, height) w proporcjach 0-1


