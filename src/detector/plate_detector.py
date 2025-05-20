import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import cv2

# === MODEL: Transfer Learning ===
class PlateDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights="IMAGENET1K_V1")
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 4)  # 4 for [x_c, y_c, w, h]

    def forward(self, x):
        x = self.backbone(x)
        # Clamp to [0,1] to force YOLO format
        x = torch.sigmoid(x)
        return x

# === TRAINING ===
def train_plate_detector(train_set, val_set, model_path, crop_size, epochs, batch_size, lr):
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    import torch.nn.functional as F
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # DATASET
    class PlateDataset(Dataset):
        def __init__(self, data):
            self.data = data
            self.crop_size = crop_size

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            img_path, box, _, _, _ = self.data[idx]
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.crop_size)
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1)
            box = torch.tensor(box, dtype=torch.float32)  # [x_c, y_c, w, h] in [0,1]
            return img, box

    train_ds = PlateDataset(train_set)
    val_ds = PlateDataset(val_set)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = PlateDetector().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            preds = model(imgs)
            loss = F.mse_loss(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * imgs.size(0)
        avg_train_loss = total_train_loss / len(train_ds)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                preds = model(imgs)
                loss = F.mse_loss(preds, targets)
                total_val_loss += loss.item() * imgs.size(0)
        avg_val_loss = total_val_loss / len(val_ds)

        print(f"Epoch {epoch}/{epochs} | train_loss: {avg_train_loss:.2f} | val_loss: {avg_val_loss:.2f}")
        if avg_val_loss < best_val_loss:
            print("Best model saved.")
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)

    print("Training finished.")

def load_plate_detector(model_path, crop_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PlateDetector().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict_box(model, img_path, crop_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img = cv2.imread(img_path)
    img_h, img_w = img.shape[:2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, crop_size)
    img_tensor = torch.from_numpy(img_resized.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(img_tensor)[0].cpu().numpy()
    # Clamp values (should be in [0,1])
    pred = np.clip(pred, 0, 1)
    # Debug print
    print(f"[DEBUG] Predicted box YOLO: {pred}")
    return pred
