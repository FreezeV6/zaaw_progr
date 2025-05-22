import os
import random
import shutil
import argparse
import xml.etree.ElementTree as ET
import cv2
from ultralytics import YOLO

# ─── PATHS ─────────────────────────────────────────────────────────────────────────
ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA    = os.path.join(ROOT, 'data')
IMAGES  = os.path.join(DATA, 'images')
ANNOT   = os.path.join(DATA, 'annotations.xml')

# **Change this** to `labels` so YOLOv8 finds them next to data/images
LBL_DIR = os.path.join(DATA, 'labels')

IMG_T   = os.path.join(IMAGES, 'train')
IMG_V   = os.path.join(IMAGES, 'val')
LBL_T   = os.path.join(LBL_DIR, 'train')
LBL_V   = os.path.join(LBL_DIR, 'val')

DATA_Y  = os.path.join(DATA, 'data.yaml')
MODEL_OUT_DIR = os.path.join(ROOT, 'models')
MODEL_OUT    = os.path.join(MODEL_OUT_DIR, 'best.pt')

# ─── PREPARE DETECTION DATA ────────────────────────────────────────────────────────
def prepare_detection(val_ratio=0.3):
    print(f"🔧 Preparing detection data (val_ratio={val_ratio})…")
    tree = ET.parse(ANNOT)
    root = tree.getroot()
    items = []
    for img in root.findall('image'):
        name = img.get('name')
        box  = img.find('box')
        xtl, ytl = float(box.get('xtl')), float(box.get('ytl'))
        xbr, ybr = float(box.get('xbr')), float(box.get('ybr'))
        items.append((name, xtl, ytl, xbr, ybr))

    random.seed(42)
    random.shuffle(items)
    n_val = int(val_ratio * len(items))
    train_items, val_items = items[n_val:], items[:n_val]

    # make dirs
    for d in [IMG_T, IMG_V, LBL_T, LBL_V]:
        os.makedirs(d, exist_ok=True)

    def write_split(split_items, img_dir, lbl_dir):
        for name, xtl, ytl, xbr, ybr in split_items:
            # copy image
            src_img = os.path.join(IMAGES, name)
            dst_img = os.path.join(img_dir, name)
            shutil.copy(src_img, dst_img)
            # read size
            im = cv2.imread(src_img)
            h, w = im.shape[:2]
            # to YOLO x_center y_center width height (normalized)
            xc = ((xtl + xbr) / 2) / w
            yc = ((ytl + ybr) / 2) / h
            bw = (xbr - xtl) / w
            bh = (ybr - ytl) / h
            lbl_file = os.path.splitext(name)[0] + '.txt'
            with open(os.path.join(lbl_dir, lbl_file), 'w') as f:
                f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    write_split(train_items, IMG_T, LBL_T)
    write_split(val_items,   IMG_V, LBL_V)

    print(f"  • images/train: {len(train_items)}, images/val: {len(val_items)}")

    # rewrite data.yaml
    with open(DATA_Y, 'w') as f:
        f.write(f"""# YOLOv8 data config (plates)
path: {DATA}
train: images/train
val:   images/val
nc: 1
names: ['plate']
""")
    print(f"  • Wrote {DATA_Y}")

# ─── TRAIN YOLOv8 ─────────────────────────────────────────────────────────────────
def train_yolo(epochs, batch, imgsz):
    print(f"🚀 Starting YOLOv8 training for {epochs} epochs…")
    model = YOLO('yolov8n.pt')  # tiny model as a starting point
    model.train(
        data=DATA_Y,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=os.path.join(ROOT, 'runs', 'train'),
        name='plate_detector'
    )
    src_best = os.path.join(ROOT, 'runs', 'train', 'plate_detector', 'weights', 'best.pt')
    os.makedirs(MODEL_OUT_DIR, exist_ok=True)
    shutil.copy(src_best, MODEL_OUT)
    print(f"✅ Trained model saved to {MODEL_OUT}")

# ─── ENTRYPOINT ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--val',    type=float, default=0.3)
    p.add_argument('--epochs', type=int,   default=50)
    p.add_argument('--batch',  type=int,   default=16)
    p.add_argument('--imgsz',  type=int,   default=640)
    args = p.parse_args()

    if not os.path.isfile(ANNOT):
        raise FileNotFoundError(f"{ANNOT} not found. Run `prepare_data.py` first.")
    prepare_detection(val_ratio=args.val)
    train_yolo(args.epochs, args.batch, args.imgsz)
