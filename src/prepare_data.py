import os
import xml.etree.ElementTree as ET
import random
import shutil

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
ANNOT_FILE = os.path.join(DATA_DIR, 'annotations.xml')
IMG_DIR    = os.path.join(DATA_DIR, 'images')
LABELS_DIR = os.path.join(DATA_DIR, 'labels')

def download_with_kaggle():
    """Attempt to download via Kaggle API if annotations/images aren't already present."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("Kaggle API not installed; skipping auto‐download.")
        return
    api = KaggleApi()
    api.authenticate()
    print("📥 Downloading dataset from Kaggle…")
    api.dataset_download_files(
        'piotrstefaskiue/poland-vehicle-license-plate-dataset',
        path=DATA_DIR, unzip=True
    )
    print("➡️  Moving images/annotations…")
    # assume it unzips into DATA_DIR/poland-vehicle-license-plate-dataset
    src = os.path.join(DATA_DIR, 'poland-vehicle-license-plate-dataset')
    # move images
    imgs = os.path.join(src, 'images')
    if os.path.isdir(imgs):
        shutil.move(imgs, IMG_DIR)
    # move annotations.xml
    annot = os.path.join(src, 'annotations.xml')
    if os.path.isfile(annot):
        shutil.move(annot, ANNOT_FILE)
    # cleanup
    shutil.rmtree(src, ignore_errors=True)

def parse_and_split():
    os.makedirs(LABELS_DIR, exist_ok=True)
    train_dir = os.path.join(LABELS_DIR, 'train')
    val_dir   = os.path.join(LABELS_DIR, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir,   exist_ok=True)

    # parse XML
    tree = ET.parse(ANNOT_FILE)
    root = tree.getroot()
    items = []
    for img in root.findall('image'):
        name = img.get('name')
        # get the plate text
        box = img.find('box')
        plate_text = box.find('attribute').text.strip()
        items.append((name, plate_text))

    random.seed(42)
    random.shuffle(items)
    n_val = int(0.3 * len(items))
    val_items   = items[:n_val]
    train_items = items[n_val:]

    # write ground truth
    with open(os.path.join(val_dir, 'gt.txt'), 'w', encoding='utf-8') as f:
        for fn, txt in val_items:
            f.write(f"{fn} {txt}\n")

    with open(os.path.join(train_dir, 'gt.txt'), 'w', encoding='utf-8') as f:
        for fn, txt in train_items:
            f.write(f"{fn} {txt}\n")

    print(f"➗ Split {len(train_items)} train / {len(val_items)} val.")

if __name__ == '__main__':
    if not os.path.isdir(IMG_DIR) or not os.path.isfile(ANNOT_FILE):
        download_with_kaggle()
    if not os.path.isdir(IMG_DIR):
        raise FileNotFoundError(f"Images folder not found at {IMG_DIR}")
    if not os.path.isfile(ANNOT_FILE):
        raise FileNotFoundError(f"Annotation file not found at {ANNOT_FILE}")
    parse_and_split()
