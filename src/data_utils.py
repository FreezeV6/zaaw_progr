import os
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import shutil
import cv2

def parse_annotations(xml_path, images_dir):
    """
    Parsuje plik CVAT XML i zwraca listę rekordów:
    { 'filename': str, 'bbox': [x1,y1,x2,y2], 'text': str }
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    records = []
    for img in root.findall('image'):
        fname = img.get('name')
        for box in img.findall('box'):
            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))
            text = box.find("attribute[@name='plate number']").text
            records.append({
                'filename': os.path.join(images_dir, fname),
                'bbox': [xtl, ytl, xbr, ybr],
                'text': text.strip().replace(' ', '')
            })
    return records

def split_data(records, test_size=0.3, seed=42):
    """
    Dzieli rekordy na train/test, zwraca (train, test).
    """
    filenames = [r['filename'] for r in records]
    train_f, test_f = train_test_split(filenames, test_size=test_size, random_state=seed)
    train = [r for r in records if r['filename'] in train_f]
    test  = [r for r in records if r['filename'] in test_f]
    return train, test

def write_splits(split, out_txt):
    """
    Zapisuje listę ścieżek obrazów do pliku .txt (jeden na wiersz).
    """
    with open(out_txt, 'w') as f:
        for r in split:
            f.write(r['filename'] + '\n')

def convert_to_yolo(records, images_subdir, labels_subdir):
    """
    records: lista rekordów z parse_annotations
    images_subdir: 'train' lub 'val'
    labels_subdir: 'train' lub 'val'
    """
    import cv2
    os.makedirs(f"data/labels/{labels_subdir}", exist_ok=True)
    for r in records:
        fname = os.path.basename(r['filename'])
        img_path = f"data/images/{images_subdir}/{fname}"
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        x1, y1, x2, y2 = r['bbox']
        x_c = ((x1 + x2) / 2) / w
        y_c = ((y1 + y2) / 2) / h
        bw  = (x2 - x1) / w
        bh  = (y2 - y1) / h
        label_path = f"data/labels/{labels_subdir}/{os.path.splitext(fname)[0]}.txt"
        with open(label_path, 'w') as f:
            f.write(f"0 {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")

def prepare_dataset(xml_path: str,
                    images_src_dir: str,
                    images_dst_dir: str = 'data/images',
                    labels_dst_dir: str = 'data/labels',
                    test_size: float = 0.3,
                    seed: int = 42):
    """
    1) Parsuje XML z adnotacjami
    2) Dzieli na train/val
    3) Tworzy katalogi:
         data/images/train, data/images/val,
         data/labels/train, data/labels/val
    4) Kopiuje obrazy
    5) Generuje .txt z bbox w formacie YOLO
    Zwraca (train_records, val_records)
    """
    # 1) Parsowanie adnotacji
    records = parse_annotations(xml_path, images_src_dir)
    # 2) Podział
    train_recs, val_recs = split_data(records, test_size=test_size, seed=seed)
    # 3) Katalogi
    for sub in ('train','val'):
        os.makedirs(f"{images_dst_dir}/{sub}", exist_ok=True)
        os.makedirs(f"{labels_dst_dir}/{sub}", exist_ok=True)
    # 4) Kopiowanie obrazów
    for r in train_recs:
        shutil.copy(r['filename'], f"{images_dst_dir}/train/")
    for r in val_recs:
        shutil.copy(r['filename'], f"{images_dst_dir}/val/")
    # 5) Generowanie etykiet
    def gen(records, imgs_sub, lbls_sub):
        for r in records:
            fname = os.path.basename(r['filename'])
            img = cv2.imread(f"{images_dst_dir}/{imgs_sub}/{fname}")
            h, w = img.shape[:2]
            x1,y1,x2,y2 = r['bbox']
            x_c = ((x1+x2)/2)/w
            y_c = ((y1+y2)/2)/h
            bw  = (x2-x1)/w
            bh  = (y2-y1)/h
            out = f"{labels_dst_dir}/{lbls_sub}/{os.path.splitext(fname)[0]}.txt"
            with open(out,'w') as f:
                f.write(f"0 {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")
    gen(train_recs, 'train', 'train')
    gen(val_recs,   'val',   'val')
    print(f"✔ Prepared: {len(train_recs)} train, {len(val_recs)} val")
    return train_recs, val_recs
