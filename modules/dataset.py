import os
import shutil
import xml.etree.ElementTree as ET
import logging
from math import floor, ceil

def load_annotations(xml_path):
    """
    Wczytuje anotacje z pliku CVAT XML i zwraca listę słowników z informacjami:
    'filename': nazwa pliku obrazu,
    'width', 'height': wymiary obrazu,
    'bbox': (xmin, ymin, xmax, ymax) jako float,
    'plate_text': tekst tablicy rejestracyjnej (string).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    data = []
    for image in root.findall('image'):
        file_name = image.get('name')
        img_width = int(image.get('width'))
        img_height = int(image.get('height'))
        # Iteracja po wszystkich obiektach 'box' (zakładamy label="plate")
        for box in image.findall('box'):
            label = box.get('label')
            if label != 'plate':
                continue  # ignoruj inne etykiety jeśli istnieją
            xmin = float(box.get('xtl'))
            ymin = float(box.get('ytl'))
            xmax = float(box.get('xbr'))
            ymax = float(box.get('ybr'))
            plate_text = None
            # Pobierz atrybut 'plate number' (tekst tablicy) jeśli istnieje
            for attrib in box.findall('attribute'):
                if attrib.get('name') == 'plate number':
                    plate_text = attrib.text.strip() if attrib.text else ""
                    break
            data.append({
                'filename': file_name,
                'width': img_width,
                'height': img_height,
                'bbox': (xmin, ymin, xmax, ymax),
                'plate_text': plate_text
            })
    return data

def split_data(data_list, train_ratio=0.7):
    """
    Dzieli listę danych na część treningową i testową zgodnie z podanym ułamkiem.
    Używa stałego seed dla losowości, aby podział był deterministyczny.
    Zwraca tuple: (train_list, test_list).
    """
    import random
    random.seed(42)
    # Skopiuj listę i wylosuj kolejność
    data_copy = data_list.copy()
    random.shuffle(data_copy)
    train_size = int(train_ratio * len(data_copy))
    train_list = data_copy[:train_size]
    test_list = data_copy[train_size:]
    return train_list, test_list

def prepare_yolo_dataset(train_data, test_data, base_dir="dataset"):
    """
    Przygotowuje strukturę folderów i pliki z adnotacjami w formacie YOLO.
    Tworzy foldery train/ i test/ wraz z podfolderami images/ i labels/.
    Kopiuje obrazy do odpowiednich folderów i zapisuje pliki .txt z bbox.
    Zwraca ścieżkę do wygenerowanego pliku data.yaml potrzebnego do trenowania YOLO.
    """
    # Ustaw ścieżki folderów dla train i test
    train_img_dir = os.path.join(base_dir, "train", "images")
    train_lbl_dir = os.path.join(base_dir, "train", "labels")
    test_img_dir = os.path.join(base_dir, "test", "images")
    test_lbl_dir = os.path.join(base_dir, "test", "labels")
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)
    os.makedirs(test_lbl_dir, exist_ok=True)
    # Funkcja pomocnicza do zapisu jednego zbioru (train/test)
    def save_dataset_subset(data_list, img_dir, lbl_dir):
        for item in data_list:
            filename = item['filename']
            src_path = os.path.join(base_dir, "photos", filename)
            dst_path = os.path.join(img_dir, filename)
            # Kopiuj plik obrazu do folderu docelowego
            shutil.copy2(src_path, dst_path)
            # Konwertuj bbox do formatu YOLO i zapisz etykietę
            x_min, y_min, x_max, y_max = item['bbox']
            img_w = item['width']
            img_h = item['height']
            # Środek i wymiary normalizowane [0,1]
            cx = ((x_min + x_max) / 2.0) / img_w
            cy = ((y_min + y_max) / 2.0) / img_h
            w = (x_max - x_min) / img_w
            h = (y_max - y_min) / img_h
            # Zapisz do pliku .txt z 6 cyframi po przecinku
            label_path = os.path.join(lbl_dir, os.path.splitext(filename)[0] + ".txt")
            with open(label_path, 'w') as f:
                f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    # Wyczyść ewentualne poprzednie dane w folderach (jeśli istnieją)
    for d in (train_img_dir, train_lbl_dir, test_img_dir, test_lbl_dir):
        # Usuń istniejące pliki w folderze
        for fname in os.listdir(d):
            file_path = os.path.join(d, fname)
            if os.path.isfile(file_path):
                os.remove(file_path)
    # Zapisz dane treningowe i testowe
    save_dataset_subset(train_data, train_img_dir, train_lbl_dir)
    save_dataset_subset(test_data, test_img_dir, test_lbl_dir)
    # Zdefiniuj konfigurację YAML dla YOLOv8
    data_yaml_path = os.path.join(base_dir, "data.yaml")
    num_classes = 1
    names = ["plate"]
    # Przygotuj treść YAML
    yaml_content = f"""
path: {base_dir}
train: {os.path.join('train', 'images')}
val: {os.path.join('test', 'images')}
nc: {num_classes}
names: {names}
"""
    # Zapisz plik data.yaml
    with open(data_yaml_path, 'w') as f:
        f.write(yaml_content.strip())
    logging.info(f"YOLO dataset prepared under folder '{base_dir}'.")
    logging.info(f"Training images: {len(train_data)}, Test images: {len(test_data)}")
    return data_yaml_path
