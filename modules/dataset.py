import os
import random
import shutil
import xml.etree.ElementTree as ET


def load_annotations(xml_path):
    """
    Wczytuje adnotacje z pliku CVAT XML i zwraca listę słowników z informacjami:
    {'filename', 'width', 'height', 'bboxes', 'texts'}
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    images_data = []
    for img in root.findall('image'):
        filename = img.get('name')
        width = float(img.get('width'))
        height = float(img.get('height'))
        bboxes = []
        texts = []
        for box in img.findall('box'):
            if box.get('label') and box.get('label').lower() in ['plate', 'license_plate', 'licence_plate', 'tablica']:
                xtl = float(box.get('xtl'))
                ytl = float(box.get('ytl'))
                xbr = float(box.get('xbr'))
                ybr = float(box.get('ybr'))
                bboxes.append((xtl, ytl, xbr, ybr))
                # atrybut 'plate number'
                plate_text = ''
                for attr in box.findall('attribute'):
                    if attr.text:
                        plate_text = attr.text.strip()
                        break
                texts.append(plate_text)
        if bboxes:
            images_data.append({'filename': filename,
                                'width': width,
                                'height': height,
                                'bboxes': bboxes,
                                'texts': texts})
    return images_data


def split_data(images_data, train_ratio=0.7):
    """
    Dzieli listę danych na train i test (domyślnie 70/30).
    """
    random.seed(42)
    random.shuffle(images_data)
    split = int(train_ratio * len(images_data))
    return images_data[:split], images_data[split:]


def prepare_yolo_dataset(train_data, test_data, base_dir='data'):
    """
    Tworzy foldery data/train/{images,labels} i data/test/{images,labels}, kopiuje obrazy
    oraz zapisuje YOLO-format labelki.
    """
    # Usuń istniejące dane split
    for subset in ['train', 'test']:
        for sub in ['images', 'labels']:
            d = os.path.join(base_dir, subset, sub)
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)
    # Funkcja pomocnicza
    def save_subset(data, subset):
        img_dir = os.path.join(base_dir, subset, 'images')
        lbl_dir = os.path.join(base_dir, subset, 'labels')
        for item in data:
            fname = item['filename']
            src = os.path.join(base_dir, 'photos', fname)
            dst = os.path.join(img_dir, fname)
            shutil.copy2(src, dst)
            lbl_path = os.path.join(lbl_dir, os.path.splitext(fname)[0] + '.txt')
            with open(lbl_path, 'w') as f:
                for (xtl, ytl, xbr, ybr) in item['bboxes']:
                    w = xbr - xtl
                    h = ybr - ytl
                    cx = xtl + w/2
                    cy = ytl + h/2
                    cxn = cx / item['width']
                    cyn = cy / item['height']
                    wn = w / item['width']
                    hn = h / item['height']
                    f.write(f"0 {cxn:.6f} {cyn:.6f} {wn:.6f} {hn:.6f}\n")
    save_subset(train_data, 'train')
    save_subset(test_data, 'test')
    # Zapis data.yaml
    yaml_path = os.path.join(base_dir, 'data.yaml')
    with open(yaml_path, 'w') as yf:
        yf.write('train: train/images\n')
        yf.write('val: test/images\n')
        yf.write('nc: 1\n')
        yf.write("names: ['plate']\n")
    return yaml_path


def prepare_dataset():
    """
    Wczytuje XML, dzieli dane, przygotowuje YOLO dataset.
    Zwraca słownik {'train': train_data, 'test': test_data}.
    """
    xml_path = os.path.join('data', 'annotations.xml')
    data_list = load_annotations(xml_path)
    train_data, test_data = split_data(data_list)
    prepare_yolo_dataset(train_data, test_data, base_dir='data')
    return {'train': train_data, 'test': test_data}
