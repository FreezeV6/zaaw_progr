import os
import shutil
import xml.etree.ElementTree as ET
import random

# ------ Ustawienia splitu ------
train_ratio = 0.8   # 80% danych do train, reszta do test
seed = 42           # dla powtarzalności

# ------ Ścieżki ------
output_dir        = '../dataset'
images_source_dir = '../data/photos'
xml_path          = '../data/annotations.xml'

# Katalogi wyjściowe
images_dir = os.path.join(output_dir, 'images')
labels_dir = os.path.join(output_dir, 'labels')

# Tworzymy podfoldery train/test
for base in (images_dir, labels_dir):
    for split in ('train', 'test'):
        os.makedirs(os.path.join(base, split), exist_ok=True)

# Parsujemy XML i zbieramy listę wszystkich plików
tree = ET.parse(xml_path)
root = tree.getroot()
all_filenames = [img.get('name') for img in root.findall('image')]

# Losowe przetasowanie i podział
random.seed(seed)
random.shuffle(all_filenames)
split_idx = int(len(all_filenames) * train_ratio)
train_set = set(all_filenames[:split_idx])
test_set  = set(all_filenames[split_idx:])

# Przetwarzamy każdy obraz
for image in root.findall('image'):
    filename = image.get('name')
    width  = float(image.get('width'))
    height = float(image.get('height'))

    # Wybieramy split
    if filename in train_set:
        split = 'train'
    else:
        split = 'test'

    # Kopiujemy obraz
    src = os.path.join(images_source_dir, filename)
    dst = os.path.join(images_dir, split, filename)
    shutil.copyfile(src, dst)

    # Tworzymy plik .txt z YOLO-label
    txt_path = os.path.join(labels_dir, split,
                            os.path.splitext(filename)[0] + '.txt')
    with open(txt_path, 'w') as f:
        for box in image.findall('box'):
            label = box.get('label')
            if label != 'plate':
                continue

            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))

            # konwersja do YOLO format: x_center, y_center, w, h (wszystko znormalizowane)
            x_center = ((xtl + xbr) / 2) / width
            y_center = ((ytl + ybr) / 2) / height
            box_width  = (xbr - xtl) / width
            box_height = (ybr - ytl) / height

            class_id = 0  # plate
            f.write(f"{class_id} "
                    f"{x_center:.6f} {y_center:.6f} "
                    f"{box_width:.6f} {box_height:.6f}\n")
