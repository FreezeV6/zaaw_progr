import xml.etree.ElementTree as ET
import os

def load_cvat_xml(xml_path, images_dir, test_ratio, seed):
    import random
    random.seed(seed)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    images = []
    for img in root.findall(".//image"):
        img_name = img.attrib['name']
        img_path = os.path.join(images_dir, img_name)
        if not os.path.isfile(img_path):
            continue
        img_w = int(img.attrib['width'])
        img_h = int(img.attrib['height'])
        boxes = img.findall("box")
        if len(boxes) == 0:
            continue
        box = boxes[0]
        xmin = float(box.attrib['xtl'])
        ymin = float(box.attrib['ytl'])
        xmax = float(box.attrib['xbr'])
        ymax = float(box.attrib['ybr'])
        x_c = (xmin + xmax) / 2 / img_w
        y_c = (ymin + ymax) / 2 / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h
        label = box.attrib.get('label', '')
        # Tablica rejestracyjna (może pusta string)
        plate = ""
        if 'plate' in box.attrib:
            plate = box.attrib['plate']
        images.append((img_path, [x_c, y_c, w, h], plate, img_w, img_h))
    random.shuffle(images)
    split = int(len(images) * (1 - test_ratio))
    train_set = images[:split]
    test_set = images[split:]
    return train_set, test_set
