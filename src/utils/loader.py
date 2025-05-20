import os
import xml.etree.ElementTree as ET
import random

def convert_box_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    x_center = (xmin + xmax) / 2 / img_w
    y_center = (ymin + ymax) / 2 / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h
    return x_center, y_center, width, height

def load_cvat_xml(xml_path, images_dir, test_ratio=0.3, seed=42):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    data = []
    for image in root.findall('image'):
        name = image.attrib['name']
        img_path = os.path.join(images_dir, name)
        if not os.path.exists(img_path):
            continue
        img_w = float(image.attrib['width'])
        img_h = float(image.attrib['height'])
        for box in image.findall('box'):
            xtl = float(box.attrib['xtl'])
            ytl = float(box.attrib['ytl'])
            xbr = float(box.attrib['xbr'])
            ybr = float(box.attrib['ybr'])
            plate_text = ""
            for attr in box.findall('attribute'):
                if attr.attrib['name'] == "plate number":
                    plate_text = attr.text.strip() if attr.text else ""
            box_yolo = convert_box_to_yolo(xtl, ytl, xbr, ybr, img_w, img_h)
            data.append((img_path, box_yolo, plate_text, img_w, img_h))
    # Shuffle i split
    random.seed(seed)
    random.shuffle(data)
    split_idx = int(len(data) * (1 - test_ratio))
    train_set = data[:split_idx]
    test_set = data[split_idx:]
    return train_set, test_set
