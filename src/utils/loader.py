import os
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

def load_cvat_xml(xml_path, images_dir, test_ratio=0.2, random_seed=42):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    data = []
    for image_tag in root.iter('image'):
        img_name = image_tag.attrib['name']
        img_path = os.path.join(images_dir, img_name)
        img_w = int(image_tag.attrib['width'])
        img_h = int(image_tag.attrib['height'])
        for box in image_tag.iter('box'):
            label = box.attrib['label']
            if label != "plate":
                continue
            xmin = float(box.attrib['xtl'])
            ymin = float(box.attrib['ytl'])
            xmax = float(box.attrib['xbr'])
            ymax = float(box.attrib['ybr'])
            plate_text = box.attrib.get('ocr_text', '').replace(" ", "").replace("-", "")
            data.append([img_path, [xmin, ymin, xmax, ymax], plate_text, img_w, img_h])
    train_set, test_set = train_test_split(
        data, test_size=test_ratio, random_state=random_seed)
    return train_set, test_set
