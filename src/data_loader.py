import os
import cv2
import xml.etree.ElementTree as ET

def load_cvat_xml(xml_path, images_dir):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    data = []
    for image in root.findall('image'):
        fname = image.get('name')
        img_path = os.path.join(images_dir, fname)
        if not os.path.exists(img_path):
            continue
        box_elem = image.find('box')
        if box_elem is not None:
            # YOLO format: x_center, y_center, width, height (normalized)
            xtl = float(box_elem.get('xtl'))
            ytl = float(box_elem.get('ytl'))
            xbr = float(box_elem.get('xbr'))
            ybr = float(box_elem.get('ybr'))
            width = xbr - xtl
            height = ybr - ytl
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            x_center = (xtl + width / 2) / w
            y_center = (ytl + height / 2) / h
            w_norm = width / w
            h_norm = height / h
            plate = ""
            for attr in image.findall('attribute'):
                if attr.get('name') == 'license_plate':
                    plate = attr.text.strip() if attr.text else ""
            data.append((img_path, [x_center, y_center, w_norm, h_norm], plate))
    return data
