import os
import xml.etree.ElementTree as ET
from config import ANNOT_PATH, YOLO_LABELS_DIR, CSV_PATH

os.makedirs(YOLO_LABELS_DIR, exist_ok=True)
tree = ET.parse(ANNOT_PATH)
root = tree.getroot()

rows = []
for img in root.findall('image'):
    fname = img.attrib['name']
    width, height = int(img.attrib['width']), int(img.attrib['height'])
    yolo_lines = []
    for box in img.findall('box'):
        xtl, ytl, xbr, ybr = map(float, [box.attrib['xtl'], box.attrib['ytl'], box.attrib['xbr'], box.attrib['ybr']])
        x_c = (xtl + xbr) / 2 / width
        y_c = (ytl + ybr) / 2 / height
        w = (xbr - xtl) / width
        h = (ybr - ytl) / height
        plate_num = box.find("./attribute[@name='plate number']").text
        yolo_lines.append(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
        rows.append(f"{fname},{int(xtl)},{int(ytl)},{int(xbr)},{int(ybr)},{plate_num}")
    out_txt = os.path.join(YOLO_LABELS_DIR, fname.replace('.jpg', '.txt'))
    with open(out_txt, 'w') as f:
        f.write('\n'.join(yolo_lines))
with open(CSV_PATH, 'w') as f:
    f.write("fname,xtl,ytl,xbr,ybr,plate\n")
    f.write('\n'.join(rows))
print("Konwersja zakończona!")
