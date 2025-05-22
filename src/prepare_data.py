import os
import random, math, shutil
import xml.etree.ElementTree as ET


def prepare_dataset(data_dir='data'):
    # Download dataset from Kaggle if not already downloaded
    ann_file = os.path.join(data_dir, 'annotations.xml')
    if not os.path.exists(ann_file):
        try:
            import kaggle
        except ImportError:
            print("Kaggle API not installed. Please install `kaggle` or download data manually.")
            return
        # Authenticate and download the dataset (unzips into data_dir)
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            'piotrstefaskiue/poland-vehicle-license-plate-dataset',
            path=data_dir, unzip=True
        )
        # If the dataset files are nested inside a subfolder, adjust ann_file path
        if not os.path.exists(ann_file):
            for root, dirs, files in os.walk(data_dir):
                if 'annotations.xml' in files:
                    ann_file = os.path.join(root, 'annotations.xml')
                    break
        if not os.path.exists(ann_file):
            raise FileNotFoundError("annotations.xml not found. Please check dataset download.")

    # Parse the XML annotations to extract image file names, plate text, and bounding boxes
    tree = ET.parse(ann_file)
    root = tree.getroot()
    entries = []
    for img_elem in root.findall('image'):
        file_name = img_elem.get('name')
        w = int(float(img_elem.get('width')))
        h = int(float(img_elem.get('height')))
        box_elem = img_elem.find('box')
        xtl = float(box_elem.get('xtl'));
        ytl = float(box_elem.get('ytl'))
        xbr = float(box_elem.get('xbr'));
        ybr = float(box_elem.get('ybr'))
        plate_text = box_elem.find('attribute').text.strip()
        entries.append({
            "file": file_name,
            "width": w, "height": h,
            "bbox": (xtl, ytl, xbr, ybr),
            "plate": plate_text
        })

    # Shuffle and split data: at least 30% for testing
    random.seed(42)
    random.shuffle(entries)
    test_count = math.ceil(0.3 * len(entries))
    test_entries = entries[:test_count]
    train_entries = entries[test_count:]

    # Create directories for split data
    os.makedirs(os.path.join(data_dir, 'images/train'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'images/test'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'labels/train'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'labels/test'), exist_ok=True)

    # Copy images to train/test folders and write YOLO label files
    for entry in train_entries:
        src_path = os.path.join(os.path.dirname(ann_file), entry["file"])
        dst_path = os.path.join(data_dir, 'images/train', entry["file"])
        shutil.copy(src_path, dst_path)
        # YOLO label (class_id x_center y_center width height in relative terms)
        xtl, ytl, xbr, ybr = entry["bbox"]
        w, h = entry["width"], entry["height"]
        x_center = (xtl + xbr) / 2 / w
        y_center = (ytl + ybr) / 2 / h
        bw = (xbr - xtl) / w
        bh = (ybr - ytl) / h
        label_file = os.path.join(data_dir, 'labels/train', entry["file"].rsplit('.', 1)[0] + '.txt')
        with open(label_file, 'w') as f:
            f.write(f"0 {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")

    for entry in test_entries:
        src_path = os.path.join(os.path.dirname(ann_file), entry["file"])
        dst_path = os.path.join(data_dir, 'images/test', entry["file"])
        shutil.copy(src_path, dst_path)
        # Write label file for potential detection evaluation (not used for OCR, but for completeness)
        xtl, ytl, xbr, ybr = entry["bbox"]
        w, h = entry["width"], entry["height"]
        x_center = (xtl + xbr) / 2 / w
        y_center = (ytl + ybr) / 2 / h
        bw = (xbr - xtl) / w
        bh = (ybr - ytl) / h
        label_file = os.path.join(data_dir, 'labels/test', entry["file"].rsplit('.', 1)[0] + '.txt')
        with open(label_file, 'w') as f:
            f.write(f"0 {x_center:.6f} {y_center:.66f} {bw:.6f} {bh:.6f}\n")

    # Save ground truth for test set (filename, plate text, bbox) for evaluation use
    gt_path = os.path.join(data_dir, 'test_ground_truth.txt')
    with open(gt_path, 'w') as f:
        for entry in test_entries:
            xtl, ytl, xbr, ybr = entry["bbox"]
            f.write(f"{entry['file']},{entry['plate']},{xtl},{ytl},{xbr},{ybr}\n")

    # (Optional) Create a data.yaml file for YOLO training configuration
    yaml_content = (
        f"path: {data_dir}\n"
        f"train: images/train\n"
        f"val: images/test\n"
        f"names: ['plate']\n"
    )
    with open(os.path.join(data_dir, "data.yaml"), 'w') as yf:
        yf.write(yaml_content)

    print(f"Dataset prepared: {len(train_entries)} training images, {len(test_entries)} test images.")
