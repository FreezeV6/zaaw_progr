# split_and_convert.py
from src.data_utils import parse_annotations, split_data, write_splits, convert_to_yolo

# 1) Parsujemy adnotacje i obrazy
records = parse_annotations(
    xml_path='data/annotations.xml',
    images_dir='data/images'
)

# 2) Dzielimy 70/30
train_records, test_records = split_data(records, test_size=0.3, seed=42)

# 3) Zapisujemy listy plików
write_splits(train_records, 'src/splits/train.txt')
write_splits(test_records,  'src/splits/test.txt')

# 4) Generujemy etykiety YOLO w formacie .txt
convert_to_yolo(train_records, 'data/labels/train')
convert_to_yolo(test_records,  'data/labels/val')

print(f'Zapisano {len(train_records)} plików do train i {len(test_records)} do test.')
