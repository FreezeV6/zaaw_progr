# Paths
XML_PATH = 'dataset/annotations.xml'
IMAGES_DIR = 'dataset/photos'

# Model
CNN_MODEL_PATH = 'src/detector/plate_cnn.pth'

# Input shape for CNN
CROP_SIZE = (128, 64)  # szerokość x wysokość

# Test/train split
TEST_RATIO = 0.3
RANDOM_SEED = 42

# OCR settings
OCR_LANGS = ['pl']

# Training params
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.001
