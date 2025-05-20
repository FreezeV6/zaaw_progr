# Paths
XML_PATH = 'dataset/annotations.xml'
IMAGES_DIR = 'dataset/photos'

# Model
CNN_MODEL_PATH = 'detector/plate_resnet.pth'

# Input shape for CNN
CROP_SIZE = (224, 64)  # szerokość x wysokość

# Test/train split
TEST_RATIO = 0.2
RANDOM_SEED = 42

# OCR settings

# Training params
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 0.001
