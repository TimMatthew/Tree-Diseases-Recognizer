from torch import float32
from torchvision.transforms import v2

PATH_TRAIN = r'C:\Users\tymop\OneDrive\Робочий стіл\Курсова\dataset\train'
PATH_VALID = r'C:\Users\tymop\OneDrive\Робочий стіл\Курсова\dataset\valid'
PATH_TEST = r'C:\Users\tymop\OneDrive\Робочий стіл\Курсова\dataset\test'
DATASET_PATH = r'C:\Users\tymop\OneDrive\Робочий стіл\Курсова\dataset'

last_trainable_layers = 2
IMAGES = 5733
LR = 1e-3
DECAY = 1e-4
BATCH_SIZE = 32
RESIZE_B0 = 224  # Default EfficientNet B0 input image size
RESIZE_B3 = 300  # Default EfficientNet B3 input image size
NUM_EPOCHS = 10

AUGMENT_TRANSFORM = v2.Compose([
    v2.Resize((RESIZE_B3, RESIZE_B3)),
    v2.ToImage(),
    v2.ToDtype(float32, scale=True),
    v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # ImageNet EfficientNet parameters
    v2.RandomRotation(30),
    # v2.RandomHorizontalFlip(),
])

NO_AUGMENT_TRANSFORM = v2.Compose([
    v2.Resize((RESIZE_B3, RESIZE_B3)),
    v2.ToImage(),
    v2.ToDtype(float32, scale=True),
    v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet EfficientNet parameters
])
