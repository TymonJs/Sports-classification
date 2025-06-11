from model_runner import run_cnn
from data_loader import get_cnn_data_generators
from consts import *
from models import create_cnn_model
from plot_utils import *


IMG_SIZE_CNN = (224, 224)
BATCH_SIZE_CNN = 32
EPOCHS_CNN = 100


#cnn run
train_generator, validation_generator = get_cnn_data_generators(
        TRAIN_DIR, TEST_DIR, img_size=IMG_SIZE_CNN, batch_size=BATCH_SIZE_CNN
)

cnn_model = create_cnn_model(
        input_shape=(IMG_SIZE_CNN[0], IMG_SIZE_CNN[1], 3), 
        num_classes=train_generator.num_classes
)
generators = (train_generator, validation_generator)

result = run_cnn(cnn_model,generators, EPOCHS_CNN)

plot_cnn(result, "regular_cnn")
plot_cnn(result, "regular_cnn_dark", True)

