from model_runner import run_simple_nn
from data_loader import load_and_preprocess_for_nn_knn
import numpy as np
from consts import *

IMG_SIZE = (64, 64)

# mlp data loading
X_train, y_train = load_and_preprocess_for_nn_knn(TRAIN_DIR, img_size=IMG_SIZE)
X_test, y_test = load_and_preprocess_for_nn_knn(TEST_DIR, img_size=IMG_SIZE)

num_classes = len(np.unique(y_train))
data = (X_train, y_train, X_test, y_test)
print(f"MLP data loaded: {len(X_train)} training samples, {len(X_test)} testing samples.")
print(f"Number of classes: {num_classes}\n")

#mlp run
mlp = run_simple_nn(data)