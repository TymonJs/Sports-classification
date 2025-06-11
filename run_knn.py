from model_runner import run_knn
from data_loader import load_and_preprocess_for_nn_knn
import numpy as np
from consts import *

IMG_SIZE = (64, 64)

# knn data loading
X_train, y_train = load_and_preprocess_for_nn_knn(TRAIN_DIR, img_size=IMG_SIZE)
X_test, y_test = load_and_preprocess_for_nn_knn(TEST_DIR, img_size=IMG_SIZE)

num_classes = len(np.unique(y_train))
data = (X_train, y_train, X_test, y_test)
print(f"KNN data loaded: {len(X_train)} training samples, {len(X_test)} testing samples.")
print(f"Number of classes: {num_classes}\n")

# knn run
knn = [run_knn(i,data) for i in range(1,15)]