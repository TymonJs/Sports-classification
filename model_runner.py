import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss, classification_report
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # type: ignore

from models import create_knn_model, create_simple_nn_model, create_cnn_model

def run_knn(n_neighbors, data):

    X_train, y_train, X_test, y_test = data
    knn_model = create_knn_model(n_neighbors=n_neighbors)

    print(f"Training k-NN({n_neighbors})...")
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)

    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    print(f"k-NN Model Accuracy: {(accuracy_knn*100):.2f}%")

    y_pred_proba = knn_model.predict_proba(X_test)
    log_loss_value = log_loss(y_test, y_pred_proba)
    print(f"k-NN Model Log Loss: {log_loss_value:.2f}\n")


    # print("Classification Report for k-NN:")
    # print(classification_report(y_test, y_pred_knn))
    return accuracy_knn

def run_simple_nn(data):
    
    X_train, y_train, X_test, y_test = data
    num_classes = len(np.unique(y_train))
    nn_model = create_simple_nn_model(input_shape=(X_train.shape[1],), num_classes=num_classes)

    print("Training Simple NN...")
    nn_model.fit(X_train, y_train)
    y_pred_nn = nn_model.predict(X_test)
    accuracy_nn = accuracy_score(y_test, y_pred_nn)

    print(f"\nSimple NN Model Accuracy: {(accuracy_nn*100):.2f}")
    # print("Classification Report for Simple NN:")
    # print(classification_report(y_test, y_pred_nn))

    return accuracy_nn

def run_cnn(cnn_model, generators, EPOCHS_CNN) :

    train_generator, validation_generator = generators

    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        min_delta=0.001,
        patience=7,
        verbose=1,
        restore_best_weights=True
    )

    checkpoint = ModelCheckpoint(
        'cnn_model_best.keras',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    
    print("\nTraining CNN...")
    history = cnn_model.fit(
        train_generator,
        epochs=EPOCHS_CNN,
        validation_data=validation_generator,
        callbacks=[
            early_stopping,
            checkpoint
        ]
    )
    
    loss, accuracy = cnn_model.evaluate(validation_generator)
    print(f"\nCNN Model Accuracy: {(accuracy*100):.2f}")

    return accuracy, loss, history