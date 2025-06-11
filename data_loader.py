import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

def load_and_preprocess_for_nn_knn(data_dir, img_size=(64, 64)):

    data = []
    labels = []
    
    # Get the class names from the subdirectories
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    label_map = {name: i for i, name in enumerate(class_names)}
    
    print(f"Loading images from: {data_dir}")
    # print(f"Class mapping: {label_map}")

    for sport_name in class_names:
        sport_dir = os.path.join(data_dir, sport_name)
        if not os.path.isdir(sport_dir):
            continue
            
        for img_file in os.listdir(sport_dir):
            img_path = os.path.join(sport_dir, img_file)
            try:
                with Image.open(img_path) as img:
                    # Convert to RGB, resize, and normalize
                    img = img.convert('RGB').resize(img_size)
                    img_array = np.asarray(img, dtype=np.float32) / 255.0
                    
                    # Flatten the image for NN and k-NN
                    data.append(img_array.flatten())
                    labels.append(label_map[sport_name])
            except Exception as e:
                print(f"Could not read image {img_path}: {e}")

    return np.array(data), np.array(labels)


def get_cnn_data_generators(train_dir, test_dir, img_size=(128, 128), batch_size=32):

    print("\nSetting up CNN Data Generators...")
    
    # Training data generator with data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Test data generator (only rescaling)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    validation_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False # Important for evaluation
    )
    
    print(f"Found {train_generator.samples} images belonging to {train_generator.num_classes} classes for training.")
    print(f"Found {validation_generator.samples} images belonging to {validation_generator.num_classes} classes for testing.")
    
    return train_generator, validation_generator