import os
import tensorflow as tf
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
import numpy as np
import matplotlib.pyplot as plt

MODEL_PATH = 'cnn_model_best.keras'
IMAGE_PATH = 'data/train/skydiving/012.jpg'
OUTPUT_DIR = 'cnn_layers_output'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

model = tf.keras.models.load_model(MODEL_PATH)
print("Model wczytany pomyślnie.")
model.summary()

layer_outputs = [layer.output for layer in model.layers]
layer_names = [layer.name for layer in model.layers]
visualization_model = Model(inputs=model.inputs, outputs=layer_outputs)

img = load_img(IMAGE_PATH, target_size=(224, 224))
img_tensor = img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.0

layer_activations = visualization_model.predict(img_tensor)

print(f"\nGenerowanie wizualizacji dla {len(layer_names)} warstw...")

for layer_name, layer_activation in zip(layer_names, layer_activations):
    print(f"  Przetwarzanie warstwy: {layer_name} | Kształt wyjścia: {layer_activation.shape}")

    if len(layer_activation.shape) == 4:
        num_features = layer_activation.shape[-1]
        n_cols = 8
        n_rows = (num_features + n_cols - 1) // n_cols
        
        plt.figure(figsize=(n_cols * 1.5, n_rows * 1.5))
        plt.suptitle(f'Wynik warstwy: {layer_name}\nKształt: {layer_activation.shape}', fontsize=16)

        for i in range(num_features):
            plt.subplot(n_rows, n_cols, i + 1)
            channel_image = layer_activation[0, :, :, i]
            if channel_image.std() > 1e-6:
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            plt.imshow(channel_image, cmap='viridis')
            plt.axis('off')

    elif len(layer_activation.shape) == 2:
        plt.figure(figsize=(12, 6))
        plt.suptitle(f'Wynik warstwy: {layer_name}\nKształt: {layer_activation.shape}', fontsize=16)
        
        activations = layer_activation[0]
        plt.plot(activations)
        plt.xlabel('Numer cechy / neuronu')
        plt.ylabel('Wartość aktywacji')

    else:
        print(f"    Pominięto wizualizację dla warstwy {layer_name} (kształt nie jest 2D ani 4D).")
        continue

    output_path = os.path.join(OUTPUT_DIR, f'{layer_name}_output.png')
    plt.savefig(output_path)
    plt.close()
    print(f"    Zapisano wizualizację do: {output_path}")

print("\nZakończono.")