import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, save_img # type: ignore

SAMPLE_IMAGE_PATH = 'data/train/archery/070.jpg' 
OUTPUT_DIR = 'augmented_images_output'
NUM_IMAGES_TO_GENERATE = 9

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Utworzono folder: {OUTPUT_DIR}")

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

try:
    img = load_img(SAMPLE_IMAGE_PATH)
except FileNotFoundError:
    print(f"BŁĄD: Nie znaleziono pliku obrazu pod ścieżką: {SAMPLE_IMAGE_PATH}")
    exit()

x = img_to_array(img)
x = x.reshape((1,) + x.shape)
print(f"Generowanie {NUM_IMAGES_TO_GENERATE} wariantów obrazu: {SAMPLE_IMAGE_PATH}")

i = 0
for batch in train_datagen.flow(x, batch_size=1):

    augmented_image_array = batch[0]
    
    output_path = os.path.join(OUTPUT_DIR, f'augmented_{i+1}.png')
    save_img(output_path, augmented_image_array)
    
    i += 1
    if i >= NUM_IMAGES_TO_GENERATE:
        break 
