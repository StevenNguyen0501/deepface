import os
import cv2
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tqdm import tqdm  # Import tqdm for progress bar

# Specify the path to the directory containing your images
data_dir = 'database/BinhTD'

# Create an ImageDataGenerator with specified augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# List all image files in the directory (including both ".jpg" and ".jpeg" files)
img_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.jpg') or f.lower().endswith('.jpeg') or f.lower().endswith('.png')]

# Specify the number of augmented images you want to generate for each original image
augmentation_factor = 3

# Create a directory to store augmented images
augmented_dir = data_dir + '_gen'
os.makedirs(augmented_dir, exist_ok=True)

# Perform data augmentation for each image with tqdm
for img_file in tqdm(img_files, desc="Augmenting Images", unit="image"):
    img_path = os.path.join(data_dir, img_file)
    img = image.load_img(img_path)
    x = image.img_to_array(img)
    x = x.reshape((1,) + x.shape)

    # Generate augmented images and save to the augmented directory
    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir=augmented_dir, save_prefix=img_file.split('.')[0], save_format='jpeg'):
        i += 1
        if i >= augmentation_factor:
            break  # break the loop after generating the desired number of augmented images

# # Visualize some of the augmented images using OpenCV
# augmented_files = [f for f in os.listdir(augmented_dir) if f.lower().endswith('.jpg') or f.lower().endswith('.jpeg') or f.lower().endswith('.png')]

# for i, augmented_file in enumerate(augmented_files):
#     img = cv2.imread(os.path.join(augmented_dir, augmented_file))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for display

#     # Display the augmented image
#     cv2.imshow(f'Augmented Image {i + 1}', img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# Copy all files from augmented_dir to data_dir
shutil.copytree(augmented_dir, data_dir, dirs_exist_ok=True)

# Remove augmented_dir
shutil.rmtree(augmented_dir)
print("Data augmentation complete. Augmented images copied to data_dir.")
