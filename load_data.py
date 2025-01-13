import cv2
import os
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
from tqdm import tqdm

# Directory paths
train_data_dir = 'data/WIDER_train'
val_data_dir = 'data/WIDER_val'
test_data_dir = 'data/WIDER_test'

# Function to load images and annotations
def load_data(data_dir):
    images = []
    annotations = []
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.jpg'):
                image_path = os.path.join(subdir, file)
                annotation_path = image_path.replace('.jpg', '.txt')
                img = cv2.imread(image_path)
                faces = []
                if os.path.exists(annotation_path):
                    with open(annotation_path, 'r') as f:
                        for line in f:
                            x, y, w, h = map(int, line.strip().split())
                            faces.append([x, y, w, h])
                images.append(img)
                annotations.append(faces)
    return images, annotations

# Load data
train_images, train_annotations = load_data(train_data_dir)
val_images, val_annotations = load_data(val_data_dir)
test_images, test_annotations = load_data(test_data_dir)

# Example of displaying an image with annotations
def display_image_with_annotations(img, annotations):
    for (x, y, w, h) in annotations:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Display an example image from the training set
display_image_with_annotations(train_images[0], train_annotations[0])

# Define augmentation sequence
seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # Horizontal flip
    iaa.Affine(rotate=(-20, 20)),  # Rotation
    iaa.Multiply((0.8, 1.2)),  # Brightness adjustment
    iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),  # Gaussian noise
])

# Augment training images with progress display
augmented_train_images = []
for img in tqdm(train_images, desc="Augmenting images"):
    augmented_train_images.append(seq(image=img))

# Example of displaying an augmented image
plt.imshow(cv2.cvtColor(augmented_train_images[0], cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Augmented Image')
plt.show()