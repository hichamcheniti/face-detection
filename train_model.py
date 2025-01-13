import cv2
import keras
import numpy as np

# Build the model
model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()
# Convert images and annotations to numpy arrays
X_train = np.array([cv2.resize(img, (256, 256)) for img in augmented_train_images])
y_train = np.array([1 if len(faces) > 0 else 0 for faces in train_annotations])  # Simplified labels for demo

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(np.array([cv2.resize(img, (256, 256)) for img in val_images]), np.array([1 if len(faces) > 0 else 0 for faces in val_annotations])))

# Convert test images and annotations to numpy arrays
X_test = np.array([cv2.resize(img, (256, 256)) for img in test_images])
y_test = np.array([1 if len(faces) > 0 else 0 for faces in test_annotations])

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'\nTest accuracy: {test_acc}')
# Save the model
model.save('face_detector.h5')