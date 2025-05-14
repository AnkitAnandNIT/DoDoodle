import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

# Dataset (Ensure the files for car, house, and tree are available)
data_files = [
    "full_numpy_bitmap_car.npy",    # Label 0: Car
    "full_numpy_bitmap_house.npy",  # Label 1: House
    "full_numpy_bitmap_tree.npy",   # Label 2: Tree
]
labels = ["car", "house", "tree"]

# Load and prepare the dataset
X = []
y = []

for idx, file in enumerate(data_files):
    data = np.load(file)  # Load the numpy array data
    X.append(data)  # Append data
    y.append(np.full(len(data), idx))  # Labels for each class (0 for car, 1 for house, 2 for tree)

# Concatenate and preprocess
X = np.concatenate(X, axis=0)  # Combine data arrays
y = np.concatenate(y, axis=0)  # Combine labels

# Normalize the data
X = 255 - X  # Invert colors to make background white
X = X / 255.0  # Normalize pixel values
X = X.reshape(-1, 28, 28, 1)  # Reshape to 28x28 images with 1 channel (grayscale)

# One-hot encode the labels
y = tf.keras.utils.to_categorical(y, len(labels))

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(labels), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Set up callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=3)
checkpoint = ModelCheckpoint("best_model.h5", save_best_only=True)

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop, checkpoint])

# Save the model after training
model.save("doodle_model.keras")



