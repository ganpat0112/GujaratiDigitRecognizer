import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Function to load and preprocess images
def load_dataset(data_dir):
    data = []
    labels = []
    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        if os.path.isdir(label_path):  # Ensure it's a directory
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load image in grayscale
                if img is not None:
                    # Resize image to 28x28
                    img = cv2.resize(img, (28, 28))
                    data.append(img)
                    labels.append(int(label))  # Convert folder name to label
    return np.array(data), np.array(labels)

# Load training data
train_dir = "gujarati-handwritten-digit-dataset-master/Train-Set"
x_train, y_train = load_dataset(train_dir)

# Load testing data
test_dir = "gujarati-handwritten-digit-dataset-master/Test-Set"
x_test, y_test = load_dataset(test_dir)

# Normalize the data (scale pixel values to range [0, 1])
x_train = x_train / 255.0
x_test = x_test / 255.0

# Ensure the input shape matches the model requirements
x_train = x_train.reshape(-1, 28, 28)
x_test = x_test.reshape(-1, 28, 28)

# Define the model (same as your current model)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=3)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Save the model
model.save('handwritten.model.keras')
