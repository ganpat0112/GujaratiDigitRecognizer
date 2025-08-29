import numpy as np
import tensorflow as tf
from collections import Counter

# Load processed dataset from .npz file
data = np.load("processed_data.npz")
x_train = data['train_data']  # Training data
y_train = data['train_labels']  # Training labels
x_test = data['test_data']  # Testing data
y_test = data['test_labels']  # Testing labels

# Normalize pixel values to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape data to include the channel dimension (for grayscale)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)


# Split training data into training and validation sets
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Calculate class weights to address class imbalance
from collections import Counter
counter = Counter(y_train)
total = len(y_train)
class_weights = {label: total / (len(counter) * count) for label, count in counter.items()}

# Define the CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # Update input shape
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),  # This will automatically flatten the output correctly
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')  # Assuming 10 classes
])


# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model with class weights
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val), class_weight=class_weights)

# Save the model
model.save('handwritten_256x256.model.h5')

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
