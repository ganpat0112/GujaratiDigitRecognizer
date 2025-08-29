import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model('handwritten_256x256.model.h5')

# Path to the folder containing test images
image_folder = "digits"

# Iterate through all test images in the folder
image_number = 0
while os.path.isfile(f"{image_folder}/digit{image_number}.png"):
    try:
        # Read and preprocess the image
        img = cv2.imread(f"{image_folder}/digit{image_number}.png", cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: Could not read image digit{image_number}.png")
            image_number += 1
            continue
        
        # Resize image to 28x28
        img = cv2.resize(img, (28, 28))  # Resize to 28x28
        
        # Normalize pixel values to [0, 1]
        img = img / 255.0
        
        # Add batch dimension and reshape
        img = np.array([img])  # Add batch dimension
        img = img.reshape(-1, 28, 28, 1)  # Reshape to (28, 28, 1)
        
        # Predict using the model
        prediction = model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}")
        
        # Display the image
        plt.imshow(img[0, :, :, 0], cmap=plt.cm.binary)
        plt.title(f"Prediction: {np.argmax(prediction)}")
        plt.show()
    except Exception as e:
        print(f"Error processing image digit{image_number}.png: {e}")
    finally:
        image_number += 1
