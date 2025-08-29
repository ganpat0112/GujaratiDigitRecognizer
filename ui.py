import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("handwritten_256x256.model.h5")

# Function to preprocess the image for prediction
def preprocess_image(img):
    img = img.resize((28, 28))  # Resize to 28x28
    img = img.convert("L")  # Convert to grayscale
    img = np.array(img).astype('float32') / 255.0  # Normalize
    img = img.reshape(1, 28, 28, 1)  # Reshape for the model
    return img

# Function to predict the digit
def predict_digit(img):
    processed_img = preprocess_image(img)
    prediction = model.predict(processed_img)
    return np.argmax(prediction), max(prediction[0]) * 100  # Predicted digit and confidence

# Function to handle canvas drawing
def predict_from_canvas():
    canvas_image = Image.new("RGB", (28, 28), "white")  # Create a blank white image
    draw = ImageDraw.Draw(canvas_image)
    canvas_data = canvas.postscript(colormode="color")
    canvas_image = Image.open(canvas_data)
    digit, confidence = predict_digit(canvas_image)
    messagebox.showinfo("Prediction", f"Predicted Digit: {digit}\nConfidence: {confidence:.2f}%")

# Function to handle file upload
def predict_from_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if not file_path:
        return
    try:
        img = Image.open(file_path)
        digit, confidence = predict_digit(img)
        messagebox.showinfo("Prediction", f"Predicted Digit: {digit}\nConfidence: {confidence:.2f}%")
    except Exception as e:
        messagebox.showerror("Error", f"Unable to process the file.\nError: {str(e)}")

# UI Design
root = tk.Tk()
root.title("Handwritten Digit Recognition")

# Canvas for drawing
canvas = tk.Canvas(root, width=256, height=256, bg="white")
canvas.grid(row=0, column=0, padx=10, pady=10)

# Buttons
btn_predict_canvas = tk.Button(root, text="Predict from Canvas", command=predict_from_canvas)
btn_predict_canvas.grid(row=1, column=0, padx=10, pady=10)

btn_upload = tk.Button(root, text="Upload Image", command=predict_from_file)
btn_upload.grid(row=2, column=0, padx=10, pady=10)

# Clear button
def clear_canvas():
    canvas.delete("all")

btn_clear = tk.Button(root, text="Clear Canvas", command=clear_canvas)
btn_clear.grid(row=3, column=0, padx=10, pady=10)

# Run the application
root.mainloop()
