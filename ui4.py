import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model("handwritten_256x256.model.h5")

# Load test data (assuming processed .npz file)
data = np.load("processed_data.npz")
x_test = data['test_data']
y_test = data['test_labels']

# Preprocess test data
x_test = x_test.astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1)

# Function to preprocess the image for prediction
def preprocess_image(img):
    img = img.resize((28, 28))  # Resize to match the model input
    img = img.convert("L")  # Convert to grayscale
    img = np.array(img).astype('float32') / 255.0  # Normalize
    img = img.reshape(1, 28, 28, 1)  # Reshape for the model
    return img

# Function to predict the digit
def predict_digit(img):
    processed_img = preprocess_image(img)
    prediction = model.predict(processed_img)
    return np.argmax(prediction), max(prediction[0]) * 100  # Predicted digit and confidence

# Canvas drawing logic
class DrawingCanvas:
    def __init__(self, canvas):
        self.canvas = canvas
        self.image = Image.new("L", (256, 256), "white")  # Create a blank white image
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.bind("<B1-Motion>", self.paint)  # Draw while mouse button is held
        self.canvas.bind("<ButtonRelease-1>", self.reset_line)  # Reset after each stroke
        self.last_x, self.last_y = None, None

    def paint(self, event):
        if self.last_x and self.last_y:
            # Draw a line between the previous and current positions
            x, y = event.x, event.y
            self.canvas.create_line(self.last_x, self.last_y, x, y, width=10, fill="black")
            self.draw.line((self.last_x, self.last_y, x, y), fill="black", width=10)
        self.last_x, self.last_y = event.x, event.y

    def reset_line(self, event):
        self.last_x, self.last_y = None, None

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (256, 256), "white")
        self.draw = ImageDraw.Draw(self.image)

# Function to handle canvas prediction
def predict_from_canvas():
    # Resize the drawn image to 28x28 for prediction
    small_image = drawing_canvas.image.resize((28, 28))
    digit, confidence = predict_digit(small_image)
    messagebox.showinfo("Prediction", f"Predicted Digit: {digit}\nConfidence: {confidence:.2f}%")


# Function to handle file upload prediction
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

# Function to display performance metrics
def show_metrics():
    y_pred_prob = model.predict(x_test)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Compute ROC curve and AUC for each class
    plt.figure(figsize=(10, 8))
    for i in range(10):  # Assuming 10 classes
        fpr, tpr, _ = roc_curve((y_test == i).astype(int), y_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.show()

    # Print classification report
    report = classification_report(y_test, y_pred, target_names=[f"Class {i}" for i in range(10)])
    print("Classification Report:\n", report)

    # Display confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks(np.arange(10), labels=[f"Class {i}" for i in range(10)])
    plt.yticks(np.arange(10), labels=[f"Class {i}" for i in range(10)])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

# UI Design
root = tk.Tk()
root.title("Handwritten Digit Recognition")

# Canvas for drawing
canvas_widget = tk.Canvas(root, width=256, height=256, bg="white")
canvas_widget.grid(row=0, column=0, padx=10, pady=10)
drawing_canvas = DrawingCanvas(canvas_widget)

# Buttons
btn_predict_canvas = tk.Button(root, text="Predict from Canvas", command=predict_from_canvas)
btn_predict_canvas.grid(row=1, column=0, padx=10, pady=10)

btn_upload = tk.Button(root, text="Upload Image", command=predict_from_file)
btn_upload.grid(row=2, column=0, padx=10, pady=10)

btn_metrics = tk.Button(root, text="Show Metrics", command=show_metrics)
btn_metrics.grid(row=3, column=0, padx=10, pady=10)

btn_clear = tk.Button(root, text="Clear Canvas", command=drawing_canvas.clear)
btn_clear.grid(row=4, column=0, padx=10, pady=10)

# Run the application
root.mainloop()
