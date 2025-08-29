import os
import numpy as np
from PIL import Image

# Define paths for train and test sets
train_path = "gujarati-handwritten-digit-dataset-master\Train-Set"
test_path = "gujarati-handwritten-digit-dataset-master\Test-Set"

def process_images(data_path):
    data = []
    labels = []
    
    
    # Loop through each label folder
    for label in os.listdir(data_path):
        label_path = os.path.join(data_path, label)
        
        if not os.path.isdir(label_path):
            continue
        
        for file_name in os.listdir(label_path):
            file_path = os.path.join(label_path, file_name)
            if os.path.isfile(file_path):
                try:
                    img = Image.open(file_path).convert('L')  # Convert to grayscale
                    img = img.resize((28, 28))               # Resize to 28x28
                    img_array = np.array(img).flatten()      # Flatten the image
                    
                    
                    data.append(img_array)
                    labels.append(int(label))
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    return np.array(data), np.array(labels)

# Process train and test datasets
train_data, train_labels = process_images(train_path)
test_data, test_labels = process_images(test_path)

# Save processed data if needed
np.savez("processed_data.npz", train_data=train_data, train_labels=train_labels, test_data=test_data, test_labels=test_labels)

print("Data processing complete.")
