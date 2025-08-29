import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

model = tf.keras.models.load_model('handwritten.model.keras')

image_number = 0
while os.path.isfile(f"digits/digit{image_number}.jpg"):
  try:
    img = cv2.imread(f"digits/digit{image_number}.jpg")[:,:,0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print(f"This digit is probably a {np.argmax(prediction)}")
    plt.imshow(img[0],cmap=plt.cm.binary)
    plt.show()
  except:
    print("Error")
  finally:
    image_number += 1