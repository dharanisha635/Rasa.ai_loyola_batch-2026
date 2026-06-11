import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("model.keras")

def predict_image(path):

    img = cv2.imread(path)

    if img is None:
        print("Error: Image not found →", path)
        return

    # Resize image
    img = cv2.resize(img, (128,128))

    # Normalize
    img = img / 255.0

    # Reshape
    img = np.reshape(img, (1,128,128,3))

    # Predict
    pred = model.predict(img)[0][0]

    # Output
    print("Prediction value:", pred)

    if pred > 0.5:
        print(path, "→ Dirty Street")
    else:
        print(path, "→ Clean Street")

# Test images
predict_image("image1.png")
predict_image("image2.png")
predict_image("image3.png")
predict_image("image4.png")
predict_image("image5.png")
predict_image("image6.png")
predict_image("image7.png")