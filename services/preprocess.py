import cv2
import numpy as np

TARGET_SIZE = (224, 224)

def preprocess_image(image_path):

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # resize
    img = cv2.resize(img, TARGET_SIZE)

    # convert grayscale → 3 channels
    img = np.stack((img,)*3, axis=-1)

    # normalize
    img = img.astype("float32") / 255.0

    # add batch dimension
    img = np.expand_dims(img, axis=0)

    return img