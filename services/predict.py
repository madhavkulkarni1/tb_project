import tensorflow as tf
from pathlib import Path
from services.preprocess import preprocess_image

# Detect project root automatically
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "final_tb_model.keras"

THRESHOLD = 0.61


import tensorflow as tf
from pathlib import Path
from services.preprocess import preprocess_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "final_tb_model.keras"

THRESHOLD = 0.15


class TBPredictor:

    def __init__(self):

        print("Loading model from:", MODEL_PATH)

        self.model = tf.keras.models.load_model(MODEL_PATH)

        print("\nModel successfully loaded\n")


    def predict(self, image_path):

        img = preprocess_image(image_path)

        pred = self.model.predict(img, verbose=0)

        prob = float(pred[0][0])
        prob = round(prob, 4)

        if prob >= THRESHOLD:
            label = "TB"
        else:
            label = "Normal"

        return {
            "prediction": label,
            "probability": prob,
            "confidence_percent": round(prob * 100, 2),
            "risk_level": "High" if prob >= 0.85 else "Moderate" if prob >= 0.61 else "Low"
        }