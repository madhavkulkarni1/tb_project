import csv
from pathlib import Path
import cv2
import numpy as np
from services.predict import TBPredictor

INPUT_FOLDER = Path("batch_images")
OUTPUT_FILE = "results.csv"

def run_batch_prediction():

    predictor = TBPredictor()

    results = []

    for file in sorted(INPUT_FOLDER.iterdir()):

        if file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        # Debug: show brightness
        img = cv2.imread(str(file))
        print(file.name, "pixel mean:", np.mean(img))

        # Run prediction
        result = predictor.predict(str(file))

        print("Prediction:", result)

        results.append([
            file.name,
            result["prediction"],
            result["probability"],
            result["confidence_percent"],
            result["risk_level"]
        ])

    with open(OUTPUT_FILE, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "filename",
            "prediction",
            "probability",
            "confidence_percent",
            "risk_level"
        ])

        writer.writerows(results)

    print("\nBatch prediction complete.")
    print("Results saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    run_batch_prediction()