import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from services.preprocess import preprocess_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "final_tb_model.keras"

# Last convolution layer of DenseNet121
LAST_CONV_LAYER = "conv5_block16_concat"


class GradCAM:

    def __init__(self):
        self.model = tf.keras.models.load_model(MODEL_PATH)

        self.grad_model = tf.keras.models.Model(
            [self.model.inputs],
            [self.model.get_layer(LAST_CONV_LAYER).output, self.model.output]
        )

    def generate(self, image_path):

        img = preprocess_image(image_path)

        with tf.GradientTape() as tape:

            conv_outputs, predictions = self.grad_model(img)
            loss = predictions[:, 0]

        grads = tape.gradient(loss, conv_outputs)

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]

        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

        heatmap = heatmap.numpy()

        return heatmap

def overlay_heatmap(image_path, heatmap, output_path="gradcam_result.jpg"):

    img = cv2.imread(image_path)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed = heatmap * 0.4 + img

    cv2.imwrite(output_path, superimposed)

    return output_path