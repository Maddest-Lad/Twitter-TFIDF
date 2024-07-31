from pathlib import Path

import PIL.Image
import numpy as np
import tensorflow as tf

# Model paths
MODEL_PATH = Path('resources/dataset/model-resnet_custom_v3.h5')
TAG_PATH = Path('resources/dataset/tags.txt')

# Parameters
TAGS = [line.strip() for line in open(TAG_PATH).readlines()]
THRESHOLD = 0.5


class DeepDanbooruConnector:
    def __init__(self):
        self.model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    def preprocess_image(self, image: PIL.Image.Image) -> np.ndarray:
        _, height, width, _ = self.model.input_shape
        image = np.asarray(image)
        image = tf.image.resize_with_pad(image, target_height=height, target_width=width)
        image = image / 255.0  # Normalize to [0, 1]
        return image

    def interrogate(self, image_path: Path) -> str:
        image = PIL.Image.open(image_path)
        processed_image = self.preprocess_image(image)
        processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension
        probs = self.model.predict(processed_image)[0]
        probs = probs.astype(float)
        res = [label for prob, label in zip(probs.tolist(), TAGS) if prob >= THRESHOLD]
        return " ".join(res)
