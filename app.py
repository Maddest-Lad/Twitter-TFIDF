import base64
from io import BytesIO
from pathlib import Path
from PIL import Image
import eel
from modules.tf_idf import TFIDFHandler

# Initialize Eel
eel.init('web')

tfidf_handler = TFIDFHandler(method='deepdanbooru', rebuild=False)


@eel.expose
def get_caption(image_base64):
    image_path = decode_image(image_base64)
    caption = tfidf_handler.get_caption(image_path)
    return caption


@eel.expose
def score_image(image_base64):
    image_path = decode_image(image_base64)
    score = tfidf_handler.score_image(image_path)
    return score


def decode_image(image_base64):
    image_data = base64.b64decode(image_base64.split(',')[1])
    image = Image.open(BytesIO(image_data))

    # Convert image to RGB if not already in RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")

    image_path = Path("temp.jpg")
    image.save(image_path)
    return image_path


if __name__ == '__main__':
    eel.start('index.html', size=(800, 600))
