import eel
from pathlib import Path
from modules.tf_idf import TFIDFHandler

# Initialize Eel
eel.init('web')

tfidf_handler = TFIDFHandler(rebuild=False)


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
    import base64
    from PIL import Image
    from io import BytesIO

    image_data = base64.b64decode(image_base64.split(',')[1])
    image = Image.open(BytesIO(image_data))
    image_path = Path("temp.jpg")
    image.save(image_path)
    return image_path


if __name__ == '__main__':
    eel.start('index.html', size=(800, 600))
