import requests
from PIL import Image
from io import BytesIO


def get_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img
