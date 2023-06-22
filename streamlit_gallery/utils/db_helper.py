import streamlit as st

import greenplumpython as gp
import os
import requests
from PIL import Image
from io import BytesIO

db = gp.database(
    params={
        "host": st.secrets["db_hostname"],
        "dbname": st.secrets["db_name"],
        "user": st.secrets["db_username"],
        "port": st.secrets["db_port"],
        "password": st.secrets["db_password"],
    }
)

gp.config.print_sql = True

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Get repr of Grenplum operator vector cosine distance
cosine_distance = gp.operator("<=>")

vector = gp.type_("vector", modifier=512)

fashion_images = db.create_dataframe(table_name="product_embeddings", schema="fashion")
images_styles = db.create_dataframe(table_name="image_styles", schema="fashion")


def get_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img
