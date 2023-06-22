import streamlit as st

from streamlit_gallery.utils.readme import readme

from sentence_transformers import SentenceTransformer, util

import greenplumpython as gp
import os

from multiprocessing import Pool

from PIL import Image
import requests
from io import BytesIO

import numpy as np

db = gp.database(uri="postgres://gpadmin:changeme@35.225.47.84:5432/warehouse")

#First, we load the respective CLIP model
model = SentenceTransformer('clip-ViT-B-32')

gp.config.print_sql = True

# Get repr of Grenplum operator vector cosine distance
cosine_distance = gp.operator("<=>") 

vector = gp.type_("vector", modifier=512)

fashion_images = db.create_dataframe(table_name="product_embeddings", schema="fashion")
images_styles = db.create_dataframe(table_name="image_styles", schema="fashion")

def get_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

def main():
    st.subheader("Instruction")
    st.markdown("You can find similar products by providing an image of the product, either by uploading an image file or providing an image URL.")

    c1, c2 = st.columns([2, 2])
    c1.subheader("Upload Your Image")
    image_search_url=c1.text_input("Enter your image url:", key="image")
    uploaded_file = c1.file_uploader("Choose a file")
    search_button = c1.button("Search")

    if search_button:
        if uploaded_file is not None:
            img_search = Image.open(uploaded_file)
        else:
            response = requests.get(image_search_url)
            img_search = Image.open(BytesIO(response.content))
        c2.image(img_search, width=200, caption="Product you would like to find")

        st.subheader("Results")
        search_image_embedding = model.encode(img_search)
        target_by_image = np.array2string(search_image_embedding, separator=',')
        result_by_image = fashion_images.assign(
            cosine_distance=lambda t: cosine_distance(
                t["image_embedding"], vector(target_by_image)
                )
            ).order_by("cosine_distance")[:50]
        print(len(list(result_by_image)))
        captions = [row["productdisplayname"] for row in result_by_image]
        pool = Pool(8) 
        images = pool.map(get_image_from_url, [row["link"] for row in result_by_image])
        st.image(images, width=200, caption=captions)


if __name__ == "__main__":
    main()
