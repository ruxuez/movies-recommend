import streamlit as st

from streamlit_gallery.utils.readme import readme

from sentence_transformers import SentenceTransformer, util

import greenplumpython as gp
import os

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

fashion_images = db.create_dataframe(table_name="image_embeddings", schema="fashion")
images_styles = db.create_dataframe(table_name="image_styles", schema="fashion")


def main():

    st.subheader("Description")
    text_search=st.text_input("Enter your text:", key="text")
    search_button = st.button("Search")

    if search_button:
        st.subheader("Results")
        search_text_embeddings = model.encode([text_search], convert_to_tensor=False, show_progress_bar=False)
        target_by_text = np.array2string(search_text_embeddings[0], separator=',')
        result_by_text = fashion_images.assign(
            cosine_distance=lambda t: cosine_distance(
                t["image_embedding"], vector(target_by_text)
                )
            ).order_by("cosine_distance")[:100]
        print(len(list(result_by_text)))
        images = []
        for row in result_by_text:
            response = requests.get(row["link"])
            img = Image.open(BytesIO(response.content))
            images.append(img)
        st.image(images, width=100)


if __name__ == "__main__":
    main()
