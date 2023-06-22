import streamlit as st

from streamlit_gallery.utils.db_helper import get_image_from_url
from sentence_transformers import SentenceTransformer, util

from multiprocessing import Pool
from PIL import Image
import requests
from io import BytesIO
import os

import greenplumpython as gp

db = gp.database(
    params={
        "host": st.secrets["db_hostname"],
        "dbname": st.secrets["db_name"],
        "user": st.secrets["db_username"],
        "port": st.secrets["db_port"],
        "password": st.secrets["db_password"],
    }
)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

gp.config.print_sql = True

# Get repr of Grenplum operator vector cosine distance
cosine_distance = gp.operator("<=>")

vector = gp.type_("vector", modifier=512)

fashion_images = db.create_dataframe(table_name="product_embeddings", schema="fashion")

# First, we load the respective CLIP model
model = SentenceTransformer("clip-ViT-B-32")


def main():
    st.subheader("Instruction")
    st.markdown(
        "You can find similar products by providing an image of the product, either by uploading an image file or providing an image URL."
    )

    c1, c2 = st.columns([2, 2])
    number_results = c1.number_input("Top-N-Search:", value=50, min_value=10, step=10)
    c1.subheader("Upload Your Image")
    image_search_url = c1.text_input("Enter your image url:", key="image")
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
        data_load_state = st.empty()
        data_load_state.markdown("Searching results...")
        search_image_embedding = model.encode(img_search)
        target_by_image = str(search_image_embedding.tolist())
        result_by_image = fashion_images.assign(
            cosine_distance=lambda t: cosine_distance(
                t["image_embedding"], vector(target_by_image)
            )
        ).order_by("cosine_distance")[:number_results]
        data_load_state.markdown(
            f"**{len(list(result_by_image))} Products Found**: ... Printing images..."
        )
        captions = [row["productdisplayname"] for row in result_by_image]
        pool = Pool(4)
        images = pool.map(get_image_from_url, [row["link"] for row in result_by_image])
        st.image(images, width=200, caption=captions)
        data_load_state.markdown(f"**{len(list(result_by_image))} Products Found**")


if __name__ == "__main__":
    main()
