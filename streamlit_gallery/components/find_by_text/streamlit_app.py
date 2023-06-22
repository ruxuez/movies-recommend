import streamlit as st

from streamlit_gallery.utils.db_helper import (
    cosine_distance,
    vector,
    fashion_images,
    get_image_from_url,
)
from sentence_transformers import SentenceTransformer, util

from multiprocessing import Pool

# First, we load the respective CLIP model
model = SentenceTransformer("clip-ViT-B-32")


def main():
    st.subheader("Instruction")
    st.markdown(
        "You can find the products you want by entering the description. **It supports multiple languages**, not just english."
    )
    number_results = st.number_input("Top-N-Search:", value=50, min_value=10, step=10)
    st.subheader("Description")
    text_search = st.text_input("Enter your text:", key="text")
    search_button = st.button("Search")

    if search_button:
        st.subheader("Results")
        data_load_state = st.empty()
        data_load_state.markdown("Searching results...")
        search_text_embeddings = model.encode(
            [text_search], convert_to_tensor=False, show_progress_bar=False
        )
        target_by_text = str(search_text_embeddings[0].tolist())
        result_by_text = fashion_images.assign(
            cosine_distance=lambda t: cosine_distance(
                t["image_embedding"], vector(target_by_text)
            )
        ).order_by("cosine_distance")[:number_results]
        data_load_state.markdown(
            f"**{len(list(result_by_text))} Products Found**: ... Printing images..."
        )
        captions = [row["productdisplayname"] for row in result_by_text]
        pool = Pool(8)
        images = pool.map(get_image_from_url, [row["link"] for row in result_by_text])
        st.image(images, width=200, caption=captions)
        data_load_state.markdown(f"**{len(list(result_by_text))} Products Found**")


if __name__ == "__main__":
    main()
