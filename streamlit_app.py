import streamlit as st

from streamlit_gallery import apps, components
from streamlit_gallery.utils.page import page_group

def main():
    page = page_group("p")

    with st.sidebar:
        st.title("🛍️ Fashion Product's Gallery")

        with st.expander("✨ APPS", True):
            page.item("Streamlit gallery", apps.gallery, default=True)

        with st.expander("🧩 COMPONENTS", True):
            page.item("Find by Category", components.find_by_category)
            page.item("Find by Text", components.find_by_text)
            page.item("Find by Image👛", components.find_by_image)

    page.show()

if __name__ == "__main__":
    st.set_page_config(page_title="Streamlit Gallery by VMware Data", page_icon="🛍️", layout="wide")
    main()