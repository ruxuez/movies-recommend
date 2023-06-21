import streamlit as st
from streamlit_chat import message
import psycopg2
import pandas as pd
import greenplumpython as gp
import os

from sentence_transformers import SentenceTransformer, util
from PIL import Image
import requests
from io import BytesIO

import numpy as np
import os
from tqdm.autonotebook import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

db = gp.database(uri="postgres://gpadmin:changeme@35.225.47.84:5432/warehouse")

#First, we load the respective CLIP model
model = SentenceTransformer('clip-ViT-B-32')

# Setting page title and header
st.set_page_config(page_title="Greenplum-pgVector Demo: Fashion Product Finder", page_icon=":robot_face:")
st.markdown("<h2 style='text-align: center;'>Tell me what fashion product are you searching 🤖</h2>", unsafe_allow_html=True)


# Initialise session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
if 'model_name' not in st.session_state:
    st.session_state['model_name'] = []
if 'cost' not in st.session_state:
    st.session_state['cost'] = []
if 'total_tokens' not in st.session_state:
    st.session_state['total_tokens'] = []
if 'total_cost' not in st.session_state:
    st.session_state['total_cost'] = 0.0

st.sidebar.title("Sidebar")
model_name = st.sidebar.radio("Choose a method:", ("Find product by Category", "Find product by Text", "Find product by Image"))
counter_placeholder = st.sidebar.empty()
clear_button = st.sidebar.button("Clear Output", key="clear")
CATEGORY = st.text_input('Product Main Category:')
SUB_CATEGORY = st.text_input('Product Sub-Category:')
TYPE = st.text_input('Product Type:')

print('Model name:', model_name)
if model_name == "Find product by Category":
    function = "match_category"
elif model_name == "Find product by Text":
    function="match_text"
else:
    function = "match_image"

# reset everything
if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    st.session_state['number_tokens'] = []
    st.session_state['model_name'] = []
    st.session_state['cost'] = []
    st.session_state['total_cost'] = 0.0
    st.session_state['total_tokens'] = []

# Get repr of Grenplum operator vector cosine distance
cosine_distance = gp.operator("<=>") 

vector = gp.type_("vector", modifier=512)

fashion_images = db.create_dataframe(table_name="image_embeddings", schema="fashion")
images_styles = db.create_dataframe(table_name="image_styles", schema="fashion")

# generate a response
def generate_response(prompt):
    print(function)
    st.session_state['messages'].append({"role": "user", "content": prompt})

    if function == 'match_category':     
        result_by_text = images_styles.where(
        lambda t: (t["mastercategory"] == CATEGORY) & (t["subcategory"] == SUB_CATEGORY) & (t["articletype"] == TYPE)
        )[:100]
        for row in result_by_text:
            response = requests.get(row["link"])
            img = Image.open(BytesIO(response.content))
        response = st.image(img)      
    elif function=="match_text":
        search_text_embeddings = model.encode([prompt], convert_to_tensor=False, show_progress_bar=False)
        target_by_text = np.array2string(search_text_embeddings[0], separator=',')
        result_by_text = fashion_images.assign(
            cosine_distance=lambda t: cosine_distance(
                t["image_embedding"], vector(target_by_text)
                )
            ).order_by("cosine_distance")[:100]
        images = []
        for row in result_by_text:
            response = requests.get(row["link"])
            img = Image.open(BytesIO(response.content))
            images.append(img)
        st.image(images, width=100)

    else:
        response = requests.get(prompt)
        img_search = Image.open(BytesIO(response.content))
        search_image_embedding = model.encode(img_search)
        target_by_image = np.array2string(search_image_embedding, separator=',')
        result_by_image = fashion_images.assign(
            cosine_distance=lambda t: cosine_distance(
                t["image_embedding"], vector(target_by_image)
                )
            ).order_by("cosine_distance")[:100]
        for row in result_by_image:
            response = requests.get(row["link"])
            img = Image.open(BytesIO(response.content))
        response = st.image(img)

    st.session_state['messages'].append({"role": "assistant", "content": response})

    return response


# container for chat history
response_container = st.container()
# container for text box
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        filter_button = st.sidebar.button("Filter")
        user_input = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        output = generate_response(user_input)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)
        st.session_state['model_name'].append(model_name)

    if filter_button and CATEGORY:
        output = generate_response(CATEGORY)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)
        st.session_state['model_name'].append(model_name)


if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            message('Answer:', key=str(i))
            st.markdown(st.session_state["generated"][i])
            st.write(
                f"Method used: {st.session_state['model_name'][i]};")
