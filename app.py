import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import base64

from test import *

st.set_page_config(
    page_title="Fashion Image Caption Creator",
    page_icon=":dress:",
)

hide_streamlit_style = """
            <style>
            # MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.markdown('''
    # Fashion Image Caption Creator
    ***
    ''')

options = st.sidebar.selectbox(
            'Choose Model: ',
            ('1', '2', '3', '4', '5')
     )

st.markdown(f'## Model selected: ```{options}```')

image_data = st.file_uploader("Upload file", type=["jpg", "png", "jpeg"])

st.markdown("<p style='text-align: center;'>OR</p>",
            unsafe_allow_html=True)

image_url = st.text_input("URL : ")

if image_data is None and image_url:
    try:
        response = requests.get(image_url)
        image_data = BytesIO(response.content)
        pred_text = " ".join(prediction(from_url = False, uploader_image_data = image_data))
        pred_text = pred_text.replace("<start> ", "").replace(" <end>", ".").capitalize()
        st.markdown(f"### {pred_text}")
    except:
        st.write("Please enter a valid URL")

elif image_data:
    # try:
    st.write("hey")
    image_data = BytesIO(image_data.read())
    pred_text = " ".join(prediction(from_url = False, uploader_image_data = image_data))
    pred_text = pred_text.replace("<start> ", "").replace(" <end>", ".").capitalize()
    st.markdown(f"### {pred_text}")
    # except:
    #     st.write("Some unexpected error occured..")
