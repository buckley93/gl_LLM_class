#streamlit run filename.py
import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Retrieve API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

def load_image(prompt):
    response = client.images.generate(
        model="dall-e-3", prompt=prompt, n=1, size="1024x1024"
    )
    return response.data[0]


with st.sidebar:
    st.title("Gen AI Program")
    st.write("Build using Open AI Models")

st.title("Image Generator App")

prompt = st.text_input("Ask for something and we will illustrate for you", "")

if prompt:
    with st.spinner("Wait for it..."):
        image_res = load_image(prompt)

    left_co, cent_co, last_co = st.columns(3)
    with cent_co:
        st.image(
            image_res.url, width=200, caption=image_res.revised_prompt.split(".")[0]
        )