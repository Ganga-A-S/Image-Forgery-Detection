import os
from MantraNet.mantranet import pre_trained_model, check_forgery
#from BusterNet.BusterNetCore import create_BusterNet_testing_model, simple_cmfd_decoder, visualize_result
import streamlit as st
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

st.header("Image Manipulation Detection Demo")

@st.cache()
def check_image(img_path):
    device = "cpu"  # to change if you have a GPU with at least 12Go RAM (it will save you a lot of time !)
    MantraNetmodel = pre_trained_model(
        weight_path="MantraNet/MantraNetv4.pt", device=device
    )
    fig = check_forgery(MantraNetmodel, img_path=img_path, device=device)



uploaded_image = st.file_uploader("Upload your image", type=["jpg", "png"])
if uploaded_image is not None:
    with open(os.path.join("images", uploaded_image.name), "wb") as f:
        f.write(uploaded_image.read())
    fig, figure  = check_image(os.path.join("images", uploaded_image.name))
    st.write("ManTraNet")
    st.pyplot(fig)
   
