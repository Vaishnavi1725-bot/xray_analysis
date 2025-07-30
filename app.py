import streamlit as st
import tensorflow as tf
from PIL import Image
#pip install pillow
import numpy as np
# @ => anotations
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("pneumonia_model.h5")
model=load_model()

def preprocess_image(image):
    image=image.resize((150,150)) #model size
    #0=> black 0.1=>gray 1=> white
    img_array=np.array(image)      #Handle grayscale (1 channnel) or RGB (4 channel)
    if img_array.ndim==2:
        img_array=np.stack((img_array,)*3,axis=-1) #it will convert grayscale into RGB
    elif img_array.shape[-1]==4:
        img_array=img_array[:,:,:3]   #drop unwanted pixels
    img_array=img_array/255.0 #convert to pixel
    img_array=np.expand_dims(img_array,axis=0)
    return img_array
st.set_page_config(page_title="pneumonia detector",layout="centered")
st.title("Pneumonia detection from chest x-ray")
st.write("upload a chest x-ray image to detect")
uploaded_file=st.file_uploader("upload x-ray image", type=["jpg","png","jpeg"])
if uploaded_file:st.success("File uploaded success")
image=Image.open(uploaded_file)
st.image(image,caption="uploaded image") 
with st.spinner("Analysis X-ray"):
    preprocessed=preprocess_image(image)
    prediction=model.predict(preprocessed)[0][0]
    st.success("prediction completed")
    if prediction>0.5:
        st.error(f"pneumonia affected(confidence:{prediction:.2f})")
    else:
        st.success(f"pneumonia affected(confidence:{prediction:.2f}")