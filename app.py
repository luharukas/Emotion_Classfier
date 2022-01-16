import keras as k
import streamlit as st
import cv2
from PIL import Image
from function import convertor
import tensorflow as tf
import numpy as np

with tf.device("cpu:0"):
    model=k.models.load_model('my_model')
    labels=[' Contempt Neutral Face',"Angry Face","Disgust Face","Fear", 'Happy Face','Sad Face',"Surprised..."]

col1, col2 = st.columns( [0.8, 0.2])
with col1:               
    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Choose one image from here</p>', unsafe_allow_html=True)

st.sidebar.markdown('<p class="font">Facial Emotion Recognition app</p>', unsafe_allow_html=True)
with st.sidebar.expander("About the App"):
     st.write("""
        This app is developed by Shubham Luharuka and his team as a project at CSE Depatment in R V Institute of Technology and Management. \n\n This app to built for recognizing your facial Emotion. We are giving few image as option to you to understand its working process.
     """)

st.markdown("""
<style>
.big-font {
    font-size:30px !important;
}
</style>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=['jpg','png','jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    ret_image=convertor(image)
    col1, col2 = st.columns( [0.5, 0.5])
    with col1:
        st.markdown('<p style="text-align: center;">Before</p>',unsafe_allow_html=True)
        st.image(image,width=300)  

    with col2:
        st.markdown('<p style="text-align: center;">After</p>',unsafe_allow_html=True)
        st.image(ret_image[0],width=300)

    ret2=np.resize(ret_image,(1,126,126,3))
    with tf.device("cpu:0"):
        ans=np.argmax(model.predict(ret2))
        
    st.write("Your Expression: "+labels[ans])

        
    