import streamlit as st 
from PIL import Image
import tensorflow as tf 
from classify import prediction
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
from tensorflow.keras.models import load_model
import pickle 
from tensorflow import keras



model1_path = "./Modeles/pickle_model/model1"
model2_path= "./Modeles/pickle_model/model2"
with open('model1', 'rb') as file1:
    model1 = pickle.load(file1)
#model2 = load_model(model2_path)
#with open('model2', 'rb') as file2:
    #model2 = pickle.load(file2)


label2 = ['bumper_dent', 'bumper_scratch', 'door_dent', 'door_scratch',
          'glass_shatter', 'head_lamp', 'tail_lamp', 'Unknow']

label1= ['dammage', 'whoole']
files_types=['jpeg','jpg', 'png']
st.title("Car Dammage Classificator")

uploaded_file = st.file_uploader("Choose an image...", type=files_types)
if uploaded_file is not None:
    Image = Image.open(uploaded_file)
    img_array = np.array(Image)
    #img = load_img(uploaded_file, target_size=(150,150))
    st.image(Image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    #label = prediction(img_array, model1, model2)
    #st.write('%s (%.2f%%)' % (label[1], label[2]*100))
    #st.write('%s (%.2f%%)' % (label2[label])
    


