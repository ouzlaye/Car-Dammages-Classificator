import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True
import streamlit as st 
from PIL import Image
import tensorflow as tf 
from classify import prediction, decode_prediction
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
from keras.models import load_model
import pickle 
import keras
from tensorflow.keras.preprocessing import image
import time



#model1_path = "./Modeles/saved_model/model1.h5"
#model2_path= "./Modeles/saved_model/model2"
#with open('model1', 'rb') as file1:
model1 = load_model('Models/model1.h5')
model2 = load_model('Models/model2.h5')


label2 = ['bumper_dent', 'bumper_scratch', 'door_dent', 'door_scratch',
          'glass_shatter', 'head_lamp', 'tail_lamp', 'Unknow']

label1= ['dammage', 'whoole']
files_types=['jpeg','jpg', 'png']
image1 = Image.open('images/damage2.jpeg')
st.image(image1, use_column_width=True)
st.title("Car Dammage Classificator")

uploaded_file = st.file_uploader("Choose an image...", type=files_types)
if uploaded_file is not None:
    Image = Image.open(uploaded_file)
    img=Image.save('images/my_upload.jpg')
    path = 'my_upload.jpg'
    img =image.load_img(path,  target_size=(150,150))
    #img = load_img(uploaded_file, target_size=(150,150))
    st.image(Image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    my_bar = st.progress(0)

    for percent_complete in range(100):
        time.sleep(0.1)
        my_bar.progress(percent_complete + 1)
    label = prediction(img, model1, model2)
    my_placeholder = st.empty()
    if type(label)== str:
        #st.write(label)
        st.success(label)
    else:
        
        message = decode_prediction(label2, label)
        st.error(message)

add_selectbox = st.sidebar.selectbox(
    "Cette application  permet de diagnostiquer une voiture afin de detecter de potentiels dommages occasionés par un accident. Pour ce faire elle se base sur l intelligences artificielle plus précisement sur les réseaux de neurones convolutionnel pour pouvoir faire la détection a partir d'images uploadées ",
    ("Single image", "Multiple images"),
    
)



#st.markdown('By Ousseynou Diop')
        

    #st.write('%s (%.2f%%)' % (label[1], label[2]*100))
    #st.write('%s (%.2f%%)' % (label2[label])
    


