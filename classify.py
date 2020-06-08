import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.applications.vgg16 import VGG16
import numpy as np
import os 
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense 
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import  GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model




def prediction (img, model1, model2):
    
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    pred1 = model1.predict(x)
    if pred1[0][0] < 0.8:
        pred2= model2.predict(x)
        return pred2[0].argmax()
    else:
        return('No dammage detected on this Car')

['bumper_dent', 'bumper_scratch', 'door_dent', 'door_scratch',
          'glass_shatter', 'head_lamp', 'tail_lamp', 'Unknow']
def decode_prediction(labels, argument):
    if labels[argument]== 'bumper_dent':
        return 'bosse detectée au niveau du par-choques'

    elif labels[argument]=='bumper_scratch':
        return 'égratignure au niveau du pare-chocs'
    
    elif labels[argument]== 'door_dent':
        return 'bosse detectée au niveau d une  porte '

    elif labels[argument]== 'door_scratch':
        return 'égratignure au niveau de la porte'
    elif labels[argument]== 'glass_shatter':
        return 'parebrise endommagée '
    
    elif labels[argument]== 'head_lamp':
        return 'Lamp frontale endommagée '

    elif labels[argument]== 'tail_lamp':
        return 'feu arrière endomagé'

    else :
        return 'any damage detect please photograph the damaged area and try again'


def predict(image1): 
    model = VGG16()
    image = load_img(image1, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # predict the probability across all output classes
    yhat = model.predict(image)
    # convert the probabilities to class labels
    label = decode_predictions(yhat)
    # retrieve the most likely result, e.g. highest probability
    label = label[0][0]
    return label 