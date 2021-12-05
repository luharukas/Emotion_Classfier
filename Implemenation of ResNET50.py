#Importing libraries
import cv2
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
#To convert array of labeled data to one-hot vector
from tensorflow.keras.utils import to_categorical

#Importing tensorflow to build and train the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,InputLayer,Conv2D,MaxPooling2D
#Importing softmax as activation function from tensorflow
from tensorflow.keras.activations import softmax
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D

img_data_array=[]
class_name=[]
#giving the path till the train folder
img_folder="C:\\Users\\luhar\\OneDrive\\Documents\\Code with ShiviSandy\\emotion_classifier\\archive\\train"
#Pre-processing the images
for dir1 in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir1)):
            image_path= os.path.join(img_folder, dir1,  file)
            image= cv2.imread( image_path)
            image=np.array(image)
            image = image.astype('float32')
            image /= 255 
            img_data_array.append(image)
            class_name.append(dir1)
label_encoder = preprocessing.LabelEncoder()
class_name= label_encoder.fit_transform(class_name)

#Converting images to numpy arrays
img_data_array=np.array(img_data_array)
img_data_array[0].shape

#Implementing the model
base_model = ResNet50(include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(8, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=predictions)
#Compiling the model by specifying the optimizer and loss function 
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(img_data_array,class_name,verbose=2,validation_split=0.2,epochs=10,)