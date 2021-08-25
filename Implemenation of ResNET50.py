import cv2
import numpy as np
import os
import tensorflow as tf
import pandas as pd
from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical

#import tensorflow.keras as k
#import matplotlib.pyplot as plt
#from google.colab.patches import cv2_imshow
#import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,InputLayer,Conv2D,MaxPooling2D
from tensorflow.keras.activations import softmax
#from tensorflow.keras.utils import to_categorical
#import seaborn as sns
#import matplotlib.pyplot as plt
#from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D



img_data_array=[]
class_name=[]
#give path only till train folder
img_folder="C:\\Users\\luhar\\OneDrive\\Documents\\Code with ShiviSandy\\emotion_classifier\\archive\\train"
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




img_data_array=np.array(img_data_array)
img_data_array[0].shape




base_model = ResNet50(include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(8, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])





model.fit(img_data_array,class_name,verbose=2,validation_split=0.2,epochs=10,)
