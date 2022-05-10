# 'CV2' :- Module to do Computer Vision Task
import cv2
# 'NUMPY' :- MOdule to do operation on array
import numpy as np

import os
# 'PANDAS':- Modules to do operation on Dataframes
import pandas as pd
# 'SKlearn' :- To do operation on Datasets and taking other Algorithm for training
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# 'MATPLOTLIB':- Module to draw graph
import matplotlib.pyplot as plt

#Importing functions from tensorflow for building and training the model
# It is used to do operation on NN models. 
import tensorflow as tf
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Flatten,InputLayer,Conv2D,MaxPooling2D,Dropout,MaxPool2D,Softmax,ReLU,GlobalAveragePooling2D
from tensorflow.keras.activations import softmax
from tensorflow.keras.utils import to_categorical

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# NORMALIZATION OF IMAGES
def mat2gray(img):
    A = np.double(img)
    out = np.zeros(A.shape, np.double)
    # MIN_MAX Normalization
    normalized = cv2.normalize(A, out, 1.0, 0.0, cv2.NORM_MINMAX)
    return out
# ADDED GAUSSIAN RANDOM NOISE FOR AGUMENTATION
def random_noise(image, mode='gaussian', seed=None, clip=True, **kwargs):
    image = mat2gray(image)
    mode = mode.lower()
    if image.min() < 0:
        low_clip = -1
    else:
        low_clip = 0
    if seed is not None:
        np.random.seed(seed=seed)
    if mode == 'gaussian':
        noise = np.random.normal(kwargs['mean'], kwargs['var'] ** 0.5,image.shape)        
        out = image  + noise
    if clip:        
        out = np.clip(out, low_clip, 1.0)
    return out


# Parameters for Gabor Filter

ksize =18  
sigma = 1.5  
theta = 3*np.pi/4 
lamda = 5  
gamma=1.5 
phi = 0

kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
kernel_resized = cv2.resize(kernel, (400, 400))  # Resize image


img_data_array=[]
class_name=[]
#give path only till train folder
# Load Dataset from give Folder 
img_folder=r"C:\Users\luhar\Projects\Emotion_Classfier\dataset CK+\ck+"
for dir1 in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir1)):
            image_path= os.path.join(img_folder, dir1,  file)
            image= cv2.imread( image_path)
            image=cv2.resize(image,(126,126))
            
            ##------------------------------Apply Gabour filter here-----------------------------##
            image = cv2.filter2D(image, cv2.CV_8UC3, kernel)
            
            ##------------------------------Code for Gabour filter Complete here_------------------##
            image=np.array(image)
            image = image.astype('float32')
            image /= 255 
            img_data_array.append(image)
            img1 = random_noise(image,'gaussian', mean=0.1,var=0.1)
            img_data_array.append(img1)
            class_name.append(dir1)
            class_name.append(dir1)

# Label Encoding
label_encoder = preprocessing.LabelEncoder()
class_name= label_encoder.fit_transform(class_name)

img_data_array=np.array(img_data_array)
img_data_array[0].shape
# Train-Test Split
x_train,x_test,y_train,y_test=train_test_split(img_data_array,class_name,test_size=0.15,shuffle=True,random_state=42)

# Print shape of training and testing dataset
print(x_train.shape)
print(x_test.shape)


# Creating the architecture of CNN
model=Sequential([
        Conv2D(32,kernel_size=(5,5),activation="relu",padding="SAME",input_shape=img_data_array[0].shape),
        Conv2D(64,kernel_size=(5,5),activation="relu",padding="SAME"),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(128,kernel_size=(3,3),activation="relu",padding="SAME"),
        Dropout(0.5 ),
        MaxPooling2D(pool_size=(2,2)),
        Flatten(),
        Dense(1024,activation="relu"),
        Dropout(0.5),
        Dense(512,activation="relu"),
        Dropout(0.5),
        Dense(7,activation="softmax")
    ])
# COmpile the model with optimizer and loss function 
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history=model.fit(x_train,y_train,validation_split=0.1,epochs=30,shuffle=True,)
frame = pd.DataFrame(history.history)
print(frame)
model.save('my_model')
    