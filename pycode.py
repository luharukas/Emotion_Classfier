#importing the modules
import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
import tensorflow.keras as k
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,InputLayer
from tensorflow.keras.activations import softmax
import seaborn as sns
import matplotlib.pyplot as plt
#Importing functions from tensorflow for preprocessinga and building the model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing

# IMAGE_DIR=[
#             "C:\\Users\\luhar\\OneDrive\\Documents\\Code with ShiviSandy\\Project x\\archive\\train\\angry",
#             "C:\\Users\\luhar\\OneDrive\\Documents\\Code with ShiviSandy\\Project x\\archive\\train\\contempt",
#             "C:\\Users\\luhar\\OneDrive\\Documents\\Code with ShiviSandy\\Project x\\archive\\train\\disgust",
#             "C:\\Users\\luhar\\OneDrive\\Documents\\Code with ShiviSandy\\Project x\\archive\\train\\fear",
#             "C:\\Users\\luhar\\OneDrive\\Documents\\Code with ShiviSandy\\Project x\\archive\\train\\happy",
#             "C:\\Users\\luhar\\OneDrive\\Documents\\Code with ShiviSandy\\Project x\\archive\\train\\neutral",
#             "C:\\Users\\luhar\\OneDrive\\Documents\\Code with ShiviSandy\\Project x\\archive\\train\\sad",
#             "C:\\Users\\luhar\\OneDrive\\Documents\\Code with ShiviSandy\\Project x\\archive\\train\\surprise",
# ]
# labels=[
#         'anger',
#         'contempt',
#         'disgust',
#         'fear',
#         'happy',
#         'neutral',
#         'sad',
#         'surprise',
# ]
# data=np.zeros([48,48,3])
# # img=cv2.imread("C:\\Users\\luhar\\OneDrive\\Documents\\Code with ShiviSandy\\Project x\\archive\\train\\angry\\S010_004_00000017.png")
# # print(img.shape)
# # data=np.stack((data,img))
# # print(data.shape)

# label=[]
# i=0
# for j in IMAGE_DIR:
#   for filename in os.listdir(j):
#     path=j+"\\"+filename
#     print(path)
#     img=cv2.imread(path)
#     print(img.shape)
#     data=np.stack((data,img))
#     if img.all()==None:
#             print("yes")
#     label.append(i)
#     print(filename)
#   i+=1
# # label=to_categorical(label,dtype='int')
# # dir={'Images':data,'anger':label[:,0],'contempt':label[:,1],'disgust':label[:,2],"fear":label[:,3],'happy':label[:,4],'neutral':label[:,5],'sad':label[:,6],'surprise':label[:,7]}
# # train_df=pd.DataFrame(dir)
# print(data.shape)




# # def get_train_generator(df, x_col, y_cols, shuffle=True, batch_size=8, seed=1, target_w = 48, target_h = 48):
  
# #     print("getting train generator...") 
# #     image_generator = ImageDataGenerator(samplewise_std_normalization=0,samplewise_center=1)
# #     generator = image_generator.flow_from_dataframe(
# #             dataframe=df,
# #             directory="C:\\Users\\luhar\\OneDrive\\Documents\\Code with ShiviSandy\\Project x\\archive\CK+48\\full image",
# #             x_col=x_col,
# #             y_col=y_cols,
# #             class_mode="raw",
# #             batch_size=batch_size,
# #             shuffle=shuffle,
# #             seed=seed,
# #             target_size=(target_w,target_h),)
    
# #     return generator


# # train_generator = get_train_generator(train_df, "Images", labels)

# # base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(48,48, 3))

# # x = base_model.output
# # x = GlobalAveragePooling2D()(x)
# # predictions = Dense(len(labels), activation="sigmoid")(x)

# # model = Model(inputs=base_model.input, outputs=predictions)
# # model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
# # history = model.fit_generator(train_generator, 
# #                               steps_per_epoch=100, 
# #                               epochs = 5)


test_img_data_array=[]
test_class_name=[]
#giving the path till the train folder
img_folder="C:\\Users\\luhar\\OneDrive\\Documents\\Code with ShiviSandy\\emotion_classifier\\archive\\test"
#Pre-processing each image from the folder
for dir1 in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir1)):
            image_path= os.path.join(img_folder, dir1,  file)
            image= cv2.imread( image_path)
            image=np.array(image)
            image = image.astype('float32')
            image /= 255 
            test_img_data_array.append(image)
            test_class_name.append(dir1)
label_encoder = preprocessing.LabelEncoder()
#Performing fit and transform on the input data to transform the data points
class_name= label_encoder.fit_transform(test_class_name)



#Converting images to numpy arrays
img_data_array=np.array(test_img_data_array)
print(img_data_array[0].shape)
print(len(test_class_name))
print(len(test_img_data_array))
