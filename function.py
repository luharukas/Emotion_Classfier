# This whole program is written to preprocess the image after taking input from streamlit input.
from lib2to3.pytree import convert
import numpy as np
import cv2

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



def convertor(image):
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

    ksize =18  
    sigma = 1.5  
    theta = 3*np.pi/4 
    lamda = 5  
    gamma=1.5 
    phi = 0
    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
    kernel_resized = cv2.resize(kernel, (400, 400))  # Resize image

    img_data_array=[]
    image=np.array(image)
    image=cv2.resize(image,(126,126))
    ##------------------------------Apply Gabour filter here-----------------------------##
    image = cv2.filter2D(image, cv2.CV_8UC3, kernel)
    
    ##------------------------------Code for Gabour filter Complete here_------------------##
    image=np.array(image)
    image = image.astype('float32')
    image /= 255 
    img_data_array.append(image)
    img_data=np.array(img_data_array)
    return img_data

