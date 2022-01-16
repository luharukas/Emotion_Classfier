# import libraries
import cv2
import numpy as np

# Gabor Kernel Size
ksize =18

# Gaussian part standard deviation
sigma = 1.5

# Gabor function orientation of the normal to the parallel stripes. It controls the rotation of the ellipse.
theta = 3*np.pi/4

# The wavelength of the sinusoidal factor. Higher value means wider ellipse.
lamda = 5

# Ellipse Spatial aspect ratio.
gamma=1.8

# Sinusoid Phase shift.
phi = 0  

kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
# read Images
img = cv2.imread(r"C:\Users\luhar\Projects\Emotion_Classfier\dataset CK+\ck+\5\S081_002_00000024.png") 
cv2.imshow("fimg",img)
# Conversion into GreyScale images
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# images pass through gabor filter
fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
# Resize of Image
kernel_resized = cv2.resize(kernel, (400, 400)) 
# Dispaly images
cv2.imshow('df',fimg)
cv2.waitKey(5000)
