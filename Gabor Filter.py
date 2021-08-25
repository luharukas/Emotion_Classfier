import cv2
import numpy as np


ksize =20  #Use size that makes sense to the image and fetaure size. Large may not be good. 
    #On the synthetic image it is clear how ksize affects imgae (try 5 and 50)
sigma = 5 #Large sigma on small features will fully miss the features. 
theta = 1*np.pi/2 #/4 shows horizontal 3/4 shows other horizontal. Try other contributions
lamda = 1*np.pi/4  #1/4 works best for angled. 
gamma=0.9 #Value of 1 defines spherical. Calue close to 0 has high aspect ratio
    #Value of 1, spherical may not be ideal as it picks up features from other regions.
phi = 0.8  #Phase offset. I leave it to 0. (For hidden pic use 0.8)
kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
img = cv2.imread(r"C:\Users\luhar\emotion\emotion classifier\ck+\0\S005_001_00000001.png") #USe ksize:15, s:5, q:pi/2, l:pi/4, g:0.9, phi:0.8
cv2.imshow("fimg",img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
kernel_resized = cv2.resize(kernel, (400, 400))  # Resize image
cv2.imshow('df',fimg)
