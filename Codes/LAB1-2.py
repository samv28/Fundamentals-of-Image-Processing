import cv2
import matplotlib.pyplot as plt
grey_img = cv2.imread("D:/SIT/FIPL/Dataset/agri_0_1017.jpeg",0)
color_img = cv2.imread("D:/SIT/FIPL/Dataset/agri_0_1017.jpeg",1)[:, :,::-1]

print(grey_img)
#cv2.imshow('image',grey_img)
cv2.waitKey(0)
#cv2.imshow('image',color_img)
cv2.waitKey(0)
print(color_img.shape)

monalisa = cv2.imread("FIPL/Dataset/monalisa.jpg",1)
#cv2.imshow('image',monalisa[:,:,1])
cv2.waitKey(0)

from PIL import Image

I3 = Image.open('D:/SIT/FIPL/Dataset/raccoon.png')
#I3.show()

from skimage import io

I4 = io.imread('D:/SIT/FIPL/Dataset/raccoon.png')

#io.imshow(I4)
#plt.show() 

I5 = plt.imread('D:/SIT/FIPL/Dataset/water_body_1.jpg')
#plt.imshow(I5)
#plt.show()

import imageio.v3 as io
I6 = io.imread('D:/SIT/FIPL/Dataset/synthetic.jpg')
#plt.imshow(I6)
#plt.show() 

import SimpleITK as sitk

I7 = sitk.ReadImage('D:/SIT/FIPL/Dataset/synthetic.jpg')
nda=sitk.GetArrayFromImage(I7)
#plt.imshow(nda)
#plt.show()

import cv2 as cv

I= cv.imread('D:/SIT/FIPL/Dataset/monalisa.jpg')
#plt.imshow(I)
#plt.show() 

print(I.shape)

width = I.shape[1]
height = I.shape[0]

print(f'the width of the original image: {width}\nthe height of the original image: {height}')