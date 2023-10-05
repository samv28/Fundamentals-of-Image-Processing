import matplotlib.pyplot as plt
import cv2
i = cv2.imread('D:/SIT/FIPL/Dataset/gtr.jpg')
print(i.shape)


hist = cv2.calcHist([i],[0],None,[256],[0,256])
plt.hist(i.ravel(),256,[0,256])

plt.show()

img = cv2.imread('D:/SIT/FIPL/Dataset/gtr.jpg')
color = ('b','g','r')

for i,col in enumerate(color):
 histr = cv2.calcHist([img],[i],None,[256],[0,256])
 plt.plot(histr,color = col)
 plt.xlim([0,256])

plt.show()


plt.subplot(2,2,1)
plt.imshow(i)
plt.title('original Image')
plt.xlabel('x')
plt.ylabel('y')
plt.subplot(2,2,2)
plt.hist(i.ravel(),256, [0,256])
plt.title('Histogram of Original Image')
plt.xlabel('Gray Levels')
plt.ylabel('Frequency of Occurance')
plt.subplot(2,2,3)
plt.imshow(i,cmap='gray')
plt.title('Histogram Equalized Image')
plt.subplot(2,2,4)
plt.hist(i.ravel(),256, [0,256])
plt.title('Equalized Histogram')
plt.show()