import cv2
import matplotlib.pyplot as plt

# Read a grayscale image
image = cv2.imread('D:/SIT/FIPL/Dataset/r34.jpg', cv2.IMREAD_GRAYSCALE)


# Calculate the histogram
hist = cv2.calcHist([image], [0], None, [256], [0, 256])

# Perform histogram equalization
equalized_image = cv2.equalizeHist(image)

# Plot the original image and its histogram
plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(2, 2, 2)
plt.hist(image.ravel(), 256, [0, 256])
plt.title('Histogram')

# Plot the equalized image and its histogram
plt.subplot(2, 2, 3)
plt.imshow(equalized_image, cmap='gray')
plt.title('Equalized Image')

plt.subplot(2, 2, 4)
plt.hist(equalized_image.ravel(), 256, [0, 256])
plt.title('Equalized Histogram')

# Display the plots
plt.tight_layout()
plt.show()
