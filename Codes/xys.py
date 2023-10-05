#dilation of 3x4 matrix
import cv2
import numpy as np
matrix = np.array([[0, 1, 0,1],[0, 1, 1,0],[0, 0, 1,0]])
matrix = matrix.astype('uint8')
structuring_element = np.array([[0, 1, 0],[0, 1, 1],[0, 0, 0]])
dilated_matrix = cv2.dilate(matrix, structuring_element, iterations=1,)
print(dilated_matrix)