import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_H = 480
IMAGE_W = 640

src = np.float32([[0, IMAGE_H], [IMAGE_W, IMAGE_H], [150, 200], [500, 200]])
dst = np.float32([[300, IMAGE_H], [400, IMAGE_H], [0, 0], [IMAGE_W, 0]])
M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
Minv = cv2.getPerspectiveTransform(dst, src) # Inverse transformation

img = cv2.imread('day.jpg') # Read the test img
img = cv2.resize(img, (640, 360))
img = img[200:(200+IMAGE_H), 0:IMAGE_W] # Apply np slicing for ROI crop
warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H)) # Image warping
plt.imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)) # Show results
plt.show()
