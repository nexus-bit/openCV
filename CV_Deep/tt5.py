import cv2
import numpy as np
test=cv2.imread('test.png', -1)
screen = test
screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY) 
screen = cv2.resize(screen, (80,60))
X = np.array([i for i in screen]).reshape(1, 80, 60,1)
np.save('test.npy', X)