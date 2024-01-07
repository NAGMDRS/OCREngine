import cv2
import numpy as np
from matplotlib import pyplot as plt

def sharpenImage(im):  # Need to check if sharpening input images would help accuracy
    kernel = np.ones((4, 2), np.float32) / 90 #(3,3) felt meh...
    im = cv2.filter2D(im, -1, kernel)  # -1 means depth of image remains unchanged
    im = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 1)
    im = 255 - im
    # cv2.imwrite("DATA/sharpend1.jpeg",im)
    return im
img = cv2.imread("DATA/word1.jpeg", 0)
img=np.expand_dims(img,axis=-1)


plt.imshow(img)
plt.show()