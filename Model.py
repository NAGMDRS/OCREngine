import string
import cv2
import numpy as np
from matplotlib import pyplot as plt

recognizableChar = string.digits + string.ascii_letters + string.punctuation


def encodeText(txt):
    digitList = []
    for char in txt:
        try:
            digitList.append(recognizableChar.index(char))
        except:
            print(char)

    return digitList


size = (32, 128)


def reshapeForNN(img, size):
    (H, W) = size
    (h, w) = img.shape
    R = min(W / w, H / h)
    img = cv2.resize(img, (int(w * R), int(h * R)), interpolation=cv2.INTER_CUBIC)
    canvas = np.zeros(size)
    npImg = np.asarray(img)
    for j in range(0, int(w * R) - 1):
        for i in range(0, int(h * R) - 1):
            canvas[i][j] = max(canvas[i][j], npImg[i][j])
    cv2.imwrite("DATA/Test.jpeg", canvas)
    return canvas


img = cv2.imread("DATA/sharpend.jpeg", 0)
reshapeForNN(img, size)
