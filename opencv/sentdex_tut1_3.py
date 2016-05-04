import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('input_images/set10_1.jpg',cv2.IMREAD_GRAYSCALE)

#IMREAD_GRAYSCALE 0
#IMREAD_COLOR = 1
#IMREAD_UNCHANGED = -1

cv2.imwrite('test.png', img)

