# Exercise 3. Use OpenCV to apply a Gaussian filter. Try with different values of sigma and compare the results.

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

# TODO: Apply the filter and show the results
img_tiger = cv2.imread('basic_filtering/img/tiger.jpg')

def gauss_filter(image, sigma):
    image_blurred = cv2.GaussianBlur(src=image, ksize=(0, 0), sigmaX = sigma)
    return image_blurred

if __name__ == '__main__':

    tiger_blurred = cv2.GaussianBlur(src=img_tiger, ksize=(0, 0), sigmaX=1)
    tiger_more_blurred = cv2.GaussianBlur(src=img_tiger, ksize=(0, 0), sigmaX=3)
    tiger_very_blurred = cv2.GaussianBlur(src=img_tiger, ksize=(0, 0), sigmaX=5)
    fig = plt.figure(figsize=(15, 15))

    # Original image

    ax1 = fig.add_subplot(221)
    ax1.imshow(img_tiger, interpolation='none', cmap='gray')
    ax1.set_title('Original image ''\u03C3'' = 0 ')

    # Filter image

    ax2 = fig.add_subplot(222)
    ax2.imshow(tiger_blurred, interpolation='none', cmap='gray')
    ax2.set_title('Image blurred ''\u03C3'' = 1')

    # Filter image

    ax3 = fig.add_subplot(223)
    ax3.imshow(tiger_more_blurred, interpolation='none', cmap='gray')
    ax3.set_title('Image more blurred ''\u03C3'' = 3')

    # Filter image

    ax4 = fig.add_subplot(224)
    ax4.imshow(tiger_very_blurred, interpolation='none', cmap='gray')
    ax4.set_title('Image very blurred ''\u03C3'' = 5')

    plt.show()
