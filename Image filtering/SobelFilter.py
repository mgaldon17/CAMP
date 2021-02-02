# Exercise 7. Apply the Sobel filter on the Charlie_Chaplin.jpg image.

#    Create a kernel for the horizontal and vertical gradient filters (remember exercise 2?)
#    Apply the convolution operation on the image with each filter.
#    Compute the magnitude of the gradient.
#    Show the gradient filters and the magnitude.

import cv2
import matplotlib.pyplot as plt
import numpy as np

img_chaplin = cv2.imread('basic_filtering/img/Charlie_Chaplin.jpg', 0)

fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(111)
ax1.imshow(img_chaplin, interpolation='none', cmap='gray')
ax1.set_title('Original image')
plt.show()

# TODO: Exercise 7!

#Vertical and horizontal approximations of the derivative of the intensity
sobel_kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

gx = cv2.filter2D(src=img_chaplin.astype(np.float), ddepth=-1, kernel=sobel_kernel_x)
gy = cv2.filter2D(src=img_chaplin.astype(np.float), ddepth=-1, kernel=sobel_kernel_y)

# We apply a kernel that can have negative values. That's why we change the type of the image to a float

g_magnitude = np.sqrt(np.power(gx, 2), np.power(gy, 2))

fig = plt.figure(figsize=(20, 20))
# Original image

ax1 = fig.add_subplot(221)
ax1.imshow(img_chaplin, interpolation='none', cmap='gray')
ax1.set_title('Original image')

# Filter image

ax3 = fig.add_subplot(222)
ax3.imshow(gx, interpolation='none', cmap='gray')
ax3.set_title('"$G_{x}$"')

# Filter image

ax4 = fig.add_subplot(223)
ax4.imshow(gy, interpolation='none', cmap='gray')
ax4.set_title("$G_{y}$")

# Filter image

ax5 = fig.add_subplot(224)
ax5.imshow(g_magnitude, interpolation='none', cmap='gray')
ax5.set_title('Magnitude G')

plt.show()