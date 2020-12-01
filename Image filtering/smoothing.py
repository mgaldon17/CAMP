# Exercise 2. Load the image tiger.jpg. Then apply the averaging filter.

# For this, create a 3×3(and 7×7) numpy matrix and fill it with the adequate values. This will be our kernel for the filter.
# Create a 3×3 numpy matrix and fill it with the values of the weighted averaging filter.
# Use the cv2.filter2D function to apply the convolution of the image with the filters.
# Show and compare the results.

import cv2
import numpy as np
import matplotlib.pyplot as plt

# TODO: Load your image and create your kernels
from sympy.printing.printer import Printer

from Plotter import Plotter

img_tiger = cv2.imread('basic_filtering/img/tiger.jpg')

# Filter kernels
avg_kernel1 = np.ones((3, 3), np.float32) / 9  # Divided by 9 for the filter values
avg_kernel7 = np.ones((7, 7), np.float32) / 49

wavg_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 16

# TODO: Apply the convolution operation and show your results
tiger_avg1 = cv2.filter2D(src=img_tiger, ddepth=-1, kernel=avg_kernel1)
tiger_avg2 = cv2.filter2D(src=img_tiger, ddepth=-1, kernel=avg_kernel7)

tiger_wavg = cv2.filter2D(src=img_tiger, ddepth=-1, kernel=wavg_kernel)
fig = plt.figure(figsize=(20, 20))

# Original image
images = []
images.append([img_tiger, "Image"])
printer = Plotter(images, False)
printer.plot()
ax1 = fig.add_subplot(221)
ax1.imshow(img_tiger, interpolation='none', cmap='gray')
ax1.set_title('Original image')

# Filter image

ax2 = fig.add_subplot(222)
ax2.imshow(tiger_avg1, interpolation='none', cmap='gray')
ax2.set_title('Image 3x3')

# Filter image

ax3 = fig.add_subplot(223)
ax3.imshow(tiger_avg2, interpolation='none', cmap='gray')
ax3.set_title('Image 7x7')

# Filter image

ax4 = fig.add_subplot(224)
ax4.imshow(tiger_wavg, interpolation='none', cmap='gray')
ax4.set_title('Image wavg')

plt.show()
