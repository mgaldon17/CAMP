#Exercise 5. Now that we have noise in our image, use OpenCV to apply a Gaussian, Averaging, and Median filter to the (noisy) image.
# Compare the results with the clean input. Show your results
import numpy as np
import cv2
import matplotlib.pyplot as plt
from GaussianFilter import gauss_filter
from Denoising import salt_pepper_noise as denoise

img_city = cv2.imread('basic_filtering/img/city.jpg')
noisy_city = denoise(img_city)

img_gaussian = gauss_filter(noisy_city, 1)
img_mean = cv2.blur(src = noisy_city, ksize=(5,5))
img_meadian = cv2.medianBlur(src = noisy_city, ksize = 5)
fig = plt.figure(figsize = (20,20))

#Original image

ax1 = fig.add_subplot(221)
ax1.imshow(noisy_city, interpolation = 'none', cmap = 'gray')
ax1.set_title('Noisy image')

#Mean filter image

ax2 = fig.add_subplot(222)
ax2.imshow(img_mean, interpolation = 'none', cmap = 'gray')
ax2.set_title('Mean image')

#Median filter image

ax2 = fig.add_subplot(223)
ax2.imshow(img_meadian, interpolation = 'none', cmap = 'gray')
ax2.set_title('Median image')

#Gaussian filter image

ax2 = fig.add_subplot(224)
ax2.imshow(img_gaussian, interpolation = 'none', cmap = 'gray')
ax2.set_title('Gaussian filter')

plt.show()