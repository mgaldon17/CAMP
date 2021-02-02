#TODO: Exercise 6 goes here!
import cv2
from Denoising import salt_pepper_noise as denoise
import matplotlib.pyplot as plt
from GaussianFilter import gauss_filter

img_city = cv2.imread('basic_filtering/img/city.jpg')
noisy_city = cv2.imread('basic_filtering/img/city_noisy.jpg') #Image with gaussian noise

city_gaussian = gauss_filter(noisy_city, 3)
city_bileteral = cv2.bilateralFilter(src= noisy_city, d = -1, sigmaColor = 30, sigmaSpace = 6)

#Original image
fig = plt.figure(figsize = (20,20))
ax1 = fig.add_subplot(221)
ax1.imshow(img_city, interpolation = 'none', cmap = 'gray')
ax1.set_title('Original image')

#Filter image
ax2 = fig.add_subplot(222)
ax2.imshow(noisy_city,  interpolation = 'none', cmap = 'gray')
ax2.set_title('Noisy image')
#Filter image

ax3 = fig.add_subplot(223)
ax3.imshow(city_bileteral, interpolation = 'none', cmap = 'gray')
ax3.set_title('Bilateral image')

#Filter image

ax4 = fig.add_subplot(224)
ax4.imshow(city_gaussian, interpolation = 'none', cmap = 'gray')
ax4.set_title('Gaussian filter')

plt.show()