from scipy import fftpack
from matplotlib.colors import LogNorm
import cv2
import matplotlib.pyplot as plt
import numpy as np

city_noisy = cv2.imread('basic_filtering/img/city_noisy.jpg') #Image with gaussian noise

fig = plt.figure(figsize = (20,20))
f = np.fft.fft2(city_noisy.astype(np.float)) # This will compute the Fourier transform
fs = np.fft.fftshift(f)

#Original image

ax1 = fig.add_subplot(221)
ax1.imshow(city_noisy, interpolation = 'none', cmap = 'gray')
ax1.set_title('Original image')

#Filter image

ax2 = fig.add_subplot(222)
ax2.imshow(np.abs(fs), interpolation = 'none', cmap = 'gray', norm = LogNorm(vmin = 15))
ax2.set_title('Fourier transform')
plt.show()