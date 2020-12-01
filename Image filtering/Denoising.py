## Denoising

# We can use the filters to reduce the effects of noise in the image, however, as we saw before, this will also affect some elements of the image.

# **Exercise 4.** Now we will see the effects of the filters on a noisy image.
# * Load the image `city.jpg`.
# * Write a function to add salt and pepper noise to the image. To do this, just randomly select a small portion of the pixels of the image and (again, randomly) change its value to 0 or 255.
# * Show the results (clean and noisy image).
import numpy as np
import cv2
import matplotlib.pyplot as plt
from GaussianFilter import gauss_filter

def salt_pepper_noise(img):
    # TODO Create your noisy image here

    salt = np.random.uniform(low=0.0, high=1.0, size=img.shape)
    pepper = np.random.uniform(low=0.0, high=1.0, size=img.shape)

    noisy_image = np.copy(img)

    noisy_image[np.where(salt > 0.8)] = 255
    noisy_image[np.where(pepper > 0.9)] = 0

    return noisy_image

if __name__ == '__main__':

    # TODO: Load and add noise to your image. Show the results.
    img_city = cv2.imread('basic_filtering/img/city.jpg')
    noisy_city = salt_pepper_noise(img_city)
    fig = plt.figure(figsize = (15,15))#Original image

    #Original image
    ax1 = fig.add_subplot(121)
    ax1.imshow(img_city, interpolation = 'none', cmap = 'gray')
    ax1.set_title('Original image')

    #Filter image

    ax2 = fig.add_subplot(122)
    ax2.imshow(noisy_city, interpolation = 'none', cmap = 'gray')
    ax2.set_title('Image with noise')

    plt.show()


