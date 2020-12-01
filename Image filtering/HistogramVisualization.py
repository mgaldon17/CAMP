import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack

# New in OpenCV? we can load an image using this method. All our images are in the folder img. The 0 means that
# we want a grayscale image (with only one channel).

def plot(image, fig, title, subplot, histogram):

    ax1 = fig.add_subplot(subplot)  # Adding a subplot for side by side view (1 row, two columns, our firs subplot)
    ax1.imshow(image, interpolation='none', cmap='gray')  # We show our cat
    ax1.set_title(title)  # Give a name to your Subplot
    if histogram:

        ax2 = fig.add_subplot(subplot+1)  # Adding a second subplot to our figure (1 row, two columns, our second subplot)
        ax2.hist(image.ravel(), bins=256)  # Here we add the histogram, but we could also show another image.
        ax2.set_title('Histogram')  # This is our histogram
        plt.show()  # Once we have are all set, we show the figure.
    else:
        plt.show()

    return fig