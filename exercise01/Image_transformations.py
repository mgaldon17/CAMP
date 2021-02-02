import os
import numpy as np
import scipy as sp
from functools import partial
from collections import namedtuple
from scipy import misc as spm
from scipy.io import loadmat
from skimage.transform import resize
from scipy.spatial.distance import euclidean
from scipy import interpolate as spi
import matplotlib.pyplot as plt
from PIL import Image
import PIL
from skimage import img_as_float

from skimage import transform as skitr

# turn of numpy warnings for now..
import warnings

warnings.filterwarnings("ignore")


# Task 1 - Error Concealment using Image Interpolation¶

# Given in the data/image_transformations_assignment.mat file are input images (ImInput) and erroneous pixel mask (ErrorMask) where imaging errors were observed. These are given as Matlab MAT file, that can be loaded using the scipy.io.loadmat function, which returns a dictionary that contains all the needed data.

# Perform image interpolation at the erroneous pixels using only the non-erroneous pixels surrounding them and generate image ImErrorConcealed.

# Compare the error-concealed image ImErrorConcealed with ground truth image (ImGroundTruth) to compare the performance of your inter- polation technique.

def error_concealment():
    assignment_data = loadmat(os.path.join("data", "image_transformations_assignment.mat"))
    assignment_data.keys()
    ImInput = assignment_data["ImInput"]
    ImOrig = assignment_data["ImOrig"]
    ErrorMask = assignment_data["ErrorMask"]
    ImGroundTruth = assignment_data["ImGroundTruth"]
    Scales = assignment_data["Scales"][0]
    plt.figure(figsize=(6, 6))
    plt.imshow(ImInput, cmap=plt.cm.gray, interpolation='nearest')
    plt.axis('off')
    plt.figure(figsize=(6, 6))
    plt.imshow(ErrorMask, cmap=plt.cm.gray, interpolation='nearest')
    plt.axis('off')
    ErrorMask.shape

    def get4ngb(rows, cols, x, y):
        # function ngb = get4ngb(rows, cols, x, y)
        #
        # Input:
        # rows, cols    Number of rows and colums from the 2D Image
        # x,y           Coordinates of the center point
        #
        # Output:
        # ngb           list of possible neighbours

        # initialize empty result list
        ngb = list()

        # east
        if x > 1:
            ngb += [[x - 1, y]];

        # north
        if y < cols - 1:
            ngb += [[x, y + 1]];

        # west
        if x < rows - 1:
            ngb += [[x + 1, y]];

        # south
        if y > 1:
            ngb += [[x, y - 1]];

        return ngb

    ImErrorConcealed = np.zeros(ImInput.shape, dtype=ImInput.dtype)

    h, w = ImInput.shape
    for y in range(h):
        for x in range(w):
            # if pixel is ok, then just store it
            if ErrorMask[y, x] == 0:
                ImErrorConcealed[y, x] = ImInput[y, x]
            else:
                # look up neighboring pixel coordinates
                points = get4ngb(w, h, x, y)
                # search for pixels that are not in error mask and are within the image boundaries
                v = 0.
                c = 0
                for p in points:
                    if ErrorMask[p[1], p[0]] == 1:
                        continue
                    v += ImInput[p[1], p[0]]
                    c += 1
                # since all distances are equal (we always use array-indices),
                # we can just divide by the number of samples we used to recreate the missing pixel
                if c > 0:
                    v /= c
                ImErrorConcealed[y, x] = v
    plt.figure(figsize=(6, 6))
    plt.imshow(ImErrorConcealed, cmap=plt.cm.gray, interpolation='nearest')
    plt.axis('off')

    plt.figure(figsize=(6, 6))
    plt.imshow(ImErrorConcealed - ImGroundTruth, cmap=plt.cm.gray, interpolation='nearest')
    plt.axis('off')

    plt.show()
if __name__ == '__main__':
    error_concealment()
