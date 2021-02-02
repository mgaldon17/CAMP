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

def saveImage(Image, name):
    plt.figure(figsize=(6, 6))
    plt.imshow(Image, cmap=plt.cm.gray, interpolation='nearest')
    plt.axis('off')
    plt.savefig(name)

def interpolate(ErrorMask, ImInput):
    ImErrorConcealed = np.zeros(ImInput.shape, dtype=ImInput.dtype)

    h, w = ImInput.shape
    for y in range(h):
        for x in range(w):
            if ErrorMask[y, x] == 0:
                ImErrorConcealed[y, x] = ImInput[y, x]
            else:
                neighbors = get4ngb(w, h, x, y)
                accu = 0
                counter = 0
                for p in neighbors:
                    if ErrorMask[p[1], p[0]] == 1:
                        continue
                    else:
                        accu += ImInput[p[1], p[0]]
                        counter += 1
                    if counter > 0:
                        ImErrorConcealed[y, x] = accu / counter
            # implement error concealment using interpolation here
    # display fixed image
    return ImErrorConcealed

def start():

    assignment_data = loadmat(os.path.join("data","image_transformations_assignment.mat"))
    assignment_data.keys()

    ImInput = assignment_data["ImInput"]
    ErrorMask = assignment_data["ErrorMask"]

    #Input
    saveImage(ImInput, 'ImInput')

    #ErrorMask
    saveImage(ErrorMask, 'ErrorMask')

    ImErrorConcealed = interpolate(ErrorMask, ImInput)

    saveImage(ImErrorConcealed, 'ImErrorConcealed')

    # display difference between fixed and ground truth

    saveImage(ErrorMask - ImErrorConcealed, 'difference')