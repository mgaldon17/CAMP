"""
Given in the AssignmentData.mat file are input images (ImOrig) and a set of geometric scale factors Scales. Isotropically scale ImOrig using these scale factors and inverse the scaling operation to bring back the image to the original scale. Lets call this ImRescaled. Perform this rescaling using different interpolation techniques. Compare the rescaled image ImRescaled to the original image ImOrig to understand reconstruction error that the interpolation technique results in.

Interpolation Kernels to Explore: Nearest Neighbor, Bilinear, Bicubic, Lanczoz.

Functions to use: Write your own nearest and bilinear interpolation functions. For the rest, you can use the function we defined here
"""
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
from task1 import saveImage, get4ngb
from skimage import transform as skitr


def resizeBicubic(image, scalingFactor):
    im = Image.fromarray(image)
    (width, height) = (image.shape[0] * scalingFactor, image.shape[1] * scalingFactor)
    im_resized = im.resize((int(width), int(height)), resample=PIL.Image.BICUBIC)
    return np.asarray(im_resized)


def resizeLanczos(image, scalingFactor):
    im = Image.fromarray(image)
    (width, height) = (image.shape[0] * scalingFactor, image.shape[1] * scalingFactor)
    im_resized = im.resize((int(width), int(height)), resample=PIL.Image.LANCZOS)
    return np.asarray(im_resized)


def NN_interpolation(img, dstH, dstW):

    scrH, scrW = img.shape
    retimg = np.zeros((dstH, dstW, 3), dtype=np.uint8)

    for i in range(dstH-1):
        for j in range(dstW-1):
            scrx=round(i*(scrH/dstH))
            scry=round(j*(scrW/dstW))
            retimg[i,j]=img[scrx,scry]
    return retimg


def start():
    assignment_data = loadmat(os.path.join("data", "image_transformations_assignment.mat"))
    assignment_data.keys()

    image = assignment_data["ImInput"]
    image1 = NN_interpolation(image, image.shape[0] * 3, image.shape[1] * 3)

    saveImage(image1, 'image1')
    saveImage(image, 'image')


start()
