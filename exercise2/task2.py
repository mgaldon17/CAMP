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
from task1 import saveImage, get4ngb, interpolate
from skimage import transform as skitr


