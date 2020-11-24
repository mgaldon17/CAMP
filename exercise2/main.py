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

#%matplotlib inline
"""   
Problem Statement

Perform a comparative analysis of standard image interpolation techniques for different image processing tasks.

Task 1 - Error Concealment using Image Interpolation

Given in the data/image_transformations_assignment.mat file are input images (ImInput) and erroneous pixel mask (ErrorMask) where imaging errors were observed. These are given as Matlab MAT file, that can be loaded using the scipy.io.loadmat function, which returns a dictionary that contains all the needed data.
Perform image interpolation at the erroneous pixels using only the non-erroneous pixels surrounding them and generate image ImErrorConcealed.
Compare the error-concealed image ImErrorConcealed with ground truth image (ImGroundTruth) to compare the performance of your inter- polation technique.
Interpolation Techniques to Explore: Linear Interpolation

Functions to use: Write your own linear interpolation function.
"""

