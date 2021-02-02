#The goal of this exercise is to register two images only using their intensity values.
#For example, we have two CT images of the same anatomy and of the same patient taken in two different moments. There will be probably some displacement or rotation between the two images. If we take one of them as the reference image, we can register the second one so that they are aligned.
#This way, for example, it will be easier to evaluate anatomies or structures shown in the two images and their variation over time.

#If you didn't get from the lecture, an extremely simplified explanation on how image based registration works is this:

#    1: take two images, one differs from the other
#    2: have a similarity metric that tells you how much they are dis-/similar
#    3: move around one of the two images using some parameters (e.g. rotation angle, displacement in x, displacement in y)
#    4: compute the similarity between these two images (the one that was moved and the fixed one)
#    5: try really really hard to find the parameters for which the difference is minimal (or the similarity is maximal), i.e. run step 3 and 4 with difference parameters changing them using a good strategy.

from matplotlib.pyplot import imread
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

def load_dataset(image1, image2):
    # Load datasets from file
    fixed_image = imread(os.path.join('img', image1)).astype('double')
    moving_image = imread(os.path.join('img', image2)).astype('double')

    return fixed_image, moving_image

def plot_images(x, y):

    '''Function to display two images and their difference on screen.
    :param x: first image to display
    :param y: second image to display
    :return: void
    '''

    # Creating a figure with subplots of a certain size.
    fig, (plot1, plot2, plot3) = plt.subplots(1, 3, figsize=(10, 3))

    # Display the two images.
    plot1.imshow(x, cmap=plt.cm.gray)
    plot1.axis('off')
    plot2.imshow(y, cmap=plt.cm.gray)
    plot2.axis('off')

    # Computing the difference of the two images and display it.
    diff = x - y #La idea es ver aqui la diferencia entre las 2 imagenes para despues comparar la similitud
    plot3.imshow(diff, cmap=plt.cm.gray)
    plot3.axis('off')

    plt.show()

#Image manipulation

#To register the two images, we need to transform one of them.
# In this section we are going to write some functions to manipulate images, i.e. we need a function to rotate an image and one to translate it.
# To write them, you can make use of built-in function contained in OpenCV. No need to reinvent the wheel.

def translate_image(x, dx, dy):
    ''' Returns the matrix [x] translated along x and y of [dx] and [dy] respectively.
      :param x: numpy matrix to translate
      :param dx: displacement in x
      :param dy: displacement in y
      :return: the translated matrix
      '''

    cols, rows = x.shape  # size of the matrix.

    # A way to build a transformation is to manually enter its values.
    # Here we only need to fill the translational part of a 3x3 matrix.
    transform = np.float32([[1, 0, dx], [0, 1, dy]])

    # Transforms the image with the given transformation.
    # The last parameter gives the size of the output, we want it to be the same of the input.
    return cv2.warpAffine(x, transform, (cols, rows))


def rotate_image(x, angle):

    ''' Returns the matrix [x] rotated counter-clock wise by [angle].
    :param x: numpy matrix to rotate
    :param angle: angle of rotation in DEGREES
    :return: the rotated matrix
    '''

    cols, rows = x.shape  # size of the matrix.

    # Creates a rotation matrix to rotate around a rotation center of a certain angle.
    # In this case we rotate around the center of the image (cols / 2, rows / 2) by the given angle.
    # The last paramters is a scale factor, 1 means no scaling.
    transform = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)

    # Transforms the image with the given transformation.
    # The last parameter gives the size of the output, we want it to be the same of the input.
    return cv2.warpAffine(x, transform, (cols, rows))


def transform_image(x, angle, dx, dy):
    ''' Returns the matrix [x] rotated counter-clock wise by [angle] and translated by [dx] and [dy] in x and y respectively.
    :param x: numpy matrix to transform
    :param angle: angle of rotation in DEGREES
    :param dx:  displacement in x
    :param dy: displacement in y
    :return: the transformed matrix
    '''

    # We just concatenate the functions to rotate and translate, rotation is always done first.
    return translate_image(rotate_image(x, angle), dx, dy)


def transform_image_param(x, parameters):
    '''
    :param x: numpy matrix to transform
    :param parameters: array containing the 3 parameters for the transformation: angle, dx and dy
    :return: the transformed matrix
    '''

    # We just call transform_image unfolding the parameters
    return transform_image(x, parameters[0], parameters[1], parameters[2])


if __name__ == '__main__':

    fixed_image, moving_image = load_dataset('ct_fixed.png', 'ct_moving.png')

    plot_images(moving_image, fixed_image)

    # We load a ground truth, it is moving_image rotated by 20 degrees and translated by (5, 5)
    ground_truth_rotation = imread(os.path.join('img', 'ground_truth_rotation.png')).astype('double')
    # Let's test the function you just wrote, transform moving_image by the same values
    rotation_test = transform_image(moving_image, 20, 5, 5)
    # Check if your function is correct
    plot_images(rotation_test, ground_truth_rotation[:, :, 1])