# SSD: sum of squared differences
from image_based_registration import *
import scipy.optimize as opt


def ssd(x, y):
    ''' Returns the sum of the squared differences of the given matrices
    '''
    n, m = y.shape
    ssd = (1 / (n * m)) * np.sum(np.square(x - y))
#    ground_truth_ssd = 1514.2839965820312
#    print("Your SSD answers with: {}. Jack says: {}".format(ssd, ground_truth_ssd) + '\n')
    return ssd


# SAD: sum of differences

def sad(x, y):
    n, m = y.shape
    ground_truth_sad = 19.36151123046875
#    sad = (1 / (n * m)) * np.sum(np.abs(x - y))
#    print("Your SAD answers with: {}. Jack says: {}".format(sad, ground_truth_sad) + '\n')
    return sad


# NCC: Normalized cross correlation

def ncc(x, y):
    n, m = y.shape
    # Demeaning the matrices.
    x -= np.mean(x)
    y -= np.mean(y)

    ncc = (1 / (n * m)) * np.sum(x * y / (x.std() * y.std()))
#   ground_truth_ncc = 0.802228403920768
#    print("Your SNCC answers with: {}. Jack says: {}".format(ncc, ground_truth_ncc) + '\n')
    return ncc


def mi(x, y):
    # Compute joint histogram. You may want to use np.histogram2d for this.
    # histogram2d takes 1D vectors, ravel() reshape the matrix to a 1D vector.
    # Don not forget to normalize the histogram in order to get a joint distribution.

    jh = np.histogram2d(x.ravel(), y.ravel())[0]
    jh = jh + np.finfo(float).eps  # add eps for stability

    # Normalize.
    sh = np.sum(jh)
    jh /= sh

    # You can get the individual distributions by marginalization of the joint distribution jh.
    # We have two random variables x and y whose joint distribution is known,
    # the marginal distribution of X is the probability distribution of X,
    # when the value of Y is not known. And vice versa.
    s1 = np.sum(jh, axis=0, keepdims=True)
    s2 = np.sum(jh, axis=1, keepdims=True)
    # Compute the MI.
    MI = np.sum(-s1 * np.log2(s1)) + np.sum(-s2 * np.log2(s2)) - np.sum(-jh * np.log2(jh))
#    ground_truth_mi = 0.74620104021828171
#    print("Your MI answers with: {}. Jack says: {}".format(MI, ground_truth_mi))

    return MI


def cost_function(transform_params, fixed_image, moving_image, similarity):
    ''' Computes a similarity measure between the given images using the given similarity metric
    :param transform_params: 3 element array with values for rotation, displacement along x axis, dislplacement along y axis.
                             The moving_image will be transformed using these values before the computation of similarity.
    :param fixed_image: the reference image for registration
    :param moving_image: the image to register to the fixed_image
    :param similarity: a string naming the similarity metric to use. e.g. SSD, SAD, ...
    :return: the compute similarity
    '''
    # Build transform parameters
    angle = transform_params[0]
    dx = transform_params[1]
    dy = transform_params[2]

    # Transform the moving_image with the current parameters (We already have code for this)
    transformed_moving_img = transform_image(moving_image, angle, dx, dy)

    d_moving_image = transformed_moving_img.astype('double')

    # Compute the similarity value using the given method.
    #
    if similarity == "SSD":
        s = ssd(transformed_moving_img, fixed_image)
    elif similarity == "SAD":
        s = sad(transformed_moving_img, fixed_image)
    elif similarity == "NCC":  # Since we want to maximize NCC, we can minimize its negative
        s = -ncc(transformed_moving_img, fixed_image)
    elif similarity == "MI":
        s = -mi(transformed_moving_img, fixed_image)  # Since we want to maximize MI, we can minimize its negative
    else:
        print("Wrong similarity measure given.")
        return -1
    return s


if __name__ == '__main__':
    fixed_image, moving_image = load_dataset('ct_fixed.png', 'ct_moving.png')
    #We have now an optimization problem, we want to find some values (the transform_params in the previous function) that minimize/maximize the dis-/similarity between the two images.
    #There are different approaches to solve the problem, a classical one is using a simplex method.
    #To do so, we are going to use the 'fmin' function from Scipy. It implements the Nelder-Mead algorithm
    # Give some initial values to the transformation parameters. Vector de transformacion de la fixed image
    x0 = [13, -15, -15]

    result_params = opt.fmin(cost_function, x0, args=(fixed_image, moving_image, 'SSD'))
    print(result_params)
    # Transform the moving images with the found parameters
    result_image = transform_image_param(moving_image, result_params)
    # Let's have a look at the result!

    plot_images(result_image, fixed_image)
    #Imagen transformada, imagen original, diferencia de imagenes por el metodo dado