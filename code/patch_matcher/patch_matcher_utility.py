import numpy as np
from utils.glob_def import CONFIG_DIR
import yaml
import matplotlib.pyplot as plt
import os


def conv2d(image: np.array, kernel: np.array) -> np.array:
    """
    Calculates 2D convolution

    :param image: Input image for convolution
    :param kernel: Kernel used for image convolution
    :return: Convolved image with kernel
    """

    # ged padding dimensions
    pad_w = int(np.floor((kernel.shape[1] - 1)/2))
    pad_h = int(np.floor((kernel.shape[0] - 1)/2))
    # flip kernel
    kernel = np.flip(kernel)
    # pad image
    img_pad = np.pad(image, pad_width=((pad_h, pad_h), (pad_w, pad_w)), mode='symmetric')
    # get shapes
    sub_shape = kernel.shape
    view_shape = tuple(np.subtract(img_pad.shape, sub_shape) + 1) + sub_shape
    # get image strides
    strides = img_pad.strides + img_pad.strides
    # generate sub matrices
    sub_matrices = np.lib.stride_tricks.as_strided(img_pad, view_shape, strides)
    # perform convolution
    convolved_img = np.einsum('ij,klij->kl', kernel, sub_matrices)

    return convolved_img


def get_2d_gauss_kernel(size: int, sigma: float) -> np.array:
    """
    Return 2d gaussian kernel for specified size and sigma

    :param size: Size of output kernel
    :param sigma: Standard deviation of gaussian distribution
    :return: Gaussian kernel (size x size)
    """

    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))

    return g/g.sum()


def return_non_maximum_suppression_matrix_r(R: np.array, neighbourhood: int, n_points: int = -1) -> np.array:
    """
    Return suppressed non maximum local values in matrix R

    :param R: Input matrix
    :param neighbourhood: Local neighbourhood around pixel
    :param n_points: Maximum local maximums in R which will be returned (first n_points with maximum response value)
    :return: Matrix R with suppressed local maximus
    """

    # extract key points
    key_points_indeces = np.where(R > 0)
    # get key points list
    list_key_points = [(key_points_indeces[1][i], key_points_indeces[0][i], R[key_points_indeces[0][i], key_points_indeces[1][i]]) for i in
                       np.arange(0, key_points_indeces[0].shape[0])]
    # filter local maximums
    R = local_non_maximum_suppression(R, list_key_points, neighbourhood, n_points)

    return R


def local_non_maximum_suppression(R: np.array, list_key_points: list, neighbourhood: int, n_points: int = -1) -> np.array:
    """
    Return suppressed non maximum local values in matrix R

    :param R: Input matrix
    :param list_key_points: List of key points each represented by tuple (x, y, response)
    :param neighbourhood: Local neighbourhood around pixel
    :param n_points: Maximum local maximums in R which will be returned (first n_points with maximum response value)
    :return: Matrix R with suppressed local maximus
    """

    # sort key points by their response
    list_key_points.sort(reverse=True, key=lambda x: x[2])
    # get copy of response matrix R
    R_new = R.copy()
    # iterate through key points
    for key_point in list_key_points:
        # get key point position
        i = key_point[1]
        j = key_point[0]
        # check if this point is not filtered
        if R_new[i, j] == 0:
            continue

        # nullify all neighbours around local maximum
        min_i = max(i - neighbourhood, 0)
        max_i = min(i + neighbourhood + 1, R.shape[0])
        min_j = max(j - neighbourhood, 0)
        max_j = min(j + neighbourhood + 1, R.shape[1])
        R_new[min_i:max_i, min_j:max_j] = 0
        R_new[i, j] = key_point[2]

    # take just first n_points from non filtered key points (nullify other responses in R)
    if n_points != -1:
        counter = 0
        for key_point in list_key_points:
            i = key_point[1]
            j = key_point[0]
            if R_new[i, j] != 0:
                counter += 1

            if counter > n_points:
                R_new[i, j] = 0

    return R_new


def get_ransac_params(num_matched_points: int):
    """
    Return RANSAC parameters depending on matched points by patch matcher

    :param num_matched_points: Number of matched points
    :return: Ransac parameters n_fit and n_trials
    """

    n_fit = 2
    n_trials = 5
    # if number of matched points is greater then 5 return different params
    if num_matched_points > 5:
        n_fit = 4
        n_trials = 10

    return n_fit, n_trials


def ransac_trial(fit_H, pt1, pt2, max_error):
    """
    Find the outliers using in fitted function H

    :param fit_H: Fitted model
    :param pt1: Patch key points
    :param pt2: Template key points
    :param max_error: Maximum error from model to be inlier
    :return: Indices of inliers
    """

    # init inliers indices
    inds = np.zeros(pt1.shape[0], dtype=bool)
    for i, pt in enumerate(pt1):
        # transform top left corner of patch (coordinate 0,0)
        pt_transformed = np.matmul(pt, fit_H)
        # calculate distance of point to model estimation
        dist = np.sqrt(np.sum((pt_transformed - pt2[i])**2))
        # check if distance is in required model area
        if dist < max_error:
            inds[i] = 1

    return inds


def ransac_filter(pt1, pt2, match, n_fit, n_trials):
    """
    RANSAC filter implementation for finding best match points
    and filtering out outliers

    :param pt1: Patch key points
    :param pt2: Template key points
    :param match: Array of matched points
    :param n_fit: Minimum number of points required to fit the model
    :param n_trials: Number of RANSAC trials
    :return: Best match points
    """

    # check if there is more then 2 points
    if match.shape[0] < 3:
        return match

    # initiate RANSAC
    best_match = np.array([])
    best_n_inliers = 0

    # loop over trials
    for i in np.arange(0, n_trials):
        # take random indices
        inds = np.random.choice(pt1.shape[0], size = n_fit, replace=False)
        # fit model
        fit_H = compute_affine_matrix(pt1[inds, :], pt2[inds, :])
        # ransac trial - check model performance
        inlier_inds = ransac_trial(fit_H, pt1, pt2, 30)
        # if we found better model save best match
        if np.sum(inlier_inds) > best_n_inliers:
            best_match = match[inlier_inds]
            best_n_inliers = np.sum(inlier_inds)
            # if all points are inliers return best match
            if best_n_inliers == pt1.shape[0]:
                return best_match

    return best_match


def compute_affine_matrix(pt1: np.array, pt2: np.array) -> np.array:
    """
    Compute affine matrix H such as patchKP * H = templateKP

    :param pt1: Patch key points
    :param pt2: Template key points
    :return: H matrix
    """

    # compute translation matrix
    H = np.zeros((3, 2))
    residuals = pt2 - pt1[:, 0:2]

    # because we can have just translation optimal solution is mean difference
    mean_residuals = np.mean(residuals, axis=0)
    H[0, 0] = 1
    H[2, 0] = mean_residuals[0]
    H[1, 1] = 1
    H[2, 1] = mean_residuals[1]

    return H


def first_and_second_smallest(numbers: np.array):
    """
    Find first and second smallest number in array

    :param numbers: Array of numbers
    :return: First and second smallest number in array ans its index
    """
    m1 = m2 = float('inf')
    i1 = i2 = -1

    for i, x in enumerate(numbers):
        if x <= m1:
            m2 = m1
            m1 = x
            i2 = i1
            i1 = i
        elif x < m2:
            m2 = x
            i2 = i

    return m1, i1, m2, i2


def debug_image(image):
    """
    Shows input image

    :param image: Input image
    """
    image_copy = image.copy().astype('float64')
    image_copy /= image_copy.max()
    image_copy *= 255.0
    plt.imshow(image_copy, cmap='gray', vmin=0, vmax=255)
    plt.show()


def load_config():
    """
    Load config file

    :return: Loaded yml config file
    """
    # read config file from directory
    cfg_path = os.path.join(CONFIG_DIR, "patch_match_cfg.yml")
    # open .yml file
    cfg_file = open(cfg_path, "r")
    config = yaml.safe_load(cfg_file)

    return config
