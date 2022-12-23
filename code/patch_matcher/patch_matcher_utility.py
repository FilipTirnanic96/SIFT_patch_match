import numpy as np
from utils.glob_def import CONFIG_DIR
import yaml
import matplotlib.pyplot as plt
import os


def conv2d(img, kernel):
    pad_w = int(np.floor((kernel.shape[1] - 1)/2))
    pad_h = int(np.floor((kernel.shape[0] - 1)/2))

    kernel = np.flip(kernel)

    img_pad = np.pad(img, pad_width=((pad_h, pad_h), (pad_w, pad_w)), mode='symmetric')
    sub_shape = kernel.shape
    view_shape = tuple(np.subtract(img_pad.shape, sub_shape) + 1) + sub_shape
    strides = img_pad.strides + img_pad.strides
    sub_matrices = np.lib.stride_tricks.as_strided(img_pad, view_shape, strides)
    convolved_img = np.einsum('ij,klij->kl', kernel, sub_matrices)
    return convolved_img


def get_gauss_filter(size, sigma):
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()


def return_non_maximum_suppression_matrix_r(R, neighbourhood, n_points = -1):
    # extract key points
    key_points_indeces = np.where(R > 0)

    list_key_points = [(key_points_indeces[1][i], key_points_indeces[0][i], R[key_points_indeces[0][i], key_points_indeces[1][i]]) for i in
                       np.arange(0, key_points_indeces[0].shape[0])]

    # filter local maximums
    R = local_non_maximum_suppression(R, list_key_points, neighbourhood, n_points)

    return R

def local_non_maximum_suppression(R, list_key_points, neighbourhood, n_points = -1):
    list_key_points.sort(reverse=True, key=lambda x: x[2])
    R_new = R.copy()
    for key_point in list_key_points:
        i = key_point[1]
        j = key_point[0]
        # check if this point is not filtered
        if R_new[i, j] == 0:
            continue

        # anullate all neighbours around local maximum
        # get patch around key point
        min_i = max(i - neighbourhood, 0)
        max_i = min(i + neighbourhood + 1, R.shape[0])
        min_j = max(j - neighbourhood, 0)
        max_j = min(j + neighbourhood + 1, R.shape[1])
        R_new[min_i:max_i, min_j:max_j] = 0
        R_new[i, j] = key_point[2]

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


def ransac_trial(fit_H, pt1, pt2, max_error):
    inds = np.zeros(pt1.shape[0], dtype=bool)
    for i, pt in enumerate(pt1):
        # transform top left corner of patch (coordinate 0,0)
        pt_transformed = np.matmul(pt, fit_H)
        dist = np.sqrt(np.sum((pt_transformed - pt2[i])**2))
        if dist < max_error:
            inds[i] = 1
    return inds


def get_ransac_params(num_matched_points):
    n_fit = 2
    n_trials = 5
    if num_matched_points > 5:
        n_fit = 4
        n_trials = 10
    return n_fit, n_trials


def ransac_filter(pt1, pt2, match, n_fit, n_trials):

    if match.shape[0] < 3:
        return match

    best_match = np.array([])
    best_n_inliers = 0
    for i in np.arange(0, n_trials):
        # take random indices
        inds = np.random.choice(pt1.shape[0], size = n_fit, replace=False)
        # fit model
        fit_H = compute_affine_matrix(pt1[inds, :], pt2[inds, :])
        # ransac trial
        inlier_inds = ransac_trial(fit_H, pt1, pt2, 30)
        if np.sum(inlier_inds) > best_n_inliers:
            best_match = match[inlier_inds]
            best_n_inliers = np.sum(inlier_inds)

            if best_n_inliers == pt1.shape[0]:
                return best_match

    return best_match


# we need to compute matrix H such as patchKP * H = templateKP
def compute_affine_matrix(pt1, pt2):
    # compute translation matrix
    H = np.zeros((3, 2))
    residuals = pt2 - pt1[:, 0:2]

    mean_residuals = np.mean(residuals, axis=0)
    H[0, 0] = 1
    H[2, 0] = mean_residuals[0]
    H[1, 1] = 1
    H[2, 1] = mean_residuals[1]

    return H


def first_and_second_smallest(numbers):
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
    image_copy = image.copy().astype('float64')
    image_copy /= image_copy.max()
    image_copy *= 255.0
    plt.imshow(image_copy, cmap='gray', vmin=0, vmax=255)
    plt.show()


def load_config():
    cfg_path = os.path.join(CONFIG_DIR, "patch_match_cfg.yml")
    cfg_file = open(cfg_path, "r")
    config = yaml.safe_load(cfg_file)
    return config