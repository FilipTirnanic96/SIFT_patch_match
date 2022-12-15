import numpy as np


# filter array with median
def median_filter(pt1, pt2, match, th, min_pts=3):    # calculate residuals

    inds = np.ones(pt1.shape[0], dtype=bool)
    if pt1.shape[0] >= min_pts:
        pts = pt2 - pt1[:, 0:2]
        medians = np.median(pts, axis=0)
        dist_from_median = abs(pts - medians)
        sum_dist_from_median = np.sum(dist_from_median, axis = 1)
        ind = np.where(sum_dist_from_median > th)
        inds[ind] = 0
        match = match[inds, :]

    return match


def gauss_filter(H, pt1, pt2, match, min_pts=2):

    if pt1.shape[0] >= min_pts:
        # transform top left corner of patch (coordinate 0,0)
        pt1_transform = np.matmul(pt1, H)
        dist = np.sqrt(np.sum((pt2 - pt1_transform) ** 2, 1))
        dist_std = np.std(dist)
        error_threshold = 3 * dist_std
        match = match[dist <= error_threshold]

    return match


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