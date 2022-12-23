import numpy as np


def compute_gradient_histogram(num_bins, gradient_magnitudes, gradient_angles):
    angle_step = 2 * np.pi / num_bins
    angles = np.arange(0, 2 * np.pi + angle_step, angle_step)

    indices = np.digitize(gradient_angles.ravel(), bins=angles)
    indices -= 1
    gradient_magnitudes_ravel = gradient_magnitudes.ravel()
    histogram = np.zeros(num_bins)
    for i in range(0, indices.shape[0]):
        histogram[indices[i]] += gradient_magnitudes_ravel[i]

    return histogram


def weight_gradient_histogram(histogram, coefficient):
    if coefficient >= 1 or coefficient < 0:
        return histogram

    #  find max index
    max_index = np.argmax(histogram)
    mult = coefficient
    i = max_index + 1

    while i < len(histogram):
        histogram[i] *= mult
        mult *= coefficient
        i += 1

    i = max_index - 1
    mult = coefficient
    while i > 0:
        histogram[i] *= mult
        mult *= coefficient
        i -= 1

    return histogram


def compute_gradient_feature(gradient_magnitudes, gradient_angles):
    feature = np.zeros((gradient_magnitudes.size * 2))
    grad_x = gradient_magnitudes.ravel() * np.cos(gradient_angles.ravel())
    gray_y = gradient_magnitudes.ravel() * np.sin(gradient_angles.ravel())

    feature[0:gradient_magnitudes.size] = grad_x
    feature[gradient_magnitudes.size:gradient_magnitudes.size * 2] = gray_y
    return feature


def compute_gray_value_feature(patch):
    feature = patch.ravel()
    return feature


def compute_color_histogram_feature(patch):
    feature = patch.ravel()
    return feature
