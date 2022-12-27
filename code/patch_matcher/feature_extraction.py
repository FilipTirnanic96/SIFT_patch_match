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
    multiply = coefficient
    i = max_index + 1

    while i < len(histogram):
        histogram[i] *= multiply
        multiply *= coefficient
        i += 1

    i = max_index - 1
    multiply = coefficient
    while i > 0:
        histogram[i] *= multiply
        multiply *= coefficient
        i -= 1

    return histogram
