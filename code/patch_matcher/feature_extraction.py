import numpy as np


def compute_gradient_histogram(num_bins: int, gradient_magnitudes: np.array, gradient_angles: np.array) -> np.array:
    """
    Computes gradient histogram from gradient magnitudes and angles

    :param num_bins: Number of bins in gradient histogram
    :param gradient_magnitudes: Gradient magnitudes
    :param gradient_angles: Gradient angles
    :return:Gradient histogram
    """

    # init bin angle array
    angle_step = 2 * np.pi / num_bins
    angles = np.arange(0, 2 * np.pi + angle_step, angle_step)
    # digitize gradient angles
    indices = np.digitize(gradient_angles.ravel(), bins=angles)
    indices -= 1
    # ravel gradient magnitudes
    gradient_magnitudes_ravel = gradient_magnitudes.ravel()
    # init histogram
    histogram = np.zeros(num_bins)
    # fill gradient histogram
    for i in range(0, indices.shape[0]):
        histogram[indices[i]] += gradient_magnitudes_ravel[i]

    return histogram


def weight_gradient_histogram(histogram: np.array, coefficient: float) -> np.array:
    """
    Weight gradient histogram bins according to distance from maximum value

    :param histogram: Gradient histogram
    :param coefficient: Weighting coefficient
    :return:Weighted gradient histogram
    """

    if coefficient >= 1 or coefficient < 0:
        return histogram

    #  find max index
    max_index = np.argmax(histogram)

    # weight right side
    multiply = coefficient
    i = max_index + 1
    while i < len(histogram):
        histogram[i] *= multiply
        multiply *= coefficient
        i += 1
    # weight left side
    i = max_index - 1
    multiply = coefficient
    while i > -1:
        histogram[i] *= multiply
        multiply *= coefficient
        i -= 1

    return histogram
