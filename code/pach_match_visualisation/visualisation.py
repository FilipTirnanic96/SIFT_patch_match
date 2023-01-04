# -*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt
import numpy as np


def show_key_points(image: np.array, key_points: np.array):
    """
    Plots key points on inputs image.

    :param image: Input image to plot key points on
    :param key_points: Detected key points on input image
    """

    # take image copy
    image_copy = image.copy()

    # flag if pic are 2d or 3d
    picture2d = (len(image_copy.shape) == 2)

    # plot key points
    plt.figure()
    plt.axis('off')
    # init thickness and radius of key points
    thickness = -1
    radius = 1

    # setup colors
    if picture2d:
        # black if picture is gray
        color = 0
    else:
        # red if picture is RGB color
        color = [230, 20, 20]

    # loop through key points
    for i in np.arange(0, key_points.shape[0]):
        # get key point location
        x, y = key_points[i]
        # plot circle at key point location
        image_copy = cv2.circle(image_copy, (x, y), radius, color, thickness)

    # show image with plotted key points
    if picture2d:
        plt.imshow(image_copy, cmap='gray', vmin=0, vmax=255)
    else:
        plt.imshow(image_copy, vmin=0, vmax=255)

    plt.show()


def show_matched_points(template: np.array, patch: np.array, template_key_points: np.array, patch_key_points: np.array,
                        match: np.array):
    """
    Plots lines connecting matched points from template and patch.

    :param template: Template image
    :param patch: Patch image
    :param template_key_points: Detected template key points
    :param patch_key_points: Detected patch key points
    :param match: Match between template and patch key points
    """

    # flag if image are 2d or 3d
    picture2d = (len(template.shape) == 2)

    # extract matched key points from patch
    pt1 = patch_key_points[match[:, 1], :]
    # extract matched key points from template
    pt2 = template_key_points[match[:, 0], :]

    # blank space between template img and patch img
    blank_space = 10
    # offset of patch
    offset_x = template.shape[1] + blank_space
    offset_y = int(np.round(template.shape[0] / 2))

    if picture2d:
        # init merge image
        merged_img = 255 * np.ones((template.shape[0], template.shape[1] + patch.shape[1] + blank_space), dtype=int)

        # merge images
        merged_img[0:template.shape[0], 0:template.shape[1]] = template
        merged_img[offset_y:offset_y + patch.shape[0], offset_x:offset_x + patch.shape[1]] = patch
    else:
        # init merge image
        merged_img = 255 * np.ones((template.shape[0], template.shape[1] + patch.shape[1] + blank_space, 3), dtype=int)

        # merge images
        merged_img[0:template.shape[0], 0:template.shape[1], :] = template
        merged_img[offset_y:offset_y + patch.shape[0], offset_x:offset_x + patch.shape[1], :] = patch

    # add offset to patch coordinates
    pt1[:, 0] += offset_x
    pt1[:, 1] += offset_y

    # init figure
    plt.figure()
    plt.axis('off')

    # draw lines and key points
    thickness = -1
    thickness_line = 1
    radius = 2

    # set key points color
    if picture2d:
        color = 0
    else:
        color = [230, 20, 20]

    # loop through key points
    for i in np.arange(0, pt1.shape[0]):
        # get key points locations
        xt, yt = pt2[i]
        xp, yp = pt1[i]

        # plot matched key points
        merged_img = cv2.circle(merged_img, (xt, yt), radius, color, thickness)
        merged_img = cv2.circle(merged_img, (xp, yp), radius, color, thickness)
        merged_img = cv2.line(merged_img, (xt, yt), (xp, yp), 0, thickness_line)

    # show plotted matched points
    if picture2d:
        plt.imshow(merged_img, cmap='gray', vmin=0, vmax=255)
    else:
        plt.imshow(merged_img, vmin=0, vmax=255)

    plt.show()
