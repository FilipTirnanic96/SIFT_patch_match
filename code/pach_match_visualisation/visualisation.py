# -*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt
import numpy as np


def show_key_points(image: np.array, key_points: np.array = np.array([])):
    """
    Plots key points on inputs image.

    :param image: Input image to plot key points on
    :param key_points: Detected key points on input image
    """

    # take image copy
    image_copy = image.copy()

    # image should be colored (3 dimensions)
    if len(image.shape) == 2:
        print("Image should be colored (3d image)")
        return

    # plot key points
    plt.figure()
    plt.axis('off')
    # init thickness and radius of key points
    thickness = -1
    radius = 1

    # setup colors
    color = [230, 20, 20]

    # loop through key points
    for i in np.arange(0, key_points.shape[0]):
        # get key point location
        x, y = key_points[i]
        # plot circle at key point location
        image_copy = cv2.circle(image_copy, (x, y), radius, color, thickness)

    # show image with plotted key points
    plt.imshow(image_copy, vmin=0, vmax=255)

    plt.show()


def show_matched_points(template: np.array, patch: np.array, template_key_points: np.array, patch_key_points: np.array,
                        match: np.array, merged_image: np.array = None, offset_x: int = -1, offset_y = -1):
    """
    Plots lines connecting matched points from template and patch. If merged image is None one patch will be plotted


    :param template: Template image
    :param patch: Patch image
    :param template_key_points: Detected template key points
    :param patch_key_points: Detected patch key points
    :param match: Match between template and patch key points
    :param merged_image: Merged image to add patch match
    :param offset_x: Offset x where to add patch
    :param offset_y: Offset y where to add patch
    """

    # image should be colored (3 dimensions)
    if len(template.shape) == 2:
        print("Image should be colored (3d image)")
        return

    flag_no_merged_img = False
    if merged_image is None:
        flag_no_merged_img = True

    # extract matched key points from patch
    pt1 = patch_key_points[match[:, 1], :]
    # extract matched key points from template
    pt2 = template_key_points[match[:, 0], :]

    # if we don't pass merged image init merged image
    if flag_no_merged_img:
        # blank space between template img and patch img
        blank_space = 10
        # offset of patch
        offset_x = template.shape[1] + blank_space
        offset_y = int(np.round(template.shape[0] / 2))

        # init merge image
        merged_image = 255 * np.ones((template.shape[0], template.shape[1] + patch.shape[1] + blank_space, 3), dtype=int)
        # merge images
        merged_image[0:template.shape[0], 0:template.shape[1], :] = template

    merged_image[offset_y:offset_y + patch.shape[0], offset_x:offset_x + patch.shape[1], :] = patch

    # add offset to patch coordinates
    pt1[:, 0] += offset_x
    pt1[:, 1] += offset_y

    # if merge image is None plot result
    if flag_no_merged_img:
        # init figure
        plt.figure()
        plt.axis('off')

    # draw lines and key points
    thickness = -1
    thickness_line = 1
    radius = 2

    # set key points color
    color = [230, 20, 20]

    # loop through key points
    for i in np.arange(0, pt1.shape[0]):
        # get key points locations
        xt, yt = pt2[i]
        xp, yp = pt1[i]

        # plot matched key points
        merged_image = cv2.circle(merged_image, (xt, yt), radius, color, thickness)
        merged_image = cv2.circle(merged_image, (xp, yp), radius, color, thickness)
        merged_image = cv2.line(merged_image, (xt, yt), (xp, yp), 0, thickness_line)

    if flag_no_merged_img:
        # show plotted matched points
        plt.imshow(merged_image, vmin=0, vmax=255)

        plt.show()

    return merged_image
