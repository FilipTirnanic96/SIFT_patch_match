# -*- coding: utf-8 -*-

import os

from matplotlib import pyplot as plt

from patch_matcher.patch_matcher import PatchMatcher
from utils.glob_def import DATA_DIR
from PIL import Image
import numpy as np
from pach_match_visualisation.visualisation import show_matched_points, show_key_points
import pandas as pd


def visualise_match(template: np.array, patch_matcher: PatchMatcher, df: pd.DataFrame):
    """
    Visualise matched point between template and patch.

    :param template: Template image
    :param patch_matcher: Patch matcher
    :param df: Dataframe got from KPI report
    """

    # original patch for visualisation
    org_template = np.array(template)

    # init root folder for patches
    root_patch_path = os.path.join(DATA_DIR, 'set')

    # get template key points
    template_key_points = patch_matcher.template_key_points
    # show template key points
    show_key_points(org_template, template_key_points)

    # for each patch visualise match with template
    for i, path_to_patch in enumerate(df['path']):

        patch_path = os.path.join(root_patch_path, path_to_patch)

        # read patch
        patch = Image.open(patch_path)

        # original patch for visualisation
        org_patch = np.array(patch)
        # match patch
        patch_matcher.match_patch(patch)

        # check if any key points detected
        if patch_matcher.patch_key_points.size == 0:
            print("No key points detected for patch", path_to_patch)
            # show key points
            show_key_points(org_patch, patch_matcher.patch_key_points)
            continue
        # show key points
        show_key_points(org_patch, patch_matcher.patch_key_points)

        # check if any match points exist
        if patch_matcher.match.size == 0:
            print("No matched poins found for patch", path_to_patch)
            continue
        # show matched points
        show_matched_points(org_template, org_patch, patch_matcher.template_key_points, patch_matcher.patch_key_points, patch_matcher.match)


def visualise_match2(template: np.array, patch_matcher: PatchMatcher, df: pd.DataFrame):
    """
    Visualise matched point between template and patch.

    :param template: Template image
    :param patch_matcher: Patch matcher
    :param df: Dataframe got from KPI report
    """

    # original patch for visualisation
    org_template = np.array(template)

    # init root folder for patches
    root_patch_path = os.path.join(DATA_DIR, 'set')

    # init merged image
    max_patch_size = 40

    patch_step_bot = 80
    init_offset_bot = 70
    num_patch_bot = np.floor((org_template.shape[1] - init_offset_bot) / (max_patch_size + patch_step_bot))

    patch_step_right = 60
    init_offset_right = 40
    max_num_patch = num_patch_bot + np.floor((org_template.shape[0] - init_offset_right) / (max_patch_size + patch_step_right))
    blank_space = 10

    # init merge image
    merged_image = 255 * np.ones((org_template.shape[0] + max_patch_size + blank_space, org_template.shape[1] + max_patch_size + blank_space, 3), dtype=int)
    # merge images
    merged_image[0:org_template.shape[0], 0:org_template.shape[1], :] = org_template
    offset_x = 0
    offset_y = 0
    # init figure
    plt.figure()
    plt.axis('off')

    # get random n patches
    df = df.sample(n = int(max_num_patch))

    # for each patch visualise match with template
    for i, path_to_patch in enumerate(df['path']):
        # add patch to bot
        if i < num_patch_bot:
            # set offset of patch
            offset_x = init_offset_bot + int(i * (max_patch_size + patch_step_bot))
            offset_y = int(org_template.shape[0] + blank_space)

        # add patch to right
        if i >= num_patch_bot:
            # set offset of patch
            offset_x = int(org_template.shape[1] + blank_space)
            offset_y = init_offset_right + int((i-num_patch_bot) * (max_patch_size + patch_step_right))

        # if there is no more space to plot patch break
        if i >= max_num_patch:
            break

        patch_path = os.path.join(root_patch_path, path_to_patch)

        # read patch
        patch = Image.open(patch_path)

        # original patch for visualisation
        org_patch = np.array(patch)
        # match patch
        patch_matcher.match_patch(patch)

        # check if any key points detected
        if patch_matcher.patch_key_points.size == 0:
            print("No key points detected for patch", path_to_patch)
            continue
        # show key points
        #show_key_points(org_patch, patch_matcher.patch_key_points)

        # check if any match points exist
        if patch_matcher.match.size == 0:
            print("No matched poins found for patch", path_to_patch)
            continue
        # show matched points
        merged_image = show_matched_points(org_template, org_patch, patch_matcher.template_key_points, patch_matcher.patch_key_points, patch_matcher.match, merged_image, offset_x, offset_y)

    plt.imshow(merged_image, vmin=0, vmax=255)
    plt.show()

    save_csv = input()
    if save_csv == 'y':
        print("saved")
        df.to_csv("./sample_patch_images3.csv")