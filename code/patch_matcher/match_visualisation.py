# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 12:43:31 2022

@author: uic52421
"""

import os

from scipy import ndimage

from patch_matcher.patch_matcher import SimplePatchMatcher, AdvancePatchMatcher
from utils.glob_def import DATA_DIR
from PIL import Image
import numpy as np
from patch_matcher.visualisation import show_matched_points, show_key_points
import pandas as pd


# visualize match of patch and template
def visualise_match(template, patch_matcher_type, path_to_patches, df):
    # original patch for visu
    org_template = np.array(template)

    if patch_matcher_type == 'simple':
        patch_matcher = SimplePatchMatcher(template)
    elif patch_matcher_type == 'advanced':
        patch_matcher = AdvancePatchMatcher(template)
    else:
        raise ValueError("Patch matcher type must be simple or advanced")

    root_patch_path = os.path.join(DATA_DIR, 'set')

    # get grad and theta img from template
    template_grad = patch_matcher.grad_mag
    template_theta = patch_matcher.grad_theta
    template_key_points = patch_matcher.template_key_points
    show_key_points(org_template, template_key_points)
    # for each patch visu match
    for i, path_to_patch in enumerate(path_to_patches):

        patch_path = os.path.join(root_patch_path, path_to_patch)

        # read patch
        patch = Image.open(patch_path)

        # original patch for visu
        org_patch = np.array(patch)

        # preprocess image
        patch_matcher.curr_image = np.array(patch) / 255
        patch = patch_matcher.preprocess(patch)

        #patch = ndimage.gaussian_filter(patch, sigma=0.6, truncate=2)
        #show_key_points(org_patch, np.array([]))
        # extract key points from patch
        patch_key_points = patch_matcher.extract_key_points(patch)

        # get grad and theta img from patch
        patch_grad = patch_matcher.grad_mag
        patch_theta = patch_matcher.grad_theta

        # get expected patch grad and theta img from template
        x_expected = df.iloc[i]['x_expected']
        y_expected = df.iloc[i]['y_expected']
        t_patch_grad = template_grad[y_expected: y_expected + patch.shape[0], x_expected: x_expected + patch.shape[1]]
        t_patch_theta = template_theta[y_expected: y_expected + patch.shape[0], x_expected: x_expected + patch.shape[1]]

        org_patch_ = org_template[y_expected: y_expected + patch.shape[0], x_expected: x_expected + patch.shape[1], :]
        show_key_points(org_patch_, np.array([]))
        # check if we have detected some key points
        if patch_key_points.size == 0:
            print("No key points 1")
            continue


        # extract features from patch key points
        patch_key_points, patch_features = patch_matcher.extract_features(patch_key_points, patch)
        show_key_points(org_patch, patch_key_points)
        # check if we have detected some features
        if patch_features.size == 0:
            print("No features")
            continue

        # nomalize features
        patch_features = patch_matcher.normalize_features(patch_features)

        # find feature matchs between patch and template
        match = patch_matcher.match_features(patch_features, patch_matcher.template_features)
        if match.size == 0:
            print("No match")
            continue
            # show matched points
        # show_matched_points(org_template, org_patch, patch_matcher.template_key_points, patch_key_points, match)

        # find top left location on template of matched patch
        x_left_top, y_left_top, match = patch_matcher.find_corresponding_location_of_patch(patch_key_points, match)
        if match.size == 0:
            print("Filtered")
            continue

        # show matched points
        show_matched_points(org_template, org_patch, patch_matcher.template_key_points, patch_key_points, match)

    return 0
