# -*- coding: utf-8 -*-

import os

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
            continue
        # show key points
        show_key_points(org_patch, patch_matcher.patch_key_points)

        # check if any match points exist
        if patch_matcher.match.size == 0:
            print("No matched poins found for patch", path_to_patch)
            continue
        # show matched points
        show_matched_points(org_template, org_patch, patch_matcher.template_key_points, patch_matcher.patch_key_points, patch_matcher.match)
