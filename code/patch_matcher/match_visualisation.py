# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 12:43:31 2022

@author: uic52421
"""

import os
from patch_matcher.patch_matcher import SimplePatchMatcher, AdvancePatchMatcher
from utils.glob_def import DATA_DIR
from PIL import Image
import numpy as np
from patch_matcher.visualisation import show_matched_points
    
# visualize match of patch and template
def visualise_match(template, patch_matcher_type, path_to_patches):
    # original patch for visu
    org_template = np.array(template)
    
    if patch_matcher_type == 'simple':
        patch_matcher = SimplePatchMatcher(template)
    elif patch_matcher_type == 'advanced':
        patch_matcher = AdvancePatchMatcher(template)
    else:
        raise ValueError("Patch matcher type must be simple or advanced")
    
    root_patch_path = os.path.join(DATA_DIR, 'set')
    # for each patch visu match
    for path_to_patch in path_to_patches:
        
        patch_path = os.path.join(root_patch_path, path_to_patch)
        
        # read patch
        patch = Image.open(patch_path)
        
        # original patch for visu
        org_patch = np.array(patch)
        
        # preprocess image
        patch = patch_matcher.preprocess(patch)
        
        # extract key points from patch
        patch_key_points = patch_matcher.extract_key_points(patch)
        # check if we have detected some key points
        if(patch_key_points.size == 0):
            continue
        
        # extract features from patch key points
        patch_key_points, patch_features = patch_matcher.extract_features(patch_key_points, patch)
        # check if we have detected some features
        if(patch_features.size == 0):
            continue
        
        # nomalize features
        patch_features = patch_matcher.nomalize_features(patch_features)       
        # find feature matchs between patch and template
        match = patch_matcher.match_features(patch_features, patch_matcher.template_features)
        if(match.size == 0):
            continue 
        
        # find top left location on template of matched patch
        x_left_top, y_left_top, match = patch_matcher.find_correspodind_location_of_patch(patch_key_points, match)
        if(match.size == 0):
            continue
        
        # show matched points
        show_matched_points(org_template, org_patch, patch_matcher.template_key_points, patch_key_points, match)

    return 0