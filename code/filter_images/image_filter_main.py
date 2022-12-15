from filter_images.image_reader import ReadImage
from patch_matcher.patch_matcher import AdvancePatchMatcher
from utils.glob_def import DATA_DIR
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
import os
from scipy import signal, ndimage


def show_filter_images(patch_matcher, image, gt_image):
    patch = patch_matcher.preprocess(image)
    patch_matcher.extract_key_points(image)
    patch_matcher.extract_key_points(gt_image)
    gaus_image = ndimage.gaussian_filter(image, sigma=0.6, truncate=4)
    patch_matcher.extract_key_points(gaus_image)
    # show images
    plt.figure('image')
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)

    plt.figure('filt image')
    plt.imshow(gaus_image, cmap='gray', vmin=0, vmax=255)

    plt.figure('gt_image')
    plt.imshow(gt_image, cmap='gray', vmin=0, vmax=255)

    plt.show()

def feature_pipeline(patch_matcher, patch):
    # extract features from patch key points
    patch_key_points = patch_matcher.extract_key_points(patch)
    # extract features from patch key points
    patch_key_points, patch_features = patch_matcher.extract_features(patch_key_points, patch)
    # nomalize features
    #patch_features = patch_matcher.normalize_features(patch_features)

    return patch_key_points, patch_features



def fcn_debug_features(patch_matcher, patch, gt_patch):
    patch_key_points, patch_features = feature_pipeline(patch_matcher, patch)
    patch_key_points_gt, patch_features_gt = feature_pipeline(patch_matcher, gt_patch)

    # find feature matchs between patch and template
    match = patch_matcher.match_features(patch_features, patch_matcher.template_features)
    match_template_features = patch_matcher.template_features[match[:, 0]]
    return


if __name__ == "__main__":
    # load 1 pixel images noise
    image_reader = ReadImage(DATA_DIR, '8.txt', 10, [])

    template_image_path = os.path.join(DATA_DIR, "set", "map.png")
    template = Image.open(template_image_path)
    patch_matcher = AdvancePatchMatcher(template)

    flag_show_filt_image = False
    debug_features = True
    while image_reader.is_there_next_image():
        # read next image pair
        image, gt_image = image_reader.read_next_image_pair()

        if flag_show_filt_image:
            show_filter_images(patch_matcher, image, gt_image)

        if debug_features:
            fcn_debug_features(patch_matcher, image, gt_image)