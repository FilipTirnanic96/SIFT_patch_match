import numpy as np
import math
from patch_matcher.patch_matcher import PatchMatcher
from patch_matcher.patch_matcher_utility import first_and_second_smallest


class SimplePatchMatcher(PatchMatcher):

    def __init__(self, template_img, pw, ph, verbose=0):
        # init parent constructor
        super().__init__(verbose)
        # preprocess image
        template_img = self.preprocess(template_img)
        # init params
        self.template = template_img
        self.pw = pw
        self.ph = ph
        if pw != 0 and ph != 0:
            # extract key points from template
            self.template_key_points = self.extract_key_points(self.template)
            # extract template features
            self.template_key_points, self.template_features = self.extract_features(self.template_key_points,
                                                                                     self.template)
            # nomalize features
            self.template_features = self.normalize_features(self.template_features)
        else:
            self.template_key_points = []
            self.template_features = []

    # override abstract method
    # every point in image is key point
    # returns x,y key points location
    def extract_key_points(self, image):
        key_points_list = []
        for y in range(math.floor(self.ph / 2), image.shape[0] - math.floor(self.ph / 2) + 1):
            for x in range(math.floor(self.pw / 2), image.shape[1] - math.floor(self.pw / 2) + 1):
                key_points_list.append((x, y))
        key_points = np.array(key_points_list)
        return key_points

    # override abstract method
    # return features for each key point
    def extract_features(self, key_points, image):
        features = []
        for i in np.arange(0, key_points.shape[0]):
            # get key point location
            x, y = key_points[i]
            # get patch around key point
            patch_feature = image[y - math.floor(self.ph / 2): y + math.floor(self.ph / 2),
                            x - math.floor(self.pw / 2): x + math.floor(self.pw / 2)]
            # ravel patch
            feature = patch_feature.ravel()
            # add feature
            features.append(feature)

        features = np.array(features)
        return key_points, features

    # simple matcher just uses one feature for patch so it will be just one match
    # outputs list of meatched features
    def match_features(self, patch_features, template_features):
        match = []
        # match features
        for i in np.arange(0, patch_features.shape[0]):
            patch_feature = patch_features[i]

            # calculate dist from ith path_feature to each template_feature
            distance = np.sqrt(np.sum((template_features - patch_feature) ** 2, axis=1))
            m1, i1, m2, i2 = first_and_second_smallest(distance)

            match.append((i1, i))

        return match

    # override abstract method
    # output top let postion of template where the patch match
    def find_corresponding_location_of_patch(self, patch_key_points, match):
        i_kp_template, i_kp_patch = match[0]

        template_center_match = self.template_key_points[i_kp_template]

        x_left_top = template_center_match[0] - math.floor(self.pw / 2)
        y_left_top = template_center_match[1] - math.floor(self.ph / 2)
        return x_left_top, y_left_top, match