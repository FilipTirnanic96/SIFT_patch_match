# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 14:17:03 2022

@author: uic52421
"""
import sys
import time
from abc import ABC, abstractmethod
import numpy as np

from patch_matcher.patch_matcher_utility import *


class PatchMatcher(ABC):

    def __init__(self, verbose=0):
        # load config file
        self.config = load_config()
        # params
        self.__load_params_from_config(self.config)

        self.verbose = verbose
        self.time_passed_sec = -1
        self.time_passed_sec_extract_kp = -1
        self.time_passed_sec_extract_feat = -1
        self.time_passed_sec_match = -1
        self.time_passed_sec_loc = -1

        self.n_points_matched = 0
        self.match_dist = []

    def __load_params_from_config(self, config):
        # normalizing features
        self.use_scaling_norm = config['patch_matcher']['feature_normalization']['scaling_norm']
        # use gaussian normalization

        self.use_gauss_norm = config['patch_matcher']['feature_normalization']['use_gaussian_norm']
        self.use_gauss_global_norm = config['patch_matcher']['feature_normalization']['use_global_norm']
        if self.use_gauss_global_norm:
            self.global_feature_mean = 0
            self.global_feature_std = 1

        # match nearest neighbour threshold
        self.nn_threshold = config['patch_matcher']['match']['nn_threshold']

    def preprocess(self, image):
        image = np.array(image.convert('L'))
        return image

    @abstractmethod
    def extract_key_points(self, image):
        pass

    @abstractmethod
    def extract_features(self, key_points, image):
        pass

    # normalize features to unit vectors
    def normalize_features(self, features):
        if self.use_gauss_norm:
            # mean and std
            features -= np.mean(features, axis=1, keepdims=True)
            features /= (np.std(features, axis=1, keepdims=True) + sys.float_info.epsilon)

        if self.use_gauss_global_norm:
            # mean and std
            features = features - self.global_feature_mean
            features = features / self.global_feature_std

        if self.use_scaling_norm:
            # scaling
            lengths = np.sqrt(np.sum(features ** 2, 1))
            lengths[lengths == 0] = 1
            features = features / lengths[:, None]

        return features

    def match_features(self, patch_features, template_features):
        match = []
        self.match_dist = []
        nn_threshold = self.nn_threshold
        # match features
        for i in np.arange(0, patch_features.shape[0]):
            patch_feature = patch_features[i]

            # calculate dist from ith path_feature to each template_feature
            distance = np.sqrt(np.sum((template_features - patch_feature) ** 2, axis=1))
            m1, i1, m2, i2 = first_and_second_smallest(distance)
            # if we have just 1 feature we won't use threshold
            if patch_features.shape[0] != 1:
                distance = m1 / (m2 + sys.float_info.epsilon)
                if distance < nn_threshold:
                    match.append((i1, i))
                    self.match_dist.append(distance)
            else:
                match.append((i1, i))
                self.match_dist.append(m1 / m2)

        match = np.array(match)
        return match

    def match_features2visu(self, patch_features, template_features):
        match = []
        self.match_dist = []
        nn_threshold = self.nn_threshold
        # match features
        for i in np.arange(0, patch_features.shape[0]):
            patch_feature = patch_features[i]

            # calculate dist from ith path_feature to each template_feature
            distance = np.sqrt(np.sum((template_features - patch_feature) ** 2, axis=1))
            m1, i1, m2, i2 = first_and_second_smallest(distance)
            # if we have just 1 feature we won't use threshold
            if patch_features.shape[0] != 1:
                distance = m1 / (m2 + sys.float_info.epsilon)
                match.append((i1, i))
                match.append((i2, i))
                self.match_dist.append(distance)
            else:
                match.append((i1, i))
                self.match_dist.append(m1 / m2)

        match = np.array(match)
        return match

    @abstractmethod
    def find_corresponding_location_of_patch(self, patch_key_points, match):
        pass

    # returns left top location on template image of matched patch
    def match_patch(self, patch):
        # calculate time taken to match patch
        start = time.time()

        # preprocess image
        self.curr_image = np.array(patch) / 255
        patch = self.preprocess(patch)

        start_extract = time.time()
        # extract key points from template
        patch_key_points = self.extract_key_points(patch)
        self.time_passed_sec_extract_kp = round(1.0 * (time.time() - start_extract), 4)

        # check if we have detected some key points
        if patch_key_points.size == 0:
            self.n_points_matched = 0
            return 0, 0

        start_extract = time.time()
        # extract key points from template
        patch_key_points, patch_features = self.extract_features(patch_key_points, patch)
        self.time_passed_sec_extract_feat = round(1.0 * (time.time() - start_extract), 4)
        # check if we have detected some features
        if patch_features.size == 0:
            self.n_points_matched = 0
            return 0, 0

        # normalize features
        patch_features = self.normalize_features(patch_features)

        start_match = time.time()
        # find feature matchs between patch and template
        match = self.match_features(patch_features, self.template_features)
        self.time_passed_sec_match = round(1.0 * (time.time() - start_match), 4)
        # check if we have matched some features
        if match.size == 0:
            self.n_points_matched = 0
            return 0, 0

        start_loc = time.time()
        # find top left location on template of matched patch
        x_left_top, y_left_top, match = self.find_corresponding_location_of_patch(patch_key_points, match)
        self.time_passed_sec_loc = round(1.0 * (time.time() - start_loc), 4)
        # set num of matched points
        if match.size > 0:
            self.n_points_matched = match.shape[0]
        else:
            self.n_points_matched = 0

        end = time.time()
        self.time_passed_sec = round(1.0 * (end - start), 4)
        if self.verbose > 0:
            print("Time taken to match the patch", round(1.0 * (end - start), 4))

        return x_left_top, y_left_top

