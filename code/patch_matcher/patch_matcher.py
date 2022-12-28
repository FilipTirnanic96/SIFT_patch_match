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
        # init patch match properties
        self.curr_image = np.array([])
        self.template_features = np.array([])
        self.verbose = verbose
        self.time_passed_sec = -1

        self.n_points_matched = 0
        self.match_dist = []

    def __load_params_from_config(self, config):
        """
        Init parameters from config yaml file

        :param config: Input loaded yaml config file
        """

        # normalizing features
        self.use_scaling_norm = config['patch_matcher']['feature_normalization']['scaling_norm']
        # use gaussian normalization
        self.use_gauss_norm = config['patch_matcher']['feature_normalization']['use_gaussian_norm']
        # match nearest neighbour threshold
        self.nn_threshold = config['patch_matcher']['match']['nn_threshold']

    def preprocess(self, image):
        """
        Preprocess image

        :param image: Input image
        :return: Preprocessed image
        """

        image = np.array(image.convert('L'))
        return image

    @abstractmethod
    def extract_key_points(self, image: np.array) -> np.array:
        """
        Extracts key points from image

        :param image: Input image
        :return: Key points
        """

        pass

    @abstractmethod
    def extract_features(self, key_points: np.array, image: np.array) -> np.array:
        """
        Extracts feature around key points from image

        :param key_points: Array of tuple representing key points
        :param image: Input image
        :return: Features for each key point
        """

        pass

    def normalize_features(self, features: np.array) -> np.array:
        """
        Normalize input features

        :param features: Input features
        :return: Normalized features
        """

        # use gaussian normalization
        if self.use_gauss_norm:
            features -= np.mean(features, axis=1, keepdims=True)
            features /= (np.std(features, axis=1, keepdims=True) + sys.float_info.epsilon)
        # using unit vector scaling
        if self.use_scaling_norm:
            lengths = np.sqrt(np.sum(features ** 2, 1))
            lengths[lengths == 0] = 1
            features = features / lengths[:, None]

        return features

    def match_features(self, patch_features: np.array, template_features: np.array) -> np.array:
        """
        Match features extracted from patch to template features

        :param patch_features: Patch features
        :param template_features: Template features
        :return: Array of tuple with indices pairs of matched features
        """

        # init match array
        match = []
        # init match distance
        self.match_dist = []
        # init nearest neighbour threshold
        nn_threshold = self.nn_threshold

        # match features
        for i in np.arange(0, patch_features.shape[0]):
            patch_feature = patch_features[i]
            # calculate dist from ith patch_feature to each template_feature
            distance = np.sqrt(np.sum((template_features - patch_feature) ** 2, axis=1))
            # get first and second nearest template feature
            m1, i1, m2, i2 = first_and_second_smallest(distance)
            # if we have just 1 feature we won't use threshold
            if patch_features.shape[0] != 1:
                distance = m1 / (m2 + sys.float_info.epsilon)
                # if distance between first and second nearest feature is less then
                # threshold, add new pain im match array
                if distance < nn_threshold:
                    match.append((i1, i))
                    self.match_dist.append(distance)
            else:
                match.append((i1, i))
                self.match_dist.append(m1 / m2)

        match = np.array(match)

        return match

    @abstractmethod
    def find_corresponding_location_of_patch(self, patch_key_points: np.array, match: np.array):
        """
        Finds the location of top left corner of the patch in template image

        :param patch_key_points: Patch key points array
        :param match: Match array
        :return: Top left position in template image and match array
        """

        pass

    def match_patch(self, patch):
        """
        Finds the location of top left corner of the patch in template image.
        Match the patch location in template image.

        :param patch: Patch of template image to be matched
        :return: Top left position in template image
        """

        # calculate time taken to match patch
        start = time.time()

        # set current processing image
        self.curr_image = np.array(patch) / 255

        # preprocess image
        patch = self.preprocess(patch)

        # extract key points from patch
        patch_key_points = self.extract_key_points(patch)

        # check if we have detected some key points
        if patch_key_points.size == 0:
            self.n_points_matched = 0
            return 0, 0

        # extract features from patch
        patch_features = self.extract_features(patch_key_points, patch)

        # check if we have detected some features
        if patch_features.size == 0:
            self.n_points_matched = 0
            return 0, 0

        # normalize features
        patch_features = self.normalize_features(patch_features)

        # find feature match between patch and template
        match = self.match_features(patch_features, self.template_features)

        # check if we have matched some features
        if match.size == 0:
            self.n_points_matched = 0
            return 0, 0

        # find top left location in template for matched patch
        x_left_top, y_left_top, match = self.find_corresponding_location_of_patch(patch_key_points, match)

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

