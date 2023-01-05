import numpy as np
import math
from patch_matcher.patch_matcher import PatchMatcher
from patch_matcher.patch_matcher_utility import first_and_second_smallest


class SimplePatchMatcher(PatchMatcher):

    def __init__(self, template_img, pw: int = 20, ph: int = 20, verbose: int = 0):
        # init parent constructor
        super().__init__(verbose)
        # preprocess image
        template_img = self.preprocess(template_img)
        # init params
        self.template = template_img
        self.pw = pw
        self.ph = ph

        # extract key points from template
        self.template_key_points = self.extract_key_points(self.template)
        # extract template features
        self.template_features = self.extract_features(self.template_key_points, self.template)
        # nomalize features
        self.template_features = self.normalize_features(self.template_features)

    def extract_key_points(self, image: np.array) -> np.array:
        """
        Extracts key points from image (override abstract method).
        Each pixel in image is key point.

        :param image: Input image
        :return: Key points
        """

        key_points_list = []
        # iterate through each pixel
        if image.shape[0] > 200:
            # iterate through each pixel
            for y in range(math.floor(self.ph / 2), image.shape[0] - math.floor(self.ph / 2) + 1):
                for x in range(math.floor(self.pw / 2), image.shape[1] - math.floor(self.pw / 2) + 1):
                    # ad position of pixel as key point
                    key_points_list.append((x, y))
        else:
            # take just top left corner of patch
            key_points_list.append((10, 10))

        key_points = np.array(key_points_list)

        return key_points

    def extract_features(self, key_points: np.array, image: np.array) -> np.array:
        """
        Extracts feature around key points from image (override abstract method).
        Features are raveled patch pixel values.

        :param key_points: Array of tuple representing key points
        :param image: Input image
        :return: Features for each key point
        """

        features = []
        # iterate over key points
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

        return features

    def match_features(self, patch_features: np.array, template_features: np.array) -> np.array:
        """
        Match features extracted from patch to template features (override abstract method).
        Finds the nearest feature from template features.

        :param patch_features: Patch features
        :param template_features: Template features
        :return: Array of tuple with indices pairs of matched features
        """

        match = []
        # match features
        for i in np.arange(0, patch_features.shape[0]):
            patch_feature = patch_features[i]

            # calculate dist from ith path_feature to each template_feature
            distance = np.sqrt(np.sum((template_features - patch_feature) ** 2, axis=1))
            m1, i1, m2, i2 = first_and_second_smallest(distance)
            # append matched indices
            match.append((i1, i))

        match = np.array(match)
        return match

    def find_corresponding_location_of_patch(self, patch_key_points: np.array, match: np.array):
        """
        Finds the location of top left corner of the patch in template image (override abstract method).
        Take just first matched template point as center of match.

        :param patch_key_points: Patch key points array
        :param match: Match array
        :return: Top left position in template image and match array
        """

        # get first match indices
        i_kp_template, i_kp_patch = match[0]

        # get template key point
        template_center_match = self.template_key_points[i_kp_template]

        # calculate patch top left location in template image
        x_left_top = template_center_match[0] - math.floor(self.pw / 2)
        y_left_top = template_center_match[1] - math.floor(self.ph / 2)
        return x_left_top, y_left_top, match
