import math

from patch_matcher.feature_extraction import compute_gradient_histogram, weight_gradient_histogram
from patch_matcher.patch_matcher import PatchMatcher
import numpy as np

from patch_matcher.patch_matcher_utility import compute_affine_matrix, ransac_filter, \
    conv2d, get_ransac_params, return_non_maximum_suppression_matrix_r, get_2d_gauss_kernel


class AdvancePatchMatcher(PatchMatcher):

    def __init__(self, template_img, verbose=0):
        super().__init__(verbose)

        # load params from config
        self.__load_params_from_config(self.config)
        # init params
        self.grad_mag = []
        self.grad_theta = []
        self.gauss_kernel = get_2d_gauss_kernel(5, 2)

        # ------- params ---------
        # preprocess image
        self.curr_image = np.array(template_img) / 255
        template_img = self.preprocess(template_img)
        self.template = template_img
        # extract key points from template
        self.template_key_points = self.extract_key_points(self.template)
        # extract template features
        self.template_features = self.extract_features(self.template_key_points,
                                                       self.template)
        # normalize features
        self.template_features = self.normalize_features(self.template_features)

    def __load_params_from_config(self, config):
        """
        Init parameters from config yaml file

        :param config: Input loaded yaml config file
        """
        advanced_pm_params = config['advanced_patch_matcher']

        # <IMAGE PROCESSING PARAMS>
        self.neighbourhood_patch_size = advanced_pm_params['neighbourhood_patch_size']

        # <KEY POINTS PARAMS>
        # corner key points params
        key_points_corner_params = advanced_pm_params['key_points_corner']
        self.n_max_points_corners_template = key_points_corner_params['n_max_points_template']
        self.n_max_points_corners_patch = key_points_corner_params['n_max_points_patch']
        self.threshold_corner_low = key_points_corner_params['threshold_low']
        self.nms_corner_neighbourhood = key_points_corner_params['nms_neighbourhood']

        # edge key points params
        key_points_corner_edge = advanced_pm_params['key_points_edge']
        self.n_max_points_edges_template = key_points_corner_edge['n_max_points_template']
        self.n_max_points_edges_patch = key_points_corner_edge['n_max_points_patch']
        self.threshold_edge_low = key_points_corner_edge['threshold_low']
        self.threshold_edge_high = key_points_corner_edge['threshold_high']
        self.nms_edge_neighbourhood = key_points_corner_edge['nms_neighbourhood']

        # <FEATURE PARAMS>
        # use all 3 channels for features
        feature_extraction_params = advanced_pm_params['feature_extraction']
        self.use_3_channels = feature_extraction_params['use_3_channels']

        # init in respect to params
        if self.use_3_channels:
            self.channels_grad_mag = []
            self.channels_grad_theta = []

        # gradient histogram
        self.num_angle_bins = feature_extraction_params['gradient_histogram']['num_angle_bins']
        self.weight_coefficient = feature_extraction_params['gradient_histogram']['weight_coefficient']

        # <MATCHING PARAMS>
        # filtering outlier matches
        match_params = advanced_pm_params['match']
        self.two_point_max_diff = match_params['two_point_max_diff']
        self.use_ransac_filter = match_params['filters']['use_ransac_filter']

    def extract_key_points(self, image):
        """
        Extracts key points from image (override abstract method).
        Use Harris detector approach to extract corner and edge points.

        :param image: Input image
        :return: Key points
        """

        if self.use_3_channels:
            # init params
            sum_channel_dx2 = 0
            sum_channel_dy2 = 0
            sum_channel_dxdy = 0
            self.channels_grad_mag.clear()
            self.channels_grad_theta.clear()

            # calculate gradients magnitude and orientation for each channel
            for channel in [0, 1, 2]:
                channel_img = self.curr_image[:, :, channel]
                # find image gradients
                channel_img_dx = conv2d(channel_img, np.reshape(np.array([-1, 0, 1]), (-1, 1)))
                channel_img_dy = conv2d(channel_img, np.reshape(np.array([-1, 0, 1]), (1, -1)))

                # calculate required matrix for each channel for response matrix R
                channel_img_dx2 = 1.0 * channel_img_dx ** 2
                sum_channel_dx2 += channel_img_dx2

                channel_img_dy2 = 1.0 * channel_img_dy ** 2
                sum_channel_dy2 += channel_img_dy2

                channel_img_dxdy = 1.0 * channel_img_dx * channel_img_dy
                sum_channel_dxdy += channel_img_dxdy

                # calculate channel gradient magnitude
                channel_grad_mag = channel_img_dx2 + channel_img_dy2
                # calculate gradient angle
                channel_grad_theta = np.arctan2(channel_img_dy, channel_img_dx)
                # map gradient from 0 - 2* pi
                channel_grad_theta = channel_grad_theta % (2 * np.pi)
                # add channel gradient magnitude and angle to list
                self.channels_grad_mag.append(channel_grad_mag)
                self.channels_grad_theta.append(channel_grad_theta)

            # average calculated required matrices for response matrix R
            img_dx2 = sum_channel_dx2 / 3.0
            img_dy2 = sum_channel_dy2 / 3.0
            img_dxy = sum_channel_dxdy / 3.0
        else:
            # find image gradients
            img_dx = conv2d(image, np.reshape(np.array([-1, 0, 1]), (-1, 1)))
            img_dy = conv2d(image, np.reshape(np.array([-1, 0, 1]), (1, -1)))

            # calculate gradient angle
            self.grad_theta = np.arctan2(img_dy, img_dx)
            # map gradient from 0 - 2* pi
            self.grad_theta = self.grad_theta % (2 * np.pi)

            # take abs of gradients
            img_dx = np.absolute(img_dx)
            img_dy = np.absolute(img_dy)

            # calculate values for response matrix R
            img_dx2 = 1.0 * img_dx ** 2
            img_dy2 = 1.0 * img_dy ** 2
            img_dxy = 1.0 * img_dx * img_dy

            self.grad_mag = img_dx2 + img_dy2

        # blur gradients
        img_dx2 = conv2d(img_dx2, self.gauss_kernel)
        img_dy2 = conv2d(img_dy2, self.gauss_kernel)
        img_dxy = conv2d(img_dxy, self.gauss_kernel)

        # calculate det and trace for finding R
        detA = (img_dx2 * img_dy2) - (img_dxy ** 2)
        traceA = (img_dx2 + img_dy2)

        # calculate response matrix R for Harris Corner detection
        k = 0.05
        R = detA - k * (traceA ** 2)

        # nullify invalid features for template image
        if image.size > 20000:
            R[:, 0:8] = 0
            R[R.shape[0] - 5:R.shape[0], :] = 0

        R = abs(R)
        step = int(np.floor(self.neighbourhood_patch_size / 2))
        # set edge pixels to zero
        R[0:step, :] = 0
        R[-step:, :] = 0
        R[:, 0:step] = 0
        R[:, -step:] = 0

        # set max keep points parameters depending on template or patch image
        if image.size > 20000:
            n_max_points_corners = self.n_max_points_corners_template
            n_max_points_edges = self.n_max_points_edges_template
        else:
            n_max_points_corners = self.n_max_points_corners_patch
            n_max_points_edges = self.n_max_points_edges_patch

        # extract corner points R
        R_corners = R.copy()
        R_corners[R_corners < self.threshold_corner_low] = 0

        # non maxima suppression
        R_corners = return_non_maximum_suppression_matrix_r(R_corners, self.nms_corner_neighbourhood,
                                                            n_max_points_corners)

        # make edge points R
        R_edge = R.copy()
        R_edge[R_edge > self.threshold_edge_high] = 0
        R_edge[R_edge < self.threshold_edge_low] = 0

        # non maxima suppression
        R_edge = return_non_maximum_suppression_matrix_r(R_edge, self.nms_edge_neighbourhood, n_max_points_edges)

        # extract key points (corners + edges)
        R_sum = R_edge + R_corners
        neighbourhood = 4
        R_sum = return_non_maximum_suppression_matrix_r(R_sum, neighbourhood)

        # get new key points
        key_points_indeces = np.where(R_sum > 0)
        key_points_list = [(key_points_indeces[1][i], key_points_indeces[0][i]) for i in
                           np.arange(0, key_points_indeces[0].shape[0])]

        key_points = np.array(key_points_list)

        return key_points

    def extract_features(self, key_points, image):
        """
        Extracts feature around key points from image (override abstract method).
        Features are gradient histograms calculated around key point.

        :param key_points: Array of tuple representing key points
        :param image: Input image
        :return: Features for each key point
        """

        # init properties
        features = []
        neighbourhood_patch_step = math.floor(self.neighbourhood_patch_size / 2)
        num_angles = self.num_angle_bins
        # loop over each key point
        for i in np.arange(0, key_points.shape[0]):
            # get key point location
            x, y = key_points[i]
            # get patch around key point
            min_y = y - neighbourhood_patch_step
            max_y = y + neighbourhood_patch_step + 1
            min_x = x - neighbourhood_patch_step
            max_x = x + neighbourhood_patch_step + 1

            if self.use_3_channels:
                feature = []
                for channel in [0, 1, 2]:
                    # get channel gradient magnitude and angle matrix
                    channel_grad_mag = self.channels_grad_mag[channel]
                    channel_grad_theta = self.channels_grad_theta[channel]
                    # extract neighbourhood patch around keypoint
                    channel_neighbourhood_patch_grad = channel_grad_mag[min_y: max_y, min_x: max_x]
                    channel_neighbourhood_patch_theta = channel_grad_theta[min_y: max_y, min_x: max_x]
                    # compute gradient histogram
                    channel_feature = compute_gradient_histogram(num_angles, channel_neighbourhood_patch_grad,
                                                                 channel_neighbourhood_patch_theta)
                    # sum gradient histograms for each channel
                    if len(feature) == 0:
                        feature = channel_feature
                    else:
                        feature += channel_feature

            else:
                # extract neighbourhood patch around keypoint
                neighbourhood_patch_grad = self.grad_mag[min_y: max_y, min_x: max_x]
                neighbourhood_patch_theta = self.grad_theta[min_y: max_y, min_x: max_x]
                # compute gradient histogram
                feature = compute_gradient_histogram(num_angles, neighbourhood_patch_grad, neighbourhood_patch_theta)
            # weight gradient histogram
            feature = weight_gradient_histogram(feature, self.weight_coefficient)
            # add new feature
            features.append(feature)

        features = np.array(features)

        return features

    def find_corresponding_location_of_patch(self, patch_key_points, match):
        """
        Finds the location of top left corner of the patch in template image (override abstract method).
        Take optimum affine transformation matrix. Uses RANSAC to filter outliers.

        :param patch_key_points: Patch key points array
        :param match: Match array
        :return: Top left position in template image and match array
        """
        # find index of best representative feature
        best_match = np.array([match[np.argmin(self.match_dist), :]])

        # extract matched key points from patch
        pt1 = np.ones((len(match), 3))
        pt1[:, 0:2] = patch_key_points[match[:, 1], :]
        # extract matched key points from template
        pt2 = self.template_key_points[match[:, 0], :]

        # use ransac filter
        if self.use_ransac_filter:
            n_fit, n_trials = get_ransac_params(match.shape[0])
            match = ransac_filter(pt1, pt2, match, n_fit, n_trials)

        if match.shape[0] == 0:
            match = best_match

        # check if 2 point are too different
        if match.shape[0] == 2:
            pt_diff = np.abs(pt2[1, :] - pt2[0, :])
            if pt_diff[0] > self.two_point_max_diff or pt_diff[1] > self.two_point_max_diff:
                # set match to best feature match
                match = best_match

        # recompute affine matrix
        # extract matched key points from patch
        pt1 = np.ones((len(match), 3))
        pt1[:, 0:2] = patch_key_points[match[:, 1], :]
        # extract matched key points from template
        pt2 = self.template_key_points[match[:, 0], :]

        # compute affine matrix
        H = compute_affine_matrix(pt1, pt2)

        # transform top left corner of patch (coordinate 0,0)
        result = np.matmul(np.array([0, 0, 1], ndmin=2), H)
        x_top_left = int(np.round(result[0, 0]))
        y_top_left = int(np.round(result[0, 1]))

        return x_top_left, y_top_left, match
