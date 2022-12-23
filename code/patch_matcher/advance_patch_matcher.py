from scipy import ndimage
import math

from patch_matcher.feature_extraction import compute_gradient_histogram, weight_gradient_histogram
from patch_matcher.patch_matcher import PatchMatcher
import numpy as np

from patch_matcher.patch_matcher_utility import compute_affine_matrix, ransac_filter, \
    conv2d, get_ransac_params, return_non_maximum_suppression_matrix_r, get_gauss_filter


class AdvancePatchMatcher(PatchMatcher):

    def __init__(self, template_img, verbose=0):
        super().__init__(verbose)

        # load params from config
        self.__load_params_from_config(self.config)
        # init params
        self.grad_mag = []
        self.grad_theta = []
        self.gauss_kernel = get_gauss_filter(5, 2)
        # debug
        self.expected_x = -1
        self.expected_y = -1

        # ------- params ---------
        # preprocess image
        self.curr_image = np.array(template_img) / 255
        template_img = self.preprocess(template_img)
        self.template = template_img
        # extract key points from template
        self.template_key_points = self.extract_key_points(self.template)
        # extract template features
        self.template_key_points, self.template_features = self.extract_features(self.template_key_points,
                                                                                 self.template)

        if self.use_gauss_global_norm:
            self.global_feature_mean = np.mean(self.template_features, axis=0, keepdims=True)
            self.global_feature_std = np.std(self.template_features, axis=0, keepdims=True)

        # nomalize features
        self.template_features = self.normalize_features(self.template_features)

    def __load_params_from_config(self, config):

        advanced_pm_params = config['advanced_patch_matcher']

        self.patch_size = advanced_pm_params['patch_size']
        # <IMAGE PROCESSING PARAMS>
        self.n_max_points_corners_template = advanced_pm_params['n_max_points_corners_template']
        self.n_max_points_edges_template = advanced_pm_params['n_max_points_edges_template']
        self.n_max_points_corners_patch = advanced_pm_params['n_max_points_corners_patch']
        self.n_max_points_edges_patch = advanced_pm_params['n_max_points_edges_patch']
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

    # override abstract method
    # use Harris detector to extract corner points
    # returns x,y key points location
    def extract_key_points(self, image):

        avg_channel_dx2 = 0
        avg_channel_dy2 = 0
        avg_channel_dxdy = 0
        if self.use_3_channels:
            self.channels_grad_mag.clear()
            self.channels_grad_theta.clear()

            # calculate gradients magnitude and orientation for each channel
            for channel in [0, 1, 2]:
                channel_img = self.curr_image[:, :, channel]
                # find image gradients
                # % gradient image, for gradients in x direction.
                channel_img_dx = conv2d(channel_img, np.reshape(np.array([-1, 0, 1]), (-1, 1)))
                channel_img_dy = conv2d(channel_img, np.reshape(np.array([-1, 0, 1]), (1, -1)))

                # calculate gradient angle
                channel_grad_theta = np.arctan2(channel_img_dy, channel_img_dx)
                # map gradient from 0 - 2* pi
                channel_grad_theta = channel_grad_theta % (2 * np.pi)

                channel_img_dx2 = 1.0 * channel_img_dx ** 2
                avg_channel_dx2 += channel_img_dx2

                channel_img_dy2 = 1.0 * channel_img_dy ** 2
                avg_channel_dy2 += channel_img_dy2

                channel_img_dxdy = 1.0 * channel_img_dx * channel_img_dy
                avg_channel_dxdy += channel_img_dxdy

                channel_grad_mag = channel_img_dx2 + channel_img_dy2

                self.channels_grad_mag.append(channel_grad_mag)
                self.channels_grad_theta.append(channel_grad_theta)

            avg_channel_dx2 /= 3.0
            avg_channel_dy2 /= 3.0
            avg_channel_dxdy /= 3.0

        # find image gradients
        if self.use_3_channels:
            img_dx2 = avg_channel_dx2
            img_dy2 = avg_channel_dy2
            img_dxy = avg_channel_dxdy
        else:
            # % gradient image, for gradients in x direction.
            img_dx = conv2d(image, np.reshape(np.array([-1, 0, 1]), (-1, 1)))
            # % gradients in y direction.
            img_dy = conv2d(image, np.reshape(np.array([-1, 0, 1]), (1, -1)))

            # anulate invalid features
            if image.size > 20000:
                img_dx[:, 0:8] = 0
                img_dy[img_dy.shape[0] - 3:img_dy.shape[0], :] = 0

            # calculate gradient angle
            self.grad_theta = np.arctan2(img_dy, img_dx)
            # map gradient from 0 - 2* pi
            self.grad_theta = self.grad_theta % (2 * np.pi)

            # take abs of gradients
            img_dx = np.absolute(img_dx)
            img_dy = np.absolute(img_dy)

            # calculate values for M matrix
            img_dx2 = 1.0 * img_dx ** 2
            img_dy2 = 1.0 * img_dy ** 2
            img_dxy = 1.0 * img_dx * img_dy

            self.grad_mag = img_dx2 + img_dy2

        # blur gradients
        #img_dx2 = ndimage.gaussian_filter(img_dx2, sigma=2, truncate=1)
        #img_dy2 = ndimage.gaussian_filter(img_dy2, sigma=2, truncate=1)
        #img_dxy = ndimage.gaussian_filter(img_dxy, sigma=2, truncate=1)

        img_dx2 = conv2d(img_dx2, self.gauss_kernel)
        img_dy2 = conv2d(img_dy2, self.gauss_kernel)
        img_dxy = conv2d(img_dxy, self.gauss_kernel)
        # calculate det and trace for finding R
        detA = (img_dx2 * img_dy2) - (img_dxy ** 2)
        traceA = (img_dx2 + img_dy2)

        # calculate response for Harris Corner equation
        k = 0.05
        R = detA - k * (traceA ** 2)

        # anulate invalid features
        if image.size > 20000:
            R[:, 0:8] = 0
            R[R.shape[0] - 5:R.shape[0], :] = 0

        R = abs(R)
        step = int(np.floor(self.patch_size/2))
        # set edge pixels to zero
        R[0:step, :] = 0
        R[-step:, :] = 0
        R[:, 0:step] = 0
        R[:, -step:] = 0

        if image.size > 20000:
            n_max_points_corners = self.n_max_points_corners_template
            n_max_points_edges = self.n_max_points_edges_template
        else:
            n_max_points_corners = self.n_max_points_corners_patch
            n_max_points_edges = self.n_max_points_edges_patch

        # make corner points R
        R_corners = R.copy()
        th_corners = 1e-5
        R_corners[R_corners < th_corners] = 0

        neighbourhood = 4
        R_corners = return_non_maximum_suppression_matrix_r(R_corners, neighbourhood, n_max_points_corners)

        # make edge points R
        R_edge = R.copy()
        th_edge_high = 7e-6
        th_edge_low = 5e-6
        R_edge[R_edge > th_edge_high] = 0
        R_edge[R_edge < th_edge_low] = 0

        neighbourhood = 8
        R_edge = return_non_maximum_suppression_matrix_r(R_edge, neighbourhood, n_max_points_edges)

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
        features = []
        patch_step = math.floor(self.patch_size / 2)
        num_angles = self.num_angle_bins
        key_points_flag = np.ones((key_points.shape[0]), dtype=bool)
        for i in np.arange(0, key_points.shape[0]):
            # get key point location
            x, y = key_points[i]
            # get patch around key point
            min_y = y - patch_step
            max_y = y + patch_step + 1
            min_x = x - patch_step
            max_x = x + patch_step + 1

            if min_y < 0:
                min_y = 0
            if max_y > image.shape[0]:
                max_y = image.shape[0]
            if min_x < 0:
                min_x = 0
            if max_x > image.shape[1]:
                max_x = image.shape[1]

            if self.use_3_channels:
                feature = []
                for channel in [0, 1, 2]:
                    channel_grad_mag = self.channels_grad_mag[channel]
                    channel_grad_theta = self.channels_grad_theta[channel]

                    channel_patch_grad = channel_grad_mag[min_y: max_y, min_x: max_x]
                    channel_patch_theta = channel_grad_theta[min_y: max_y, min_x: max_x]
                    channel_feature = compute_gradient_histogram(num_angles, channel_patch_grad,
                                                                 channel_patch_theta)

                    if len(feature) == 0:
                        feature = channel_feature
                    else:
                        feature += channel_feature

            else:
                patch_grad = self.grad_mag[min_y: max_y, min_x: max_x]
                patch_theta = self.grad_theta[min_y: max_y, min_x: max_x]

                feature = compute_gradient_histogram(num_angles, patch_grad, patch_theta)

            feature = weight_gradient_histogram(feature, self.weight_coefficient)
            features.append(feature)

        key_points = key_points[key_points_flag, :]
        features = np.array(features)
        return key_points, features

    def find_corresponding_location_of_patch(self, patch_key_points, match):
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